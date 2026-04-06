# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
from ml_dtypes import bfloat16

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator, FusedFullELFCallable, load_elf, patch_elf
from iron.common.utils import torch_to_numpy
from iron.operators.gemv.op import AIEGEMV
from iron.operators.rope.op import AIERope
from iron.operators.strided_copy.op import AIEStridedCopy
from iron.operators.repeat.op import AIERepeat
from iron.operators.softmax.op import AIESoftmax
from iron.operators.transpose.op import AIETranspose
from iron.operators.elementwise_mul.op import AIEElementwiseMul


def _get_patch_locs(elf_data, magic):
    """Find all 32-bit word locations in the ELF matching the magic value."""
    magic = np.uint32(magic & 0xFFFFFFFF)
    return np.where(elf_data == magic)[0]


class AttentionDecodeFusedCallable:
    """Callable for the fused attention decode operator.

    Usage:
        callable = op.get_callable()
        callable.set_token_position(t)    # before each token
        callable(input_buf, rope_angles_buf, output_buf)
    """

    def __init__(self, op):
        self.op = op
        H, G, d, E, S = op.num_heads, op.num_kv_groups, op.head_dim, op.embedding_dim, op.max_seq_len

        # Load base ELF and find patch locations
        self.elf_data_base = load_elf(op.fused_op)

        # Identify strided_copy patch locations in the ELF
        # The strided_copy operator embeds: keys_cache_base_offset + MAGIC * 2
        # and values_cache_base_offset + MAGIC * 2
        _, keys_cache_offs, _ = op.fused_op.get_layout_for_buffer("keys_cache")
        _, values_cache_offs, _ = op.fused_op.get_layout_for_buffer("values_cache")
        magic = op.STRIDED_COPY_MAGIC

        keys_locs = _get_patch_locs(
            self.elf_data_base,
            (keys_cache_offs + magic * 2) & 0xFFFFFFFF,
        )
        values_locs = _get_patch_locs(
            self.elf_data_base,
            (values_cache_offs + magic * 2) & 0xFFFFFFFF,
        )
        # Also find zero-base patches (no offset added in ELF)
        no_offset_locs = _get_patch_locs(
            self.elf_data_base,
            (magic * 2) & 0xFFFFFFFF,
        )

        # patch_locations: {elf_word_index: base_offset_bytes}
        self.keys_patch_locs = {int(l): keys_cache_offs for l in keys_locs}
        self.values_patch_locs = {int(l): values_cache_offs for l in values_locs}
        self.no_offset_patch_locs = {int(l): 0 for l in no_offset_locs}

        # Softmax patch locations
        self.softmax_patch_locs = _get_patch_locs(
            self.elf_data_base, op.SOFTMAX_MAGIC
        )

        # Start with the base ELF (patched for position 0)
        self.fused_callable = FusedFullELFCallable(
            op.fused_op, elf_data=self.elf_data_base.copy()
        )
        self.elf_data = self.fused_callable.elf_data if hasattr(self.fused_callable, 'elf_data') else None

        # Load weight matrices into sub-buffers
        for name, weight in [
            ("W_query", op.W_query),
            ("W_key", op.W_key),
            ("W_value", op.W_value),
            ("W_output", op.W_output),
        ]:
            if weight is not None:
                buf = self.fused_callable.get_buffer(name)
                buf.view_as_np()[:] = torch_to_numpy(weight).flatten()

        # Load scale factor
        if op.attn_scale_factor is not None:
            scale_buf = self.fused_callable.get_buffer("attn_scale_factor")
            scale_buf.view_as_np()[:] = torch_to_numpy(op.attn_scale_factor).flatten()

        # Pre-fill KV caches into scratch buffer if provided
        if op.keys_cache_init is not None:
            keys_buf = self.fused_callable.get_buffer("keys_cache")
            keys_buf.view_as_np()[:] = torch_to_numpy(op.keys_cache_init).flatten()
        if op.values_cache_init is not None:
            values_buf = self.fused_callable.get_buffer("values_cache")
            values_buf.view_as_np()[:] = torch_to_numpy(op.values_cache_init).flatten()

        # Sync input buffer (weights + scale) and scratch buffer (KV cache) to NPU
        self.fused_callable.input_buffer.to("npu")
        self.fused_callable.scratch_buffer.to("npu")

    def set_token_position(self, position):
        """Patch the ELF for the given token position.

        Updates strided_copy output offsets and softmax mask to match
        the current token position in the KV cache.

        Must be called before each token if position != max_seq_len - 1.
        """
        d = self.op.head_dim
        # Byte offset into the KV cache for this token position
        # Each position stores G*d bfloat16 values; offset is per-position
        offset_val = position * d * 2  # byte offset

        # Build strided_copy patches (keys and values only; skip no_offset)
        patches = {}
        for elf_idx, base_offs in self.keys_patch_locs.items():
            patches[elf_idx] = (base_offs + offset_val, 0xFFFFFFFF)
        for elf_idx, base_offs in self.values_patch_locs.items():
            patches[elf_idx] = (base_offs + offset_val, 0xFFFFFFFF)
        # Note: no_offset_patch_locs are NOT patched - they appear to be input DMA
        # descriptors for the strided_copy that should not be modified

        # Softmax mask: set rtp_vector_size to context_len = position + 1
        context_len = position + 1
        for elf_idx in self.softmax_patch_locs:
            patches[int(elf_idx)] = (context_len, 0xFFFFFFFF)

        # Copy base ELF, apply patches, reload
        patched = self.elf_data_base.copy()
        patch_elf(patched, patches)
        self.fused_callable.reload_elf(patched)

    def __call__(self, input_buf, rope_angles_buf, output_buf):
        """Run one decode token through the fused attention operator.

        Args:
            input_buf: AIEBuffer (E,) — input token embedding
            rope_angles_buf: AIEBuffer (1, d) — RoPE angles for current position
            output_buf: AIEBuffer (E,) — attention output (written in-place)
        """
        fc = self.fused_callable

        # Write input to sub-buffer using direct memory view (avoids resetting `on` state)
        input_sub = fc.get_buffer("input")
        np.copyto(
            np.frombuffer(
                input_sub.memory_view,
                dtype=input_sub.dtype,
                count=int(np.prod(input_sub.shape)),
            ).reshape(input_sub.shape),
            input_buf.view_as_np().flatten(),
        )

        # Write rope_angles to sub-buffer
        rope_sub = fc.get_buffer("rope_angles")
        np.copyto(
            np.frombuffer(
                rope_sub.memory_view,
                dtype=rope_sub.dtype,
                count=int(np.prod(rope_sub.shape)),
            ).reshape(rope_sub.shape),
            rope_angles_buf.view_as_np().flatten(),
        )

        # Force-sync input buffer to NPU
        fc.input_buffer.on = "cpu"
        fc.input_buffer.to("npu")

        # Execute the fused operator
        fc()

        # Force-sync output buffer from NPU
        fc.output_buffer.on = "npu"
        fc.output_buffer.to("cpu")
        output_sub = fc.get_buffer("attn_output")
        out_np = np.frombuffer(
            output_sub.memory_view,
            dtype=output_sub.dtype,
            count=int(np.prod(output_sub.shape)),
        ).reshape(output_sub.shape)
        np.copyto(output_buf.view_as_np().reshape(out_np.shape), out_np)


class AIEAttentionDecodeFused:
    """Single-dispatch fused attention decode operator using FusedMLIROperator.

    Chains all operations in a single NPU dispatch:
      Q/K/V projections → RoPE → KV cache update → GQA broadcast →
      score computation → scaling → softmax → value transpose →
      context computation → output projection

    Parameters:
        num_heads (H): number of query attention heads
        num_kv_groups (G): number of KV heads (G=H for MHA, G<H for GQA)
        head_dim (d): dimension per head (must be 64)
        embedding_dim (E): model embedding dimension
        max_seq_len (S): maximum sequence length (KV cache size);
            must be a multiple of 256 (GEMV tiling) and 16 (softmax)
    """

    # Magic value embedded in ELF for strided_copy output offset patching
    STRIDED_COPY_MAGIC = 0xDEADBEE0
    # Magic value embedded in ELF for softmax mask patching
    SOFTMAX_MAGIC = 0xBA5EBA11

    def __init__(
        self,
        num_heads,
        num_kv_groups,
        head_dim,
        embedding_dim,
        max_seq_len,
        context=None,
    ):
        assert head_dim == 64, "head_dim must be 64 (hardware constraint)"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        assert num_heads % 16 == 0, "num_heads must be a multiple of 16 (softmax rows constraint)"
        assert max_seq_len % 256 == 0, "max_seq_len must be a multiple of 256"
        assert max_seq_len % 16 == 0, "max_seq_len must be divisible by 16 (softmax)"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.context = context

        # Weights (set before calling get_callable())
        self.W_query = None          # (H*d, E)
        self.W_key = None            # (G*d, E)
        self.W_value = None          # (G*d, E)
        self.W_output = None         # (E, H*d)
        self.attn_scale_factor = None  # (H*S,) filled with 1/sqrt(d)
        self.keys_cache_init = None  # (G*S*d,) initial KV cache (optional)
        self.values_cache_init = None  # (G*S*d,) initial KV cache (optional)

        self.fused_op = None

    def compile(self):
        """Build the FusedMLIROperator and compile the ELF."""
        H, G, d, E, S = (
            self.num_heads, self.num_kv_groups, self.head_dim,
            self.embedding_dim, self.max_seq_len,
        )
        group_size = H // G

        elf_ctx = AIEContext()

        # ---- Sub-operators ----

        # 1. Q projection: (H*d, E) @ (E,) → (H*d,)
        gemv_query_op = AIEGEMV(
            M=H * d,
            K=E,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=d // 2,
            context=elf_ctx,
        )

        # 2. K/V projection: (G*d, E) @ (E,) → (G*d,)  — shared for K and V
        gemv_kv_op = AIEGEMV(
            M=G * d,
            K=E,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=d // 2,
            context=elf_ctx,
        )

        # 3. RoPE for queries: rows=H, cols=d, angle_rows=1
        rope_queries_op = AIERope(
            rows=H,
            cols=d,
            angle_rows=1,
            context=elf_ctx,
        )

        # 4. RoPE for keys: rows=G, cols=d, angle_rows=1
        rope_keys_op = AIERope(
            rows=G,
            cols=d,
            angle_rows=1,
            context=elf_ctx,
        )

        # 5. StridedCopy for KV cache update — shared for K and V
        # Copies G rows of d elements into the cache at (G, S, d) output
        strided_copy_op = AIEStridedCopy(
            input_sizes=(G, d),
            input_strides=(d, 1),
            input_offset=0,
            output_sizes=(1, G, d),
            output_strides=(0, S * d, 1),
            output_offset=7 * d * 2,  # placeholder value; patched at runtime
            input_buffer_size=1 * G * d,
            output_buffer_size=G * S * d,
            num_aie_channels=1,
            output_offset_patch_marker=self.STRIDED_COPY_MAGIC,
            context=elf_ctx,
        )

        # 6. Repeat for GQA broadcast — shared for K and V
        # Input: (G, S*d) → Output: (H, S*d)
        repeat_op = AIERepeat(
            rows=G,
            cols=S * d,
            repeat=group_size,
            transfer_size=d,
            context=elf_ctx,
        )

        # 7. GEMV for attention scores: batched (H, S, d) @ (H, d) → (H, S)
        gemv_scores_op = AIEGEMV(
            M=S,
            K=d,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=S // 8,
            num_batches=H,
            context=elf_ctx,
        )

        # 8. Scale: element-wise multiply (H*S,) × (H*S,)
        scale_op = AIEElementwiseMul(
            size=H * S,
            tile_size=S // 8,
            num_aie_columns=8,
            context=elf_ctx,
        )

        # 9. Softmax over attention scores: (H*S,) → (H*S,)
        softmax_op = AIESoftmax(
            rows=H,
            cols=S,
            num_aie_columns=1,
            num_channels=1,
            rtp_vector_size=S,
            mask_patch_value=self.SOFTMAX_MAGIC,
            context=elf_ctx,
        )

        # 10. Transpose values per head: (S, d) → (d, S) — one instance, used H times
        transpose_values_op = AIETranspose(
            M=S,
            N=d,
            num_aie_columns=2,
            num_channels=1,
            m=256,
            n=32,
            s=8,
            context=elf_ctx,
        )

        # 11. GEMV for context: batched (H, d, S) @ (H, S) → (H, d)
        gemv_context_op = AIEGEMV(
            M=d,
            K=S,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=4,
            num_batches=H,
            context=elf_ctx,
        )

        # 12. Output projection: (E, H*d) @ (H*d,) → (E,)
        gemv_output_op = AIEGEMV(
            M=E,
            K=H * d,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=E // 8,
            context=elf_ctx,
        )

        # ---- Runlist ----
        # Slice indices in the FusedMLIROperator are BYTE offsets within the parent buffer
        values_per_head_bytes = S * d * 2  # bytes per head (bf16 = 2 bytes/element)

        runlist = [
            # Q/K/V projections
            (gemv_query_op, "W_query", "input", "queries"),
            (gemv_kv_op, "W_key", "input", "keys"),
            (gemv_kv_op, "W_value", "input", "values"),
            # RoPE
            (rope_queries_op, "queries", "rope_angles", "queries"),
            (rope_keys_op, "keys", "rope_angles", "keys"),
            # KV cache update
            (strided_copy_op, "keys", "keys_cache"),
            (strided_copy_op, "values", "values_cache"),
            # GQA broadcast
            (repeat_op, "keys_cache", "attn_scores_keys"),
            (repeat_op, "values_cache", "attn_scores_values"),
            # Score computation
            (gemv_scores_op, "attn_scores_keys", "queries", "attn_scores"),
            # Scale
            (scale_op, "attn_scores", "attn_scale_factor", "attn_scores"),
            # Softmax
            (softmax_op, "attn_scores", "attn_weights"),
            # Transpose values per head (H ops, one per head, using byte-offset slice notation)
            *[
                (
                    transpose_values_op,
                    f"attn_scores_values[{h * values_per_head_bytes}:{(h + 1) * values_per_head_bytes}]",
                    f"attn_scores_values_T[{h * values_per_head_bytes}:{(h + 1) * values_per_head_bytes}]",
                )
                for h in range(H)
            ],
            # Context computation
            (gemv_context_op, "attn_scores_values_T", "attn_weights", "attn_context"),
            # Output projection
            (gemv_output_op, "W_output", "attn_context", "attn_output"),
        ]

        # ---- Explicit buffer sizes ----
        # KV caches and score buffers are larger than any single op would infer
        cache_buffer_size = G * S * d * 2       # bytes: G groups × S positions × d dims × 2 bytes
        scores_buffer_size = H * S * d * 2      # bytes: H heads × S positions × d dims × 2 bytes

        buffer_sizes = {
            "keys_cache": cache_buffer_size,
            "values_cache": cache_buffer_size,
            "attn_scores_keys": scores_buffer_size,
            "attn_scores_values": scores_buffer_size,
            "attn_scores_values_T": scores_buffer_size,
        }

        self.fused_op = FusedMLIROperator(
            "attention_decode_fused",
            runlist,
            input_args=[
                "input",
                "rope_angles",
                "W_query",
                "W_key",
                "W_value",
                "W_output",
                "attn_scale_factor",
            ],
            output_args=["attn_output"],
            buffer_sizes=buffer_sizes,
            context=elf_ctx,
        ).compile()

        return self

    def get_callable(self):
        """Return a callable that runs the fused attention operator.

        Weights (W_query, W_key, W_value, W_output, attn_scale_factor) must
        be set on the operator before calling this method.

        Call set_token_position(t) on the returned callable before each token.
        """
        return AttentionDecodeFusedCallable(self)
