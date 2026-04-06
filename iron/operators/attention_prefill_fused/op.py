# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
from ml_dtypes import bfloat16

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator, FusedFullELFCallable
from iron.common.utils import torch_to_numpy
from iron.operators.gemm.op import AIEGEMM
from iron.operators.rope.op import AIERope
from iron.operators.strided_copy.op import AIEStridedCopy
from iron.operators.repeat.op import AIERepeat
from iron.operators.softmax.op import AIESoftmax
from iron.operators.transpose.op import AIETranspose
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.elementwise_add.op import AIEElementwiseAdd


class AttentionPrefillFusedCallable:
    """Callable for the fused attention prefill operator.

    Usage:
        callable = op.get_callable()
        callable(input_buf, rope_angles_buf, output_buf)
    """

    def __init__(self, op):
        self.op = op

        self.fused_callable = FusedFullELFCallable(op.fused_op)

        # Load weight matrices into sub-buffers
        for name, weight in [
            ("W_query", op.W_query),
            ("W_key", op.W_key),
            ("W_value", op.W_value),
            ("W_output", op.W_output),
        ]:
            if weight is not None:
                try:
                    buf = self.fused_callable.get_buffer(name)
                    buf.view_as_np()[:] = torch_to_numpy(weight).flatten()
                except (KeyError, ValueError):
                    pass  # Buffer not in truncated runlist

        # Load scale factor
        if op.attn_scale_factor is not None:
            try:
                scale_buf = self.fused_callable.get_buffer("attn_scale_factor")
                scale_buf.view_as_np()[:] = torch_to_numpy(op.attn_scale_factor).flatten()
            except (KeyError, ValueError):
                pass

        # Load causal mask
        if op.causal_mask is not None:
            try:
                mask_buf = self.fused_callable.get_buffer("causal_mask")
                mask_buf.view_as_np()[:] = torch_to_numpy(op.causal_mask).flatten()
            except (KeyError, ValueError):
                pass

        # Sync input buffer (weights + scale + mask) to NPU
        self.fused_callable.input_buffer.to("npu")

    def __call__(self, input_buf, rope_angles_buf, output_buf):
        """Run prefill attention through the fused operator.

        Args:
            input_buf: AIEBuffer (S, E) — input token embeddings
            rope_angles_buf: AIEBuffer (S, d) — RoPE angles for all positions
            output_buf: AIEBuffer (S, E) — attention output (written in-place)
        """
        fc = self.fused_callable

        # Write input to sub-buffer
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
        try:
            rope_sub = fc.get_buffer("rope_angles")
            np.copyto(
                np.frombuffer(
                    rope_sub.memory_view,
                    dtype=rope_sub.dtype,
                    count=int(np.prod(rope_sub.shape)),
                ).reshape(rope_sub.shape),
                rope_angles_buf.view_as_np().flatten(),
            )
        except (KeyError, ValueError):
            pass  # Not in truncated runlist

        # Force-sync input buffer to NPU
        fc.input_buffer.on = "cpu"
        fc.input_buffer.to("npu")

        # Execute the fused operator
        fc()

        # Force-sync output buffer from NPU
        fc.output_buffer.on = "npu"
        fc.output_buffer.to("cpu")
        try:
            output_sub = fc.get_buffer("attn_output")
            out_np = np.frombuffer(
                output_sub.memory_view,
                dtype=output_sub.dtype,
                count=int(np.prod(output_sub.shape)),
            ).reshape(output_sub.shape)
            np.copyto(output_buf.view_as_np().reshape(out_np.shape), out_np)
        except (KeyError, ValueError):
            pass  # Not in truncated runlist


class AIEAttentionPrefillFused:
    """Single-dispatch fused attention prefill operator using FusedMLIROperator.

    Chains all operations in a single NPU dispatch:
      Q/K/V projections → RoPE → deinterleave → GQA broadcast →
      score computation → scaling → causal mask → softmax →
      context computation → re-interleave → output projection

    Parameters:
        num_heads (H): number of query attention heads
        num_kv_groups (G): number of KV heads (G=H for MHA, G<H for GQA)
        head_dim (d): dimension per head (must be 64)
        embedding_dim (E): model embedding dimension
        seq_len (S): sequence length for prefill
    """

    def __init__(
        self,
        num_heads,
        num_kv_groups,
        head_dim,
        embedding_dim,
        seq_len,
        context=None,
        max_runlist_entries=None,
    ):
        assert head_dim == 64, "head_dim must be 64 (hardware constraint)"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        assert seq_len % 256 == 0, "seq_len must be a multiple of 256 (GEMM tiling)"
        assert (num_heads * seq_len) % 16 == 0, "num_heads * seq_len must be a multiple of 16 (softmax)"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.context = context

        # Weights (set before calling get_callable())
        self.W_query = None          # (E, H*d)
        self.W_key = None            # (E, G*d)
        self.W_value = None          # (E, G*d)
        self.W_output = None         # (H*d, E)
        self.attn_scale_factor = None  # (H*S*S,) filled with 1/sqrt(d)
        self.causal_mask = None      # (H*S, S) with -inf for future positions

        self.fused_op = None
        self.max_runlist_entries = max_runlist_entries

    def compile(self):
        """Build the FusedMLIROperator and compile the ELF."""
        H, G, d, E, S = (
            self.num_heads, self.num_kv_groups, self.head_dim,
            self.embedding_dim, self.seq_len,
        )
        group_size = H // G

        elf_ctx = AIEContext()

        # ---- Sub-operators ----

        # Helper: pick tile_n that satisfies N % (tile_n * cols) == 0
        # and tile_n <= max_tile_n to fit in per-tile memory.
        def pick_tile_n(N, num_cols, max_tile_n=64):
            tile_n = N // num_cols
            while tile_n > max_tile_n:
                tile_n //= 2
            assert N % (tile_n * num_cols) == 0, (
                f"Cannot find valid tile_n for N={N}, num_cols={num_cols}"
            )
            return tile_n

        # 1. Q projection: (S, E) @ (E, H*d) → (S, H*d)
        gemm_query_op = AIEGEMM(
            M=S,
            K=E,
            N=H * d,
            num_aie_columns=8,
            tile_m=16,
            tile_k=64,
            tile_n=pick_tile_n(H * d, 8),
            context=elf_ctx,
        )

        # 2. K projection: (S, E) @ (E, G*d) → (S, G*d)
        gemm_kv_op = AIEGEMM(
            M=S,
            K=E,
            N=G * d,
            num_aie_columns=8,
            tile_m=16,
            tile_k=64,
            tile_n=pick_tile_n(G * d, 8),
            context=elf_ctx,
        )

        # 3. RoPE for queries: rows=S*H, cols=d, angle_rows=S
        # Input layout after Q proj: (S, H*d) = (S*H, d) with heads interleaved
        # Each group of H consecutive rows shares the same position → angle_rows=S
        rope_queries_op = AIERope(
            rows=S * H,
            cols=d,
            angle_rows=S,
            context=elf_ctx,
        )

        # 4. RoPE for keys: rows=S*G, cols=d, angle_rows=S
        rope_keys_op = AIERope(
            rows=S * G,
            cols=d,
            angle_rows=S,
            context=elf_ctx,
        )

        # 5. Deinterleave Q: (S*H, d) → (H*S, d) i.e. (S, H, d) → (H, S, d)
        # Input: row (pos*H + head) has query for position pos, head head
        # Output: row (head*S + pos) has query for head head, position pos
        deinterleave_q_op = AIEStridedCopy(
            input_sizes=(H, S, d),
            input_strides=(d, H * d, 1),
            input_offset=0,
            output_sizes=(H, S, d),
            output_strides=(S * d, d, 1),
            output_offset=0,
            input_buffer_size=S * H * d,
            output_buffer_size=H * S * d,
            transfer_size=S * d,
            num_aie_channels=1,
            context=elf_ctx,
        )

        # 6. Deinterleave K: (S*G, d) → (G*S, d) i.e. (S, G, d) → (G, S, d)
        deinterleave_k_op = AIEStridedCopy(
            input_sizes=(G, S, d),
            input_strides=(d, G * d, 1),
            input_offset=0,
            output_sizes=(G, S, d),
            output_strides=(S * d, d, 1),
            output_offset=0,
            input_buffer_size=S * G * d,
            output_buffer_size=G * S * d,
            transfer_size=S * d,
            num_aie_channels=1,
            context=elf_ctx,
        )

        # 7. Deinterleave V: (S*G, d) → (G*S, d) — same pattern as K
        # (shared op with K deinterleave - same dimensions)
        deinterleave_v_op = deinterleave_k_op

        # 8. Transpose K per group: (S, d) → (d, S)
        transpose_keys_op = AIETranspose(
            M=S,
            N=d,
            num_aie_columns=2,
            num_channels=1,
            m=256,
            n=32,
            s=8,
            context=elf_ctx,
        )

        # 9. Repeat keys for GQA: (G, d*S) → (H, d*S)
        repeat_keys_op = AIERepeat(
            rows=G,
            cols=d * S,
            repeat=group_size,
            transfer_size=d,
            context=elf_ctx,
        )

        # 10. Repeat values for GQA: (G, S*d) → (H, S*d)
        repeat_values_op = AIERepeat(
            rows=G,
            cols=S * d,
            repeat=group_size,
            transfer_size=d,
            context=elf_ctx,
        )

        # 11. Score GEMM per head: (S, d) @ (d, S) → (S, S)
        gemm_scores_op = AIEGEMM(
            M=S,
            K=d,
            N=S,
            num_aie_columns=8,
            tile_m=16,
            tile_k=64,
            tile_n=pick_tile_n(S, 8),
            context=elf_ctx,
        )

        # 12. Scale: elementwise multiply (H*S*S,)
        scale_op = AIEElementwiseMul(
            size=H * S * S,
            tile_size=S * S // 8,
            num_aie_columns=8,
            context=elf_ctx,
        )

        # 13. Causal mask: elementwise add (H*S*S,)
        mask_op = AIEElementwiseAdd(
            size=H * S * S,
            tile_size=S * S // 8,
            num_aie_columns=8,
            context=elf_ctx,
        )

        # 14. Softmax: rows=H*S, cols=S
        softmax_op = AIESoftmax(
            rows=H * S,
            cols=S,
            num_aie_columns=1,
            num_channels=1,
            rtp_vector_size=S,
            context=elf_ctx,
        )

        # 15. Context GEMM per head: (S, S) @ (S, d) → (S, d)
        gemm_context_op = AIEGEMM(
            M=S,
            K=S,
            N=d,
            num_aie_columns=4,
            tile_m=16,
            tile_k=64,
            tile_n=16,
            context=elf_ctx,
            prio_accuracy=True,
        )

        # 16. Re-interleave context: (H*S, d) → (S*H, d) i.e. (H, S, d) → (S, H, d)
        # Inverse of deinterleave_q: swap the strided pattern
        reinterleave_op = AIEStridedCopy(
            input_sizes=(H, S, d),
            input_strides=(S * d, d, 1),
            input_offset=0,
            output_sizes=(H, S, d),
            output_strides=(d, H * d, 1),
            output_offset=0,
            input_buffer_size=H * S * d,
            output_buffer_size=S * H * d,
            transfer_size=S * d,
            num_aie_channels=1,
            context=elf_ctx,
        )

        # 17. Output projection: (S, H*d) @ (H*d, E) → (S, E)
        gemm_output_op = AIEGEMM(
            M=S,
            K=H * d,
            N=E,
            num_aie_columns=8,
            tile_m=16,
            tile_k=64,
            tile_n=pick_tile_n(E, 8),
            context=elf_ctx,
            prio_accuracy=True,
        )

        # ---- Runlist ----
        bytes_per_elem = 2  # bf16
        query_head_bytes = S * d * bytes_per_elem
        kv_group_bytes_dS = d * S * bytes_per_elem  # (d, S) per group after transpose
        kv_group_bytes_Sd = S * d * bytes_per_elem   # (S, d) per group
        scores_head_bytes = S * S * bytes_per_elem
        context_head_bytes = S * d * bytes_per_elem

        runlist = [
            # Q/K/V projections
            (gemm_query_op, "input", "W_query", "queries_raw"),
            (gemm_kv_op, "input", "W_key", "keys_raw"),
            (gemm_kv_op, "input", "W_value", "values_raw"),
            # RoPE
            (rope_queries_op, "queries_raw", "rope_angles", "queries_roped"),
            (rope_keys_op, "keys_raw", "rope_angles", "keys_roped"),
            # Deinterleave
            (deinterleave_q_op, "queries_roped", "queries_deint"),
            (deinterleave_k_op, "keys_roped", "keys_deint"),
            (deinterleave_v_op, "values_raw", "values_deint"),
            # Transpose keys per group
            *[
                (
                    transpose_keys_op,
                    f"keys_deint[{g * kv_group_bytes_Sd}:{(g + 1) * kv_group_bytes_Sd}]",
                    f"keys_transposed[{g * kv_group_bytes_dS}:{(g + 1) * kv_group_bytes_dS}]",
                )
                for g in range(G)
            ],
            # GQA repeat (keys and values)
            (repeat_keys_op, "keys_transposed", "keys_for_scores"),
            (repeat_values_op, "values_deint", "values_for_context"),
            # Score GEMM per head: Q_head(S, d) @ K_head(d, S) → scores(S, S)
            *[
                (
                    gemm_scores_op,
                    f"queries_deint[{h * query_head_bytes}:{(h + 1) * query_head_bytes}]",
                    f"keys_for_scores[{h * kv_group_bytes_dS}:{(h + 1) * kv_group_bytes_dS}]",
                    f"attn_scores[{h * scores_head_bytes}:{(h + 1) * scores_head_bytes}]",
                )
                for h in range(H)
            ],
            # Scale
            (scale_op, "attn_scores", "attn_scale_factor", "attn_scores"),
            # Causal mask
            (mask_op, "attn_scores", "causal_mask", "attn_scores_masked"),
            # Softmax
            (softmax_op, "attn_scores_masked", "attn_weights"),
            # Context GEMM per head: weights(S, S) @ values(S, d) → context(S, d)
            *[
                (
                    gemm_context_op,
                    f"attn_weights[{h * scores_head_bytes}:{(h + 1) * scores_head_bytes}]",
                    f"values_for_context[{h * kv_group_bytes_Sd}:{(h + 1) * kv_group_bytes_Sd}]",
                    f"attn_context[{h * context_head_bytes}:{(h + 1) * context_head_bytes}]",
                )
                for h in range(H)
            ],
            # Re-interleave context: (H, S, d) → (S, H, d) = (S, H*d)
            (reinterleave_op, "attn_context", "context_interleaved"),
            # Output projection
            (gemm_output_op, "context_interleaved", "W_output", "attn_output"),
        ]

        # ---- Explicit buffer sizes ----
        buffer_sizes = {
            "queries_raw": S * H * d * bytes_per_elem,
            "queries_roped": S * H * d * bytes_per_elem,
            "queries_deint": H * S * d * bytes_per_elem,
            "keys_raw": S * G * d * bytes_per_elem,
            "keys_roped": S * G * d * bytes_per_elem,
            "keys_deint": G * S * d * bytes_per_elem,
            "keys_transposed": G * d * S * bytes_per_elem,
            "keys_for_scores": H * d * S * bytes_per_elem,
            "values_raw": S * G * d * bytes_per_elem,
            "values_deint": G * S * d * bytes_per_elem,
            "values_for_context": H * S * d * bytes_per_elem,
            "attn_scores": H * S * S * bytes_per_elem,
            "attn_scores_masked": H * S * S * bytes_per_elem,
            "attn_weights": H * S * S * bytes_per_elem,
            "attn_context": H * S * d * bytes_per_elem,
            "context_interleaved": S * H * d * bytes_per_elem,
        }

        # Truncate runlist for incremental debugging
        if self.max_runlist_entries is not None:
            runlist = runlist[: self.max_runlist_entries]

        # Collect buffer names actually used in the (possibly truncated) runlist
        used_bufs = set()
        for entry in runlist:
            for arg in entry[1:]:
                base = arg.split("[")[0]
                used_bufs.add(base)

        all_input_args = [
            "input", "rope_angles", "W_query", "W_key", "W_value",
            "W_output", "attn_scale_factor", "causal_mask",
        ]
        all_output_args = ["attn_output"]

        active_input_args = [a for a in all_input_args if a in used_bufs]
        active_output_args = [a for a in all_output_args if a in used_bufs]

        # Filter buffer_sizes to only include buffers referenced in runlist
        active_buffer_sizes = {
            k: v for k, v in buffer_sizes.items() if k in used_bufs
        }

        self.fused_op = FusedMLIROperator(
            "attention_prefill_fused",
            runlist,
            input_args=active_input_args,
            output_args=active_output_args,
            buffer_sizes=active_buffer_sizes,
            context=elf_ctx,
        ).compile()

        return self

    def get_callable(self):
        """Return a callable that runs the fused attention operator.

        Weights (W_query, W_key, W_value, W_output, attn_scale_factor,
        causal_mask) must be set on the operator before calling this method.
        """
        return AttentionPrefillFusedCallable(self)
