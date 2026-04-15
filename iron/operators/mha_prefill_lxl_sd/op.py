# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A layer-by-layer (LxL) single-dispatch (SD) implementation of multi-head attention (MHA).
"""

import aie.utils as aie_utils

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.gemm.op import GEMM
from iron.operators.rope.op import RoPE
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.repeat.op import Repeat
from iron.operators.softmax.op import Softmax
from iron.operators.transpose.op import Transpose
from iron.operators.elementwise_mul.op import ElementwiseMul
from iron.operators.elementwise_add.op import ElementwiseAdd


def _pick_tile_n(N, num_cols, max_tile_n=64):
    tile_n = N // num_cols
    while tile_n > max_tile_n:
        tile_n //= 2
    assert N % (tile_n * num_cols) == 0
    return tile_n


def _build_core_ops(H, G, d, S, elf_ctx, causal_mask=True, num_cols=None):
    """Build core attention sub-ops and runlist (no projections/RoPE/GQA).

    Expects pre-processed inputs:
      queries: (H, S, d) deinterleaved, contiguous per head
      keys: (H, d, S) transposed and GQA-repeated
      values: (H, S, d) GQA-repeated

    Produces:
      attn_context: (H, S, d) — per-head context vectors

    If causal_mask=False, the elementwise-add masking step is omitted.
    """
    if num_cols is None:
        num_cols = aie_utils.get_current_device().cols
    B = 2  # bytes per bf16 element

    gemm_scores = GEMM(
        M=S,
        K=d,
        N=S,
        num_aie_columns=num_cols,
        tile_m=16,
        tile_k=64,
        tile_n=_pick_tile_n(S, num_cols),
        context=elf_ctx,
    )
    scale = ElementwiseMul(
        size=H * S * S,
        tile_size=S * S // num_cols,
        num_aie_columns=num_cols,
        context=elf_ctx,
    )
    if causal_mask:
        mask = ElementwiseAdd(
            size=H * S * S,
            tile_size=S * S // num_cols,
            num_aie_columns=num_cols,
            context=elf_ctx,
        )
    # Use online/partial softmax when full-row tiles would exhaust AIE local
    # memory (each double-buffered FIFO pair uses 4 * tile_size bytes; at
    # S >= 8192 the in+out FIFOs alone consume the full 64 KB data memory).
    softmax_chunk_size = 1024 if S >= 8192 else None
    softmax = Softmax(
        rows=H * S,
        cols=S,
        num_aie_columns=1,
        num_channels=1,
        rtp_vector_size=S,
        chunk_size=softmax_chunk_size,
        context=elf_ctx,
    )
    gemm_context = GEMM(
        M=S,
        K=S,
        N=d,
        num_aie_columns=min(4, num_cols),
        tile_m=16,
        tile_k=64,
        tile_n=_pick_tile_n(d, min(4, num_cols)),
        context=elf_ctx,
        prio_accuracy=True,
    )

    qh = S * d * B
    kdS = d * S * B
    kSd = S * d * B
    sh = S * S * B
    ch = S * d * B

    runlist = [
        *[
            (
                gemm_scores,
                f"queries[{h*qh}:{(h+1)*qh}]",
                f"keys[{h*kdS}:{(h+1)*kdS}]",
                f"attn_scores[{h*sh}:{(h+1)*sh}]",
            )
            for h in range(H)
        ],
        (scale, "attn_scores", "attn_scale_factor", "attn_scores_scaled"),
    ]

    if causal_mask:
        runlist += [
            (mask, "attn_scores_scaled", "causal_mask", "attn_scores_masked"),
            (softmax, "attn_scores_masked", "attn_weights"),
        ]
    else:
        runlist += [
            (softmax, "attn_scores_scaled", "attn_weights"),
        ]

    runlist += [
        *[
            (
                gemm_context,
                f"attn_weights[{h*sh}:{(h+1)*sh}]",
                f"values[{h*kSd}:{(h+1)*kSd}]",
                f"attn_context[{h*ch}:{(h+1)*ch}]",
            )
            for h in range(H)
        ],
    ]

    buffer_sizes = {
        "queries": H * S * d * B,
        "keys": H * d * S * B,
        "values": H * S * d * B,
        "attn_scores": H * S * S * B,
        "attn_scores_scaled": H * S * S * B,
        "attn_weights": H * S * S * B,
        "attn_context": H * S * d * B,
    }
    if causal_mask:
        buffer_sizes["attn_scores_masked"] = H * S * S * B

    return runlist, buffer_sizes


class AttentionPrefillFused(FusedMLIROperator):
    """Fused attention prefill (core, no projections/RoPE).

    Accepts pre-projected Q (S*H,d), K (S*G,d), V (S*G,d) in interleaved layout.
    """

    def __init__(
        self,
        num_heads,
        num_kv_groups,
        head_dim,
        embedding_dim,
        seq_len,
        causal_mask=True,
        context=None,
        dispatch="auto",
    ):
        assert head_dim == 64
        assert num_heads % num_kv_groups == 0
        assert seq_len % 256 == 0
        assert (num_heads * seq_len) % 16 == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        elf_ctx = context or AIEContext()
        runlist, buffer_sizes = _build_core_ops(
            num_heads,
            num_kv_groups,
            head_dim,
            seq_len,
            elf_ctx,
            causal_mask=causal_mask,
        )

        mask_suffix = "_causal" if causal_mask else "_nomask"
        input_args = ["queries", "keys", "values", "attn_scale_factor"]
        if causal_mask:
            input_args.append("causal_mask")

        super().__init__(
            name=f"attention_prefill_fused_{num_heads}h{num_kv_groups}g{head_dim}d{embedding_dim}e{seq_len}s{mask_suffix}",
            runlist=runlist,
            input_args=input_args,
            output_args=["attn_context"],
            buffer_sizes=buffer_sizes,
            dispatch=dispatch,
            context=elf_ctx,
        )


class AttentionPrefillProjectedFused(FusedMLIROperator):
    """Fused attention prefill with Q/K/V projections and RoPE.

    Accepts raw input (S, E) and rope_angles (S, d).
    """

    def __init__(
        self,
        num_heads,
        num_kv_groups,
        head_dim,
        embedding_dim,
        seq_len,
        causal_mask=True,
        context=None,
        dispatch="auto",
    ):
        assert head_dim == 64
        assert num_heads % num_kv_groups == 0
        assert seq_len % 256 == 0
        assert (num_heads * seq_len) % 16 == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self._dispatch_arg = dispatch

        H, G, d, E, S = num_heads, num_kv_groups, head_dim, embedding_dim, seq_len
        group_size = H // G
        B = 2
        num_cols = aie_utils.get_current_device().cols

        elf_ctx = context or AIEContext()

        # ---- Projection + RoPE ----
        gemm_query = GEMM(
            M=S,
            K=E,
            N=H * d,
            num_aie_columns=num_cols,
            tile_m=16,
            tile_k=64,
            tile_n=_pick_tile_n(H * d, num_cols),
            context=elf_ctx,
        )
        gemm_kv = GEMM(
            M=S,
            K=E,
            N=G * d,
            num_aie_columns=num_cols,
            tile_m=16,
            tile_k=64,
            tile_n=_pick_tile_n(G * d, num_cols),
            context=elf_ctx,
        )
        rope_queries = RoPE(rows=S * H, cols=d, angle_rows=S, context=elf_ctx)
        rope_keys = RoPE(rows=S * G, cols=d, angle_rows=S, context=elf_ctx)

        # ---- Deinterleave ----
        deinterleave_q = StridedCopy(
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
        deinterleave_kv = StridedCopy(
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

        # ---- Transpose keys + GQA repeat ----
        transpose_keys = Transpose(
            M=S,
            N=d,
            num_aie_columns=2,
            num_channels=1,
            m=256,
            n=32,
            s=8,
            context=elf_ctx,
        )
        repeat_kv = Repeat(
            rows=G,
            cols=d * S,
            repeat=group_size,
            transfer_size=d,
            context=elf_ctx,
        )

        kSd = S * d * B
        kdS = d * S * B

        prefix_runlist = [
            (gemm_query, "input", "W_query", "queries_projected"),
            (gemm_kv, "input", "W_key", "keys_projected"),
            (gemm_kv, "input", "W_value", "values_projected"),
            (rope_queries, "queries_projected", "rope_angles", "queries_roped"),
            (rope_keys, "keys_projected", "rope_angles", "keys_roped"),
            (deinterleave_q, "queries_roped", "queries"),
            (deinterleave_kv, "keys_roped", "keys_deint"),
            (deinterleave_kv, "values_projected", "values_deint"),
            *[
                (
                    transpose_keys,
                    f"keys_deint[{g*kSd}:{(g+1)*kSd}]",
                    f"keys_transposed[{g*kdS}:{(g+1)*kdS}]",
                )
                for g in range(G)
            ],
            (repeat_kv, "keys_transposed", "keys"),
            (repeat_kv, "values_deint", "values"),
        ]
        prefix_buffer_sizes = {
            "queries_projected": S * H * d * B,
            "keys_projected": S * G * d * B,
            "values_projected": S * G * d * B,
            "queries_roped": S * H * d * B,
            "keys_roped": S * G * d * B,
            "keys_deint": G * S * d * B,
            "values_deint": G * S * d * B,
            "keys_transposed": G * d * S * B,
        }

        core_runlist, core_buffer_sizes = _build_core_ops(
            H,
            G,
            d,
            S,
            elf_ctx,
            causal_mask=causal_mask,
            num_cols=num_cols,
        )

        # ---- Reinterleave + output projection ----
        reinterleave = StridedCopy(
            input_sizes=(1, 1, 1, H * S * d),
            input_strides=(0, 0, 0, 1),
            input_offset=0,
            output_sizes=(H, 256, S // 256, d),
            output_strides=(d, 256 * H * d, H * d, 1),
            output_offset=0,
            input_buffer_size=H * S * d,
            output_buffer_size=S * H * d,
            transfer_size=S * d,
            num_aie_channels=1,
            context=elf_ctx,
        )
        gemm_output = GEMM(
            M=S,
            K=H * d,
            N=E,
            num_aie_columns=num_cols,
            tile_m=16,
            tile_k=64,
            tile_n=_pick_tile_n(E, num_cols),
            context=elf_ctx,
            prio_accuracy=True,
        )

        suffix_runlist = [
            (reinterleave, "attn_context", "context_interleaved"),
            (gemm_output, "context_interleaved", "W_output", "attn_output"),
        ]
        suffix_buffer_sizes = {
            "context_interleaved": S * H * d * B,
        }

        mask_suffix = "_causal" if causal_mask else "_nomask"
        input_args = [
            "input",
            "rope_angles",
            "W_query",
            "W_key",
            "W_value",
            "W_output",
            "attn_scale_factor",
        ]
        if causal_mask:
            input_args.append("causal_mask")

        super().__init__(
            name=f"attention_prefill_projected_fused_{H}h{G}g{d}d{E}e{S}s{mask_suffix}",
            runlist=prefix_runlist + core_runlist + suffix_runlist,
            input_args=input_args,
            output_args=["attn_output"],
            buffer_sizes={
                **prefix_buffer_sizes,
                **core_buffer_sizes,
                **suffix_buffer_sizes,
            },
            dispatch=dispatch,
            context=elf_ctx,
        )
