# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from iron.common.test_utils import verify_buffer
from iron.common.utils import torch_to_numpy

from iron.operators.mha_prefill_lxl_sd.op import (
    AIEAttentionPrefillFused,
    AIEAttentionPrefillProjectedFused,
)
from iron.operators.mha_prefill_lxl_sd.reference import generate_golden_reference

REL_TOL = 0.08
ABS_TOL = 2.0
MAX_ERROR_RATE = 0.03


def get_params():
    return [
        pytest.param(2, 2, 64, 256, 256, id="H2"),
        pytest.param(32, 8, 64, 2048, 256, id="H32"),
    ]


def _load_input(fc, name, tensor):
    """Load a tensor into a named sub-buffer of the fused callable."""
    fc.get_buffer(name).view_as_np()[:] = torch_to_numpy(tensor).flatten()


def _get_scratch_tensor(fc, name, shape):
    """Read a named buffer from the fused callable's scratch space."""
    fc.scratch_buffer.on = "npu"
    fc.scratch_buffer.to("cpu")
    sub = fc.get_buffer(name)
    return np.frombuffer(
        sub.memory_view, dtype=bfloat16, count=int(np.prod(shape))
    ).reshape(shape).astype(np.float32)


def _verify_output(fc, golden, H, d, S, E):
    """Chain-consistent output verification shared by both test variants."""
    npu_context = torch.from_numpy(
        _get_scratch_tensor(fc, "context_interleaved", (S, H * d))
    ).bfloat16()
    chain_ref = (npu_context.float() @ golden["W_output"].float()).to(torch.bfloat16)

    fc.output_buffer.on = "npu"
    fc.output_buffer.to("cpu")
    output_np = fc.get_buffer("attn_output").view_as_np()
    output = torch.from_numpy(output_np.reshape(S, E).astype(np.float32)).bfloat16()

    errors = verify_buffer(
        output, "attn_output", chain_ref.reshape(S, E),
        rel_tol=REL_TOL, abs_tol=ABS_TOL, max_error_rate=MAX_ERROR_RATE,
    )
    assert not errors, f"Output verification failed with {len(errors)} errors"


def _core_gemm_flops(H, G, d, E, S):
    """Count GEMM FLOPs for the core attention operator."""
    score_flops = H * 2 * S * d * S       # H x (S,d)@(d,S)
    context_flops = H * 2 * S * S * d     # H x (S,S)@(S,d)
    output_flops = 2 * S * (H * d) * E    # (S,H*d)@(H*d,E)
    return score_flops + context_flops + output_flops


def _projected_gemm_flops(H, G, d, E, S):
    """Count GEMM FLOPs for the projected attention operator."""
    query_proj = 2 * S * E * (H * d)      # (S,E)@(E,H*d)
    kv_proj = 2 * (2 * S * E * (G * d))   # key + value: (S,E)@(E,G*d) each
    return query_proj + kv_proj + _core_gemm_flops(H, G, d, E, S)


def _print_metrics(label, elapsed_s, flops):
    """Print latency and throughput metrics."""
    gflops = flops / elapsed_s / 1e9
    print(f"  {label}: {elapsed_s*1e3:.2f} ms, {gflops:.2f} GFLOPS")


# ---------------------------------------------------------------------------
# Core attention tests (pre-projected Q, K, V)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H,G,d,E,S", get_params())
def test_mha_pefill_lxl_sd(H, G, d, E, S):
    """Core attention: score GEMM -> scale -> mask -> softmax -> context GEMM -> output."""
    golden = generate_golden_reference(H, G, d, E, S)

    op = AIEAttentionPrefillFused(H, G, d, E, S)
    op.compile()
    fc = op.get_callable()

    _load_input(fc, "queries", golden["queries_deinterleaved"])
    _load_input(fc, "keys", golden["keys_for_scores"])
    _load_input(fc, "values", golden["values_for_context"])
    _load_input(fc, "W_output", golden["W_output"])
    _load_input(fc, "attn_scale_factor", golden["attn_scale_factor"])
    _load_input(fc, "causal_mask", golden["causal_mask"])

    t0 = time.perf_counter()
    fc()
    elapsed = time.perf_counter() - t0
    _print_metrics("core", elapsed, _core_gemm_flops(H, G, d, E, S))

    _verify_output(fc, golden, H, d, S, E)


# ---------------------------------------------------------------------------
# Projected attention tests (with Q/K/V projections + RoPE)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H,G,d,E,S", get_params())
def test_attention_prefill_projected_fused(H, G, d, E, S):
    """Projected attention: Q/K/V proj -> RoPE -> GQA -> attention -> output proj."""
    golden = generate_golden_reference(H, G, d, E, S)

    op = AIEAttentionPrefillProjectedFused(H, G, d, E, S)
    op.compile()
    fc = op.get_callable()

    _load_input(fc, "input", golden["input"])
    _load_input(fc, "rope_angles", golden["rope_angles"])
    _load_input(fc, "W_query", golden["W_query"])
    _load_input(fc, "W_key", golden["W_key"])
    _load_input(fc, "W_value", golden["W_value"])
    _load_input(fc, "W_output", golden["W_output"])
    _load_input(fc, "attn_scale_factor", golden["attn_scale_factor"])
    _load_input(fc, "causal_mask", golden["causal_mask"])

    t0 = time.perf_counter()
    fc()
    elapsed = time.perf_counter() - t0
    _print_metrics("projected", elapsed, _projected_gemm_flops(H, G, d, E, S))

    _verify_output(fc, golden, H, d, S, E)


# ---------------------------------------------------------------------------
# Intermediate checks (extensive, not run by default)
# ---------------------------------------------------------------------------

INTERMEDIATE_CHECKS = [
    ("attn_scores", "attn_scores", lambda H, G, S, d: (H, S, S)),
    ("attn_scores_masked", "attn_scores_masked", lambda H, G, S, d: (H, S, S)),
    ("attn_weights", "attn_weights", lambda H, G, S, d: (H, S, S)),
    ("attn_context", "attn_context", lambda H, G, S, d: (H, S, d)),
    ("context_interleaved", "context_interleaved", lambda H, G, S, d: (S, H * d)),
]


@pytest.mark.extensive
@pytest.mark.parametrize("H,G,d,E,S", get_params())
def test_mha_pefill_lxl_sd_intermediates(H, G, d, E, S):
    """Check intermediate buffers of core attention (for debugging)."""
    golden = generate_golden_reference(H, G, d, E, S)

    op = AIEAttentionPrefillFused(H, G, d, E, S)
    op.compile()
    fc = op.get_callable()

    _load_input(fc, "queries", golden["queries_deinterleaved"])
    _load_input(fc, "keys", golden["keys_for_scores"])
    _load_input(fc, "values", golden["values_for_context"])
    _load_input(fc, "W_output", golden["W_output"])
    _load_input(fc, "attn_scale_factor", golden["attn_scale_factor"])
    _load_input(fc, "causal_mask", golden["causal_mask"])

    fc()

    for buf_name, golden_key, shape_fn in INTERMEDIATE_CHECKS:
        shape = shape_fn(H, G, S, d)
        actual = _get_scratch_tensor(fc, buf_name, shape)
        expected = golden[golden_key].float().numpy().reshape(shape)
        diff = np.abs(actual - expected)
        print(
            f"  [{buf_name}] shape={shape} "
            f"nan={int(np.isnan(actual).sum())} "
            f"max_abs_err={diff.max():.4f} mean_abs_err={diff.mean():.6f}"
        )
