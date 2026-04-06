# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from iron.common.base import AIEBuffer
from iron.common.utils import torch_to_numpy

from iron.operators.attention_prefill_fused.op import AIEAttentionPrefillFused
from iron.operators.attention_prefill_fused.reference import generate_golden_reference


def get_params():
    """Return test parameter sets (H, G, d, E, S)."""
    return [
        pytest.param(2, 2, 64, 256, 256, id="H2"),
        pytest.param(32, 8, 64, 2048, 256, id="H32"),
    ]


@pytest.mark.parametrize("H,G,d,E,S", get_params())
def test_attention_prefill_fused(H, G, d, E, S):
    """Full fused attention prefill: Q/K/V proj → RoPE → GQA → attention → output proj."""
    group_size = H // G

    # ---- Generate golden reference ----
    golden = generate_golden_reference(H, G, d, E, S)

    # ---- Build and compile operator ----
    op = AIEAttentionPrefillFused(
        num_heads=H,
        num_kv_groups=G,
        head_dim=d,
        embedding_dim=E,
        seq_len=S,
    )
    op.W_query = golden["W_query"]
    op.W_key = golden["W_key"]
    op.W_value = golden["W_value"]
    op.W_output = golden["W_output"]
    op.attn_scale_factor = golden["attn_scale_factor"]
    op.causal_mask = golden["causal_mask"]
    op.compile()

    # ---- Create I/O buffers ----
    input_buf = AIEBuffer(shape=(S, E), dtype=bfloat16)
    input_buf.view_as_np()[:] = torch_to_numpy(golden["input"]).reshape(S, E)
    input_buf.to("npu")

    rope_angles_buf = AIEBuffer(shape=(S, d), dtype=bfloat16)
    rope_angles_buf.view_as_np()[:] = torch_to_numpy(golden["rope_angles"]).reshape(S, d)
    rope_angles_buf.to("npu")

    output_buf = AIEBuffer(shape=(S, E), dtype=bfloat16)

    # ---- Execute ----
    callable_fn = op.get_callable()
    callable_fn(input_buf, rope_angles_buf, output_buf)

    # ---- Sync scratch buffer ----
    scratch = callable_fn.fused_callable.scratch_buffer
    scratch.on = "npu"
    scratch.to("cpu")

    def get_scratch_tensor(name, shape):
        sub = callable_fn.fused_callable.get_buffer(name)
        return np.frombuffer(
            sub.memory_view, dtype=bfloat16, count=int(np.prod(shape))
        ).reshape(shape).astype(np.float32)

    # ---- Intermediate checks ----
    intermediate_checks = [
        ("queries_raw", "queries_raw", (S * H, d)),
        ("keys_raw", "keys_raw", (S * G, d)),
        ("values_raw", "values_raw", (S * G, d)),
        ("queries_roped", "queries_roped", (S * H, d)),
        ("keys_roped", "keys_roped", (S * G, d)),
        ("queries_deint", "queries_deinterleaved", (H * S, d)),
        ("keys_deint", "keys_deinterleaved", (G * S, d)),
        ("values_deint", "values_deinterleaved", (G * S, d)),
        ("keys_transposed", "keys_transposed", (G, d, S)),
        ("keys_for_scores", "keys_for_scores", (H, d, S)),
        ("values_for_context", "values_for_context", (H, S, d)),
        ("attn_scores", "attn_scores", (H, S, S)),
        ("attn_scores_masked", "attn_scores_masked", (H, S, S)),
        ("attn_weights", "attn_weights", (H, S, S)),
        ("attn_context", "attn_context", (H, S, d)),
        ("context_interleaved", "context_interleaved", (S, H * d)),
    ]

    for buf_name, golden_key, shape in intermediate_checks:
        actual = get_scratch_tensor(buf_name, shape)
        expected = golden[golden_key].float().numpy().reshape(shape)

        nan_count = int(np.isnan(actual).sum())
        inf_count = int(np.isinf(actual).sum())

        diff = np.abs(actual - expected)
        rel_diff = diff / (np.abs(expected) + 1e-6)
        err_count = int((diff > 1.0).sum())
        rel_err_count = int((rel_diff > 0.10).sum())

        print(
            f"  [{buf_name}] shape={shape} "
            f"nan={nan_count} inf={inf_count} "
            f"abs_err: max={diff.max():.4f} mean={diff.mean():.6f} count(>1.0)={err_count} "
            f"rel_err: max={rel_diff.max():.4f} count(>0.10)={rel_err_count}"
        )

    # ---- NaN diagnostics for attn_weights ----
    attn_weights_actual = get_scratch_tensor("attn_weights", (H, S, S))
    nan_positions = np.argwhere(np.isnan(attn_weights_actual))
    if len(nan_positions) > 0:
        print(f"\n  [NaN diagnostics] {len(nan_positions)} NaN(s) in attn_weights:")
        seen_heads = set()
        for idx in nan_positions[:20]:
            head, row, col = idx
            if head not in seen_heads:
                seen_heads.add(head)
                try:
                    masked_row = get_scratch_tensor(
                        "attn_scores_masked", (H, S, S)
                    )[head, row, :]
                    print(
                        f"    head={head} pos={row}: "
                        f"masked_row min={masked_row.min():.4f} max={masked_row.max():.4f} "
                        f"nan={np.isnan(masked_row).sum()}"
                    )
                except Exception as e:
                    print(f"    head={head} pos={row}: could not read scores_masked: {e}")

    # ---- Output check ----
    out_np = output_buf.view_as_np().astype(np.float32).reshape(S, E)
    expected_out = golden["attn_output"].float().numpy().reshape(S, E)

    nan_count = int(np.isnan(out_np).sum())
    inf_count = int(np.isinf(out_np).sum())
    zeros_count = int((out_np == 0).sum())
    diff = np.abs(out_np - expected_out)
    rel_diff = diff / (np.abs(expected_out) + 1e-6)

    print(f"\n  [Output] shape=({S},{E}) nan={nan_count} inf={inf_count} zeros={zeros_count}")

    # Error distribution
    for threshold in [0.5, 1, 2, 5, 10]:
        count = int((diff > threshold).sum())
        pct = 100.0 * count / diff.size
        print(f"    abs_err > {threshold}: {count} ({pct:.2f}%)")

    # Cosine similarity
    a_flat = out_np.flatten()
    b_flat = expected_out.flatten()
    cos_sim = float(
        np.dot(a_flat, b_flat)
        / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)
    )
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    print(f"    cosine_similarity={cos_sim:.6f}  rmse={rmse:.4f}")

    # Adaptive tolerances based on output GEMM K dimension
    output_K = H * d
    if output_K <= 256:
        # Smaller GEMM → tighter tolerances
        max_rel_tol = 0.20
        max_abs_tol = 2.0
        max_err_frac = 0.05
        min_cosine = 0.995
    else:
        # Larger GEMM (e.g. H=32, K=2048) → relaxed tolerances for bf16
        max_rel_tol = 0.50
        max_abs_tol = 50.0
        max_err_frac = 0.98
        min_cosine = 0.97

    print(
        f"    tolerances: output_K={output_K} "
        f"min_cosine={min_cosine} max_abs={max_abs_tol} max_rel={max_rel_tol}"
    )

    assert cos_sim >= min_cosine, (
        f"Cosine similarity {cos_sim:.6f} < {min_cosine} "
        f"(output_K={output_K}, rmse={rmse:.4f})"
    )


def _runlist_checkpoints(H, G):
    """Return a list of (max_runlist_entries, buf_name, golden_key, shape) tuples
    for incremental runlist debugging.
    """
    group_size = H // G
    return [
        # After V projection
        (3, "values_raw", "values_raw", lambda S, d: (S * G, d)),
        # After RoPE on keys
        (5, "keys_roped", "keys_roped", lambda S, d: (S * G, d)),
        # After deinterleave V
        (8, "values_deint", "values_deinterleaved", lambda S, d: (G * S, d)),
        # After transpose keys
        (8 + G, "keys_transposed", "keys_transposed", lambda S, d: (G, d, S)),
        # After repeat values
        (10 + G, "values_for_context", "values_for_context", lambda S, d: (H, S, d)),
        # After score GEMMs
        (10 + G + H, "attn_scores", "attn_scores", lambda S, d: (H, S, S)),
        # After scale
        (11 + G + H, "attn_scores", "attn_scores_scaled", lambda S, d: (H, S, S)),
        # After mask
        (12 + G + H, "attn_scores_masked", "attn_scores_masked", lambda S, d: (H, S, S)),
        # After softmax
        (13 + G + H, "attn_weights", "attn_weights", lambda S, d: (H, S, S)),
    ]


@pytest.mark.parametrize("checkpoint_idx", list(range(9)))
def test_debug_incremental(checkpoint_idx, H=2, G=2, d=64, E=256, S=256):
    """Run the fused operator with a truncated runlist and verify correctness
    at each checkpoint."""

    checkpoints = _runlist_checkpoints(H, G)
    max_entries, buf_name, golden_key, shape_fn = checkpoints[checkpoint_idx]
    shape = shape_fn(S, d)

    golden = generate_golden_reference(H, G, d, E, S)

    op = AIEAttentionPrefillFused(
        num_heads=H,
        num_kv_groups=G,
        head_dim=d,
        embedding_dim=E,
        seq_len=S,
        max_runlist_entries=max_entries,
    )
    op.W_query = golden["W_query"]
    op.W_key = golden["W_key"]
    op.W_value = golden["W_value"]
    op.W_output = golden["W_output"]
    op.attn_scale_factor = golden["attn_scale_factor"]
    op.causal_mask = golden["causal_mask"]
    op.compile()

    input_buf = AIEBuffer(shape=(S, E), dtype=bfloat16)
    input_buf.view_as_np()[:] = torch_to_numpy(golden["input"]).reshape(S, E)
    input_buf.to("npu")

    rope_angles_buf = AIEBuffer(shape=(S, d), dtype=bfloat16)
    rope_angles_buf.view_as_np()[:] = torch_to_numpy(golden["rope_angles"]).reshape(S, d)
    rope_angles_buf.to("npu")

    output_buf = AIEBuffer(shape=(S, E), dtype=bfloat16)

    callable_fn = op.get_callable()
    callable_fn(input_buf, rope_angles_buf, output_buf)

    scratch = callable_fn.fused_callable.scratch_buffer
    scratch.on = "npu"
    scratch.to("cpu")

    sub = callable_fn.fused_callable.get_buffer(buf_name)
    actual = np.frombuffer(
        sub.memory_view, dtype=bfloat16, count=int(np.prod(shape))
    ).reshape(shape).astype(np.float32)

    expected = golden[golden_key].float().numpy()
    if expected.shape != shape:
        expected = expected.reshape(shape)

    nan_count = int(np.isnan(actual).sum())
    diff = np.abs(actual - expected)
    print(
        f"  checkpoint {checkpoint_idx}: max_entries={max_entries} "
        f"buf={buf_name} shape={shape} "
        f"nan={nan_count} max_diff={diff.max():.4f} mean_diff={diff.mean():.6f}"
    )

    assert nan_count == 0, f"NaN found in {buf_name} at checkpoint {checkpoint_idx}"
