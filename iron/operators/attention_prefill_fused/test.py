# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from iron.common.base import AIEBuffer
from iron.common.test_utils import verify_buffer
from iron.common.utils import torch_to_numpy

from iron.operators.attention_prefill_fused.op import AIEAttentionPrefillFused
from iron.operators.attention_prefill_fused.reference import generate_golden_reference

REL_TOL = 0.08
ABS_TOL = 2.0
MAX_ERROR_RATE = 0.03


def get_params():
    return [
        pytest.param(2, 2, 64, 256, 256, id="H2"),
        pytest.param(32, 8, 64, 2048, 256, id="H32"),
    ]


def _build_and_run(H, G, d, E, S, golden):
    """Build, compile, and execute the fused attention prefill operator."""
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

    input_buf = AIEBuffer(shape=(S, E), dtype=bfloat16)
    input_buf.view_as_np()[:] = torch_to_numpy(golden["input"]).reshape(S, E)
    input_buf.to("npu")

    rope_angles_buf = AIEBuffer(shape=(S, d), dtype=bfloat16)
    rope_angles_buf.view_as_np()[:] = torch_to_numpy(golden["rope_angles"]).reshape(S, d)
    rope_angles_buf.to("npu")

    output_buf = AIEBuffer(shape=(S, E), dtype=bfloat16)

    callable_fn = op.get_callable()
    callable_fn(input_buf, rope_angles_buf, output_buf)

    return callable_fn, output_buf


def _get_scratch_tensor(callable_fn, name, shape):
    """Read a named buffer from the fused operator's scratch space."""
    scratch = callable_fn.fused_callable.scratch_buffer
    scratch.on = "npu"
    scratch.to("cpu")
    sub = callable_fn.fused_callable.get_buffer(name)
    return np.frombuffer(
        sub.memory_view, dtype=bfloat16, count=int(np.prod(shape))
    ).reshape(shape).astype(np.float32)


@pytest.mark.parametrize("H,G,d,E,S", get_params())
def test_attention_prefill_fused(H, G, d, E, S):
    """Full fused attention prefill: Q/K/V proj -> RoPE -> GQA -> attention -> output proj."""
    golden = generate_golden_reference(H, G, d, E, S)
    callable_fn, output_buf = _build_and_run(H, G, d, E, S, golden)

    # Use the NPU's actual context_interleaved as input to compute a
    # chain-consistent output reference. This isolates the output GEMM's
    # accuracy from error propagation through earlier stages (score GEMM
    # amplifies small Q/K errors via dot product).
    npu_context = torch.from_numpy(
        _get_scratch_tensor(callable_fn, "context_interleaved", (S, H * d))
    ).bfloat16()
    chain_ref = (npu_context.float() @ golden["W_output"].float()).to(torch.bfloat16)

    output = output_buf.view_as_torch().reshape(S, E)
    errors = verify_buffer(
        output, "attn_output", chain_ref.reshape(S, E),
        rel_tol=REL_TOL, abs_tol=ABS_TOL, max_error_rate=MAX_ERROR_RATE,
    )
    assert not errors, f"Output verification failed with {len(errors)} errors"


INTERMEDIATE_CHECKS = [
    ("queries_raw", "queries_raw", lambda H, G, S, d: (S * H, d)),
    ("keys_raw", "keys_raw", lambda H, G, S, d: (S * G, d)),
    ("values_raw", "values_raw", lambda H, G, S, d: (S * G, d)),
    ("queries_roped", "queries_roped", lambda H, G, S, d: (S * H, d)),
    ("keys_roped", "keys_roped", lambda H, G, S, d: (S * G, d)),
    ("queries_deint", "queries_deinterleaved", lambda H, G, S, d: (H * S, d)),
    ("keys_deint", "keys_deinterleaved", lambda H, G, S, d: (G * S, d)),
    ("values_deint", "values_deinterleaved", lambda H, G, S, d: (G * S, d)),
    ("keys_transposed", "keys_transposed", lambda H, G, S, d: (G, d, S)),
    ("keys_for_scores", "keys_for_scores", lambda H, G, S, d: (H, d, S)),
    ("values_for_context", "values_for_context", lambda H, G, S, d: (H, S, d)),
    ("attn_scores", "attn_scores", lambda H, G, S, d: (H, S, S)),
    ("attn_scores_masked", "attn_scores_masked", lambda H, G, S, d: (H, S, S)),
    ("attn_weights", "attn_weights", lambda H, G, S, d: (H, S, S)),
    ("attn_context", "attn_context", lambda H, G, S, d: (H, S, d)),
    ("context_interleaved", "context_interleaved", lambda H, G, S, d: (S, H * d)),
]


@pytest.mark.extensive
@pytest.mark.parametrize("H,G,d,E,S", get_params())
def test_attention_prefill_fused_intermediates(H, G, d, E, S):
    """Check all intermediate buffers (for debugging, not run by default)."""
    golden = generate_golden_reference(H, G, d, E, S)
    callable_fn, _ = _build_and_run(H, G, d, E, S, golden)

    for buf_name, golden_key, shape_fn in INTERMEDIATE_CHECKS:
        shape = shape_fn(H, G, S, d)
        actual = _get_scratch_tensor(callable_fn, buf_name, shape)
        expected = golden[golden_key].float().numpy().reshape(shape)
        diff = np.abs(actual - expected)
        print(
            f"  [{buf_name}] shape={shape} "
            f"nan={int(np.isnan(actual).sum())} "
            f"max_abs_err={diff.max():.4f} mean_abs_err={diff.mean():.6f}"
        )
