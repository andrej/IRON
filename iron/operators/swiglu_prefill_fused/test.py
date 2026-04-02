#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16
from iron.common.base import AIEBuffer
from iron.common.utils import torch_to_numpy
from iron.operators.swiglu_prefill_fused.op import AIESwiGLUPrefillFused
from iron.operators.swiglu_decode.reference import generate_golden_reference
from iron.common.test_utils import verify_buffer


def _read_scratch_buffer(fc, buf_name, seq_len, hidden_dim):
    """Read a scratch buffer from the fused callable as a torch tensor."""
    buf = fc.get_buffer(buf_name)
    fc.scratch_buffer.on = "npu"
    fc.scratch_buffer.to("cpu")
    arr = np.frombuffer(
        buf.memory_view,
        dtype=buf.dtype,
        count=int(np.prod(buf.shape)),
    ).copy()
    return torch.from_numpy(arr.astype(np.float32)).bfloat16().reshape(seq_len, hidden_dim)


def get_params():
    params_list = [(256, 2048, 2048, False)]

    params = []
    for p in params_list:
        seq_len, emb, hid, _ = p
        name = f"swiglu_prefill_fused_{seq_len}x{emb}x{hid}"
        params.append(pytest.param(*p, id=name))
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("seq_len,embedding_dim,hidden_dim,prio_accuracy", get_params())
def test_swiglu_prefill_fused(seq_len, embedding_dim, hidden_dim, prio_accuracy):
    golden_ref = generate_golden_reference(M=seq_len, K=embedding_dim, N=hidden_dim)

    operator = AIESwiGLUPrefillFused(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        prio_accuracy=bool(prio_accuracy),
    )
    # Weights stored as .T (like the existing prefill operator)
    operator.weights_1 = golden_ref["w_gate"].T
    operator.weights_2 = golden_ref["w_up"].T
    operator.weights_3 = golden_ref["w_down"].T

    operator.compile()
    op_func = operator.get_callable()

    input_buf = AIEBuffer.from_np(torch_to_numpy(golden_ref["input"]))
    output_buf = AIEBuffer(shape=(seq_len * embedding_dim,), dtype=bfloat16)

    op_func(input_buf, output_buf)

    errors = {}

    # Read intermediate buffers from scratch to build a chain-consistent reference
    fc = op_func.fused_callable
    left_activated = _read_scratch_buffer(fc, "left_activated", seq_len, hidden_dim)
    right = _read_scratch_buffer(fc, "right", seq_len, hidden_dim)
    intermediate = _read_scratch_buffer(fc, "intermediate", seq_len, hidden_dim)

    # Verify intermediate (eltwise_mul: left_activated * right)
    ref_intermediate = left_activated * right
    errors_intermediate = verify_buffer(
        intermediate, "intermediate", ref_intermediate, rel_tol=0.04, abs_tol=0.4
    )
    if errors_intermediate:
        errors["intermediate"] = errors_intermediate

    # Verify output using AIE-computed intermediate as reference (isolates GEMM_2 errors)
    # This matches the approach of the existing non-fused prefill test.
    ref_output = intermediate @ golden_ref["w_down"]
    output = output_buf.view_as_torch().reshape((seq_len, embedding_dim))
    errors_output = verify_buffer(
        output, "output", ref_output, rel_tol=0.08, abs_tol=0.4, max_error_rate=0.05
    )
    if errors_output:
        errors["output"] = errors_output

    assert not errors, f"Test failed with errors: {errors}"
