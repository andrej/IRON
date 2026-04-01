#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from ml_dtypes import bfloat16
from iron.common.base import AIEBuffer
from iron.common.utils import torch_to_numpy
from iron.operators.swiglu_decode_fused.op import AIESwiGLUDecodeFused
from iron.operators.swiglu_decode.reference import generate_golden_reference
from iron.common.test_utils import verify_buffer


def get_params():
    params_list = [(2048, 2048)]

    params = []
    for p in params_list:
        emb, hid = p
        name = f"swiglu_decode_fused_1x{emb}x{hid}"
        params.append(pytest.param(*p, id=name))
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("embedding_dim,hidden_dim", get_params())
def test_swiglu_decode_fused(embedding_dim, hidden_dim):
    golden_ref = generate_golden_reference(M=1, K=embedding_dim, N=hidden_dim)

    operator = AIESwiGLUDecodeFused(
        embedding_dim=embedding_dim, hidden_dim=hidden_dim
    )
    # weights_1 = W_gate transposed to (hidden_dim, embedding_dim)
    operator.weights_1 = golden_ref["w_gate"].T
    # weights_2 = W_up transposed to (hidden_dim, embedding_dim)
    operator.weights_2 = golden_ref["w_up"].T
    # weights_3 = W_down shape (embedding_dim, hidden_dim)
    operator.weights_3 = golden_ref["w_down"].T

    operator.compile()
    op_func = operator.get_callable()

    input_buf = AIEBuffer.from_np(torch_to_numpy(golden_ref["input"]))
    output_buf = AIEBuffer(shape=(1, embedding_dim), dtype=bfloat16)

    op_func(input_buf, output_buf)

    errors = {}

    # Verify output
    intermediate = golden_ref["intermediate"]
    ref_output = intermediate @ golden_ref["w_down"]
    output = output_buf.view_as_torch().reshape((1, embedding_dim))
    errors_output = verify_buffer(output, "output", ref_output, rel_tol=0.07, abs_tol=0.7, max_error_rate=0.05)
    if errors_output:
        errors["output"] = errors_output

    assert not errors, f"Test failed with errors: {errors}"
