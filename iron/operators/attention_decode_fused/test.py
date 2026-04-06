#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from iron.common.base import AIEBuffer
from iron.common.utils import torch_to_numpy
from iron.common.test_utils import verify_buffer
from iron.operators.attention_decode_fused.op import AIEAttentionDecodeFused
from iron.operators.attention_decode_fused.reference import generate_golden_reference


def get_params():
    # (num_heads, num_kv_groups, head_dim, embedding_dim, max_seq_len, token_position)
    params_list = [
        # Small MHA test: H=16 (must be multiple of 16 for softmax), full context (t=S-1)
        # E = H*d = 16*64 = 1024, S=256 (multiple of 256 for GEMV tiling)
        (16, 16, 64, 1024, 256, 255),
    ]
    params = []
    for p in params_list:
        H, G, d, E, S, t = p
        name = f"attn_decode_fused_H{H}G{G}d{d}E{E}S{S}t{t}"
        params.append(pytest.param(*p, id=name))
    return params


@pytest.mark.parametrize(
    "num_heads,num_kv_groups,head_dim,embedding_dim,max_seq_len,token_position",
    get_params(),
)
def test_attention_decode_fused(
    num_heads, num_kv_groups, head_dim, embedding_dim, max_seq_len, token_position
):
    H, G, d, E, S, t = num_heads, num_kv_groups, head_dim, embedding_dim, max_seq_len, token_position

    golden = generate_golden_reference(
        num_heads=H,
        num_kv_groups=G,
        head_dim=d,
        embedding_dim=E,
        max_seq_len=S,
        token_position=t,
    )

    # Build operator
    op = AIEAttentionDecodeFused(
        num_heads=H,
        num_kv_groups=G,
        head_dim=d,
        embedding_dim=E,
        max_seq_len=S,
    )

    # Set weights (GEMV stores matrix as (M, K) — already the correct layout)
    op.W_query = golden["W_query"]   # (H*d, E)
    op.W_key = golden["W_key"]       # (G*d, E)
    op.W_value = golden["W_value"]   # (G*d, E)
    op.W_output = golden["W_output"] # (E, H*d)
    op.attn_scale_factor = golden["attn_scale_factor"]  # (H*S,)

    # Pre-fill KV cache with golden reference initial state (positions 0..t-1)
    # The reference generates keys_cache_init as (G, S, d); flatten for the op
    op.keys_cache_init = golden["keys_cache_init"].flatten()
    op.values_cache_init = golden["values_cache_init"].flatten()

    # Compile
    op.compile()

    # Get callable
    op_func = op.get_callable()

    # Patch ELF for the current token position
    op_func.set_token_position(t)

    # Prepare input buffers
    input_buf = AIEBuffer(shape=(E,), dtype=bfloat16)
    input_buf.view_as_np()[:] = torch_to_numpy(golden["input"]).flatten()

    rope_angles_buf = AIEBuffer(shape=(1, d), dtype=bfloat16)
    rope_angles_buf.view_as_np()[:] = torch_to_numpy(golden["rope_angles"]).flatten()

    output_buf = AIEBuffer(shape=(E,), dtype=bfloat16)

    # Run
    op_func(input_buf, rope_angles_buf, output_buf)

    # Verify output against golden reference
    ref_output = golden["attn_output"]
    aie_output = torch.from_numpy(output_buf.view_as_np().reshape(E,).astype(np.float32))

    errors = {}
    errs = verify_buffer(
        aie_output,
        "attn_output",
        ref_output.float(),
        rel_tol=0.10,
        abs_tol=1.0,
        max_error_rate=0.01,
    )
    if errs:
        errors["attn_output"] = errs

    assert not errors, f"Test failed with errors: {errors}"
