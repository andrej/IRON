#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import aie.utils as aie_utils

from iron.operators.flm_gemm_prebuilt.op import FLMGEMMPrebuilt
from iron.operators.flm_gemm.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    dev = aie_utils.get_current_device()
    # The overlay is a fixed 4x8 grid, so it needs all 8 columns.
    if dev.cols < 8 or dev.resolve().name != "npu2":
        return []

    # The same shapes iron/operators/flm_gemm/test.py runs, so the port and
    # the overlay it was ported from can be compared run for run.
    # fmt: off
    #      M,    K,     N, epilogue,    clamp
    regular_params = [
        (  256,  512,  1024, "none",     None),
        (  512, 1024,  2048, "none",     None),
        (  256,  512,  1536, "none",     None),  # remainder: 4 of 8 cols
        (  256,  512,   128, "none",     None),  # remainder only: 1 col
        (  256,  512,  1024, "silu",     None),
        (  256,  512,  1024, "gelu",     None),
        (  256,  512,  1024, "none", (-2.0, 2.0)),
    ]
    extensive_params = [
        ( 1024, 2048,  2048, "none",     None),
        ( 2048, 2048,  2048, "none",     None),
        ( 1024, 2560,  2560, "none",     None),  # E4B o-proj
        (  512, 1536,  1536, "silu",     None),  # E2B down-proj
        (  256,  512,  1024, "sigmoid",  None),
        (  512, 1024,  2048, "silu", (-4.0, 4.0)),
    ]
    # fmt: on

    params = [pytest.param(*p) for p in regular_params]
    params += [
        pytest.param(*p, marks=[pytest.mark.extensive]) for p in extensive_params
    ]
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize("M,K,N,epilogue,clamp", get_params())
def test_flm_gemm_prebuilt(M, K, N, epilogue, clamp, aie_context):
    # Keep the activation tests in the range where the curve is not flat.
    scale = 4.0 if epilogue == "none" else 0.5
    golden_ref = generate_golden_reference(
        M=M, K=K, N=N, epilogue=epilogue, clamp=clamp, scale=scale
    )

    operator = FLMGEMMPrebuilt(
        M=M, K=K, N=N, epilogue=epilogue, clamp=clamp, context=aie_context
    )

    input_buffers = {
        "A": golden_ref["input"].flatten(),
        # B is consumed pre-packed; see FLMGEMMPrebuilt.pack_B.
        "B": operator.pack_B(golden_ref["input_b"]),
    }
    output_buffers = {"C": golden_ref["output"].flatten()}

    # The bound is on absolute error against the accumulated mass, for the
    # reasons flm_gemm/test.py sets out: with signed A the K-term sum cancels
    # by ~sqrt(K), so near-zero outputs are not relatively checkable, while the
    # error tracks the accumulated magnitude.
    #
    # The overlay's cores never leave the power-up rounding mode, so every
    # bf16 conversion truncates and the bias accumulates over the K reduction
    # instead of cancelling. That is ~0.0099 of mass rather than the ~0.00042
    # of FLMGEMM's default, so this budget matches FLMGEMM's floor mode, in
    # which the two are bit-identical.
    mass = (
        K
        * golden_ref["input"].abs().float().mean()
        * (golden_ref["input_b"].abs().float().mean())
    )
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.04,
        abs_tol=float(0.05 * mass),
    )

    gflops = (2.0 * M * K * N) / (latency_us * 1e-6) / 1e9
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    assert not errors, "Test failed"
