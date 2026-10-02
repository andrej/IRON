#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from iron.common.test_utils import assert_matches_reference
from iron.operators.flm import q4nx
from iron.operators.flm.lm_head.op import LMHead
from iron.operators.flm.q4nx import K_TILE, M_TILE
from iron.operators.flm.testing import requires_aie2p

# The initial value of y. No logit of the test inputs reaches it, so an
# unwritten output fails the check.
SENTINEL = 99.0

GEMMA4_SOFTCAP = 30.0

METRICS = dict(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)


def _inputs(dim, vocab, seed):
    """A q4nx vocabulary and a token with its RMS weight.

    Negative minima and small scales keep the logits in the range a softcap
    of 30 bends.
    """
    rng = np.random.default_rng(seed)
    n_blocks = vocab * dim // (M_TILE * K_TILE)
    w = q4nx.random_blocks(rng, n_blocks, (0, 0.02), (-0.15, 0), "rne").reshape(-1)
    x = np.concatenate([rng.standard_normal(dim), rng.uniform(0.5, 1.5, dim)])
    return w, x.astype(bfloat16)


def _check(dim, vocab, softcap, aie_context, seed=0, tanh_error=True):
    """Compare the device's logits for seed's inputs with the reference.
    Returns the logits."""
    op = LMHead(dim=dim, vocab=vocab, softcap=softcap, context=aie_context)
    w, x = _inputs(dim, vocab, seed)
    got = assert_matches_reference(
        op,
        torch.from_numpy(w.view(np.uint32)),
        torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16),
        tolerance=op.reference_tolerance(tanh_error=tanh_error),
        output_fill=SENTINEL,
        flops=2 * vocab * dim,
    )
    return got.astype(np.float64)


@requires_aie2p
@pytest.mark.parametrize("dim", [1536, 2560])
def test_projection_matches_reference(dim, aie_context):
    """A softcap of 1000 keeps each tanh argument near zero.

    The test therefore checks the projection alone.

    1536 and 2560 are Gemma 4 E2B's and E4B's hidden sizes.
    """
    _check(dim, 4096, 1000.0, aie_context, tanh_error=False)


@requires_aie2p
@pytest.mark.parametrize("dim", [1536, 2560])
def test_gemma4_softcap(dim, aie_context):
    _check(dim, 4096, GEMMA4_SOFTCAP, aie_context, seed=1)


@requires_aie2p
def test_softcap_bounds_the_logits(aie_context):
    """A softcap of 5 saturates most logits. No logit may exceed the softcap."""
    got = _check(1536, 1024, 5.0, aie_context, seed=2)
    assert np.abs(got).max() <= 5.0


@requires_aie2p
@pytest.mark.extensive
@pytest.mark.metrics(**METRICS)
@pytest.mark.parametrize("dim", [pytest.param(1536, marks=pytest.mark.bench), 2560])
def test_gemma4_vocabulary(dim, aie_context):
    """Gemma 4's whole vocabulary of 262144 logits."""
    _check(dim, 262144, GEMMA4_SOFTCAP, aie_context, seed=3)


@requires_aie2p
@pytest.mark.parametrize(
    "dim, vocab, softcap, match",
    [
        (1000, 4096, 30.0, "multiple of"),
        (1536, 1000, 30.0, "multiple of"),
        (1536, 4096, 0.0, "finite and positive"),
        (1536, 4096, float("inf"), "finite and positive"),
    ],
)
def test_rejects_unservable_shapes(dim, vocab, softcap, match, aie_context):
    with pytest.raises(ValueError, match=match):
        LMHead(dim=dim, vocab=vocab, softcap=softcap, context=aie_context)
