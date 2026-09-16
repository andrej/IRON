#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build must not be reused across target architectures.

aie_kernels/generic/mul.cc is one source that both aie2 and aie2p compile, to
different machine code. An aie2 build of ElementwiseMul that an aie2p run
picked up would link that machine code into an aie2p core. The compilation
cache keys on the target device, so the two builds address separate entries.

These tests build the cache keys for both arches without invoking a compiler.
"""

import aie.utils as aie_utils
from aie.iron.device import NPU1, NPU2

from iron.common import AIEContext
from iron.operators.elementwise_mul.op import ElementwiseMul


def _cache_key(device):
    aie_utils.set_current_device(device)
    op = ElementwiseMul(
        size=4096, tile_size=4096, num_aie_columns=1, context=AIEContext()
    )
    return op.build_compilable()._compute_cache_hash()


def test_two_arches_do_not_share_a_cache_entry():
    assert _cache_key(NPU1()) != _cache_key(NPU2())


def test_one_arch_reaches_the_same_cache_entry_twice():
    """The key must be built from content, not from anything that moves between
    calls -- an address or a timestamp would miss every cache entry."""
    assert _cache_key(NPU2()) == _cache_key(NPU2())
