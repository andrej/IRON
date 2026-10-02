#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""verify_buffer's bound tolerances and fill check. The tests need no NPU."""

import numpy as np
import pytest
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from iron.common.test_utils import verify_buffer

BOUND = Tolerance.bounded(lambda: None)


def test_bound_flags_the_elements_past_their_limit():
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    output = np.array([1.5, 2.0, 3.0, 5.0], bfloat16)
    limit = np.array([0.5, 0.0, 0.0, 0.5])
    assert verify_buffer(output, "y", expected, tolerance=BOUND, bound=limit) == [3]


def test_bound_needs_its_limit():
    with pytest.raises(ValueError, match="bound="):
        verify_buffer(np.zeros(4, bfloat16), "y", np.zeros(4), tolerance=BOUND)
    with pytest.raises(ValueError, match="bound="):
        verify_buffer(np.zeros(4, bfloat16), "y", np.zeros(4), bound=np.zeros(4))


def test_fill_flags_unwritten_elements():
    """Element 2 within the bound still holds the fill. Element 3 holds the
    fill value that the reference also holds."""
    expected = np.array([1.0, 2.0, 98.5, 99.0])
    output = np.array([1.0, 2.0, 99.0, 99.0], bfloat16)
    limit = np.full(4, 1.0)
    errors = verify_buffer(
        output, "y", expected, tolerance=BOUND, bound=limit, fill=99.0
    )
    assert errors == [2]
