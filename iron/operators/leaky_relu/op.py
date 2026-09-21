# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, ClassVar, Dict

from iron.common import ChanneledUnaryOperator


@dataclass
class LeakyReLU(ChanneledUnaryOperator):
    """AIE-accelerated Leaky ReLU operator"""

    alpha: float = 0.01

    kernel_factory: ClassVar[str] = "leaky_relu"
    _name_aliases: ClassVar[Dict[str, str]] = {
        **ChanneledUnaryOperator._name_aliases,
        "alpha": "a",
    }

    # Minimum per-core line length (in bfloat16 elements) required by the
    # vectorized kernels. They tell the pipeliner a minimum loop-trip count via
    # AIE_LOOP_MIN_ITERATION_COUNT -- a hard contract under xchesscc -- so that
    # promise must be backed by a lower bound on the line length, or the
    # compiler may drop the low-trip guard and corrupt results. The kernels
    # vectorize by 16 (aie2) or 32 (aie2p) elements and promise 4 / 2 iterations
    # respectively, i.e. at least 64 elements per line.
    min_line_size: ClassVar[int] = 64

    def __post_init__(self) -> None:
        line_size = min(self.tile_size, self.tile_cap)
        if line_size < self.min_line_size:
            raise ValueError(
                f"tile_size ({self.tile_size}) yields a per-core line of "
                f"{line_size} bfloat16 elements; leaky_relu requires at least "
                f"{self.min_line_size} to satisfy the kernel's minimum "
                f"loop-iteration promise"
            )
        super().__post_init__()

    def _design_kwargs(self) -> dict[str, Any]:
        return {**super()._design_kwargs(), "kernel_args": (self.alpha,)}
