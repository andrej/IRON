# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar

from iron.common import BinaryElementwiseOperator

from iron.operators.axpy.design import my_axpy


@dataclass
class AXPY(BinaryElementwiseOperator):
    """AIE-accelerated aX + Y operator"""

    scalar_factor: float = 3.0

    kernel_factory: ClassVar[str] = "axpy"

    def get_design(self):
        return my_axpy

    def get_design_kwargs(self):
        return {
            "num_elements": self.size,
            "num_columns": self.num_aie_columns,
            "tile_size": self.tile_size,
            "trace_size": 0,
            "scalar_factor": self.scalar_factor,
        }
