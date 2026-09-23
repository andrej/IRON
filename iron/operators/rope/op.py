# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
)


@dataclass
class RoPE(MLIROperator):
    """AIE-accelerated RoPE (Rotary Position Embedding) operator"""

    rows: int
    cols: int
    angle_rows: int | None = None
    num_aie_columns: int = 1
    method_type: int = 0
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "angle_rows": "arows",
        "method_type": "m",
    }

    def __post_init__(self):
        if self.angle_rows is None:
            self.angle_rows = self.rows

        if not (self.cols % (16 * 2) == 0 and self.cols >= (16 * 2)):
            raise ValueError("cols must be multiple of 32 and >= 32")
        if self.rows % self.num_aie_columns != 0:
            raise ValueError("rows must be divisible by num_aie_columns")
        if not (self.angle_rows <= self.rows and self.rows % self.angle_rows == 0):
            raise ValueError("angle_rows must divide rows")
        if not (
            self.angle_rows >= self.num_aie_columns
            and self.angle_rows % self.num_aie_columns == 0
        ):
            raise ValueError("angle_rows must be divisible by num_aie_columns")
        if self.method_type not in {0, 1}:
            raise ValueError(f"method_type must be 0 or 1, got {self.method_type}")

        MLIROperator.__init__(self, context=self.context)

    def get_design(self):
        from iron.operators.rope.design import rope

        return rope

    def get_design_kwargs(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "angle_rows": self.angle_rows,
            "num_aie_columns": self.num_aie_columns,
            "trace_size": 0,
            "method_type": self.method_type,
        }

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.rows, self.cols)),  # input tensor
            AIERuntimeArgSpec("in", (self.angle_rows, self.cols)),  # angles
            AIERuntimeArgSpec("out", (self.rows, self.cols)),  # output
        ]

    def reference(self, x, angles):
        """CPU reference for RoPE.

        Assumes ``angles`` holds interleaved [cos, sin, cos, sin, ...] pairs
        along the last dim (length ``cols``).  Only ``method_type == 0``
        (TWO_HALVES) is currently supported.

        ``angles`` may have fewer rows than ``x``; in that case the angles
        are tiled along the row dimension to match ``x``."""
        from iron.operators.rope.reference import reference

        return reference(x, angles, self.method_type, self.rows, self.cols)
