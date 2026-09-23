# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict
from ml_dtypes import bfloat16

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
)


@dataclass
class Repeat(MLIROperator):
    """AIE-accelerated repeat-interleave operator"""

    rows: int
    cols: int
    repeat: int
    transfer_size: int | None = None
    dtype: object = field(default=bfloat16, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "repeat": "by",
        "transfer_size": "ts",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_design(self):
        from iron.operators.repeat.design import repeat

        return repeat

    def get_design_kwargs(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "rows": self.rows,
            "cols": self.cols,
            "repeat": self.repeat,
            "transfer_size": self.transfer_size,
        }

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.rows, self.cols), dtype=self.dtype),
            AIERuntimeArgSpec(
                "out", (self.rows * self.repeat, self.cols), dtype=self.dtype
            ),
        ]

    def reference(self, x):
        """CPU reference: repeat-interleave along the leading dimension."""
        from iron.operators.repeat.reference import reference

        return reference(x, self.repeat)
