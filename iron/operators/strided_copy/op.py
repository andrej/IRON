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
class StridedCopy(MLIROperator):
    """AIE-accelerated strided copy operator"""

    input_sizes: list
    input_strides: list
    input_offset: int
    output_sizes: list
    output_strides: list
    output_offset: int
    input_buffer_size: int = field(repr=False)
    output_buffer_size: int = field(repr=False)
    dtype: object = field(default=bfloat16, repr=False)
    transfer_size: int | None = None
    num_aie_channels: int = 1
    input_offset_parameter: str | None = None
    output_offset_parameter: str | None = None
    kwargs: dict = field(default_factory=dict, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "input_sizes": "isz",
        "input_strides": "ist",
        "input_offset": "ioff",
        "output_sizes": "osz",
        "output_strides": "ost",
        "output_offset": "ooff",
        "transfer_size": "tr",
        "num_aie_channels": "ch",
        "input_offset_parameter": "ipar",
        "output_offset_parameter": "opar",
    }

    def __post_init__(self):
        if len(self.input_sizes) != len(self.input_strides):
            raise ValueError(
                f"input_sizes and input_strides must have the same length "
                f"({len(self.input_sizes)} vs {len(self.input_strides)})"
            )
        if len(self.output_sizes) != len(self.output_strides):
            raise ValueError(
                f"output_sizes and output_strides must have the same length "
                f"({len(self.output_sizes)} vs {len(self.output_strides)})"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_design(self):
        from iron.operators.strided_copy.design import strided_copy

        return strided_copy

    def get_design_kwargs(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "input_buffer_size": self.input_buffer_size,
            "input_sizes": tuple(self.input_sizes),
            "input_strides": tuple(self.input_strides),
            "input_offset": self.input_offset,
            "output_buffer_size": self.output_buffer_size,
            "output_sizes": tuple(self.output_sizes),
            "output_strides": tuple(self.output_strides),
            "output_offset": self.output_offset,
            "transfer_size": self.transfer_size,
            "num_aie_channels": self.num_aie_channels,
            **self.kwargs,
            "input_offset_parameter": self.input_offset_parameter,
            "output_offset_parameter": self.output_offset_parameter,
        }

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (int(self.input_buffer_size),), dtype=self.dtype),
            AIERuntimeArgSpec("out", (int(self.output_buffer_size),), dtype=self.dtype),
        ]
