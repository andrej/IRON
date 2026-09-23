# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
)


@dataclass
class MemCopy(MLIROperator):
    """AIE-accelerated memory copy operator."""

    size: int
    num_cores: int
    num_channels: int
    bypass: bool
    tile_size: int
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_cores": "cores",
        "num_channels": "chans",
        "tile_size": "tile",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_design(self):
        from iron.operators.mem_copy.design import my_mem_copy

        return my_mem_copy

    def get_design_kwargs(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "num_cores": self.num_cores,
            "num_channels": self.num_channels,
            "bypass": self.bypass,
            "tile_size": self.tile_size,
            "trace_size": 0,
        }

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]
