# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

from iron.common import ChanneledUnaryOperator


@dataclass
class SiLU(ChanneledUnaryOperator):
    """AIE-accelerated SiLU activation function"""

    num_channels: int = field(default=1, init=False, repr=False)

    kernel_factory: ClassVar[str] = "silu_sized"

    def reference(self, x):
        from iron.operators.silu.reference import reference

        return reference(x)
