# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar

from iron.common import ChanneledUnaryOperator


@dataclass
class GELU(ChanneledUnaryOperator):
    """AIE-accelerated GELU activation function"""

    kernel_factory: ClassVar[str] = "gelu_sized"
    tile_cap: ClassVar[int] = 8192
