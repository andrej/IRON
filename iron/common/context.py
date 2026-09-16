# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class AIEContext:
    """Context for managing AIE operator compilation state.

    Attributes:
        base_dir: Repository root directory (three levels above this file).
        mlir_verbose: Enable verbose MLIR output during compilation.
        compiler: Kernel compiler to use: "peano" (default) or "chess".
                  When "chess", all kernels and aiecc linking use xchesscc.
                  Requires Vitis/aietools in PATH.
    """

    # Repo root: iron/common/../../.. = three levels up from this file.
    base_dir: ClassVar[Path] = Path(__file__).parent.parent.parent

    mlir_verbose: bool = False
    compiler: str = "peano"

    def __post_init__(self) -> None:
        if self.compiler not in ("peano", "chess"):
            raise ValueError(
                f"compiler must be 'peano' or 'chess', got {self.compiler!r}"
            )
