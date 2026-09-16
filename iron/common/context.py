# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import os

import aie.utils.config


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

    @property
    def kernels_dir(self) -> Path:
        """C++ kernel sources bundled with the installed mlir-aie package.

        IRON_AIE_KERNELS_DIR overrides this to point at a local mlir-aie
        checkout for kernel development.
        """
        # Lazy: root_path() needs the package importable at call time.
        override = os.environ.get("IRON_AIE_KERNELS_DIR")
        if override:
            return Path(override)
        return Path(aie.utils.config.root_path()) / "include" / "aie_kernels"

    def __post_init__(self) -> None:
        if self.compiler not in ("peano", "chess"):
            raise ValueError(
                f"compiler must be 'peano' or 'chess', got {self.compiler!r}"
            )
