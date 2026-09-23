# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import aie.utils as aie_utils

from .base import MLIROperator, AIERuntimeArgSpec
from .context import AIEContext
from .compilation import (
    KernelArchiveArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
from .device_utils import get_kernel_dir
from .utils import get_shim_dma_limit


def lut_based_ops_artifacts(kernel_dir: str) -> list[KernelObjectArtifact]:
    """Return the lut_based_ops kernel artifact for aie2 devices, empty list otherwise."""
    if kernel_dir != "aie2":
        return []
    mlir_aie_dir = Path(aie_utils.config.root_path())
    return [
        KernelObjectArtifact(
            "lut_based_ops.o",
            dependencies=[
                SourceArtifact(
                    mlir_aie_dir / "aie_runtime_lib" / "AIE2" / "lut_based_ops.cpp"
                )
            ],
        )
    ]


@dataclass
class ChanneledUnaryOperator(MLIROperator):
    """Base class for channeled unary AIE operators (single input, single output).

    Subclasses name the mlir-aie kernel factory that writes their compute kernel:

        kernel_factory: attribute of ``aie.iron.kernels`` (e.g. "gelu_sized")
        tile_cap:       largest tile the kernel takes, in elements

    Customization points:
        - For operators with extra parameters (e.g. alpha), add dataclass fields
          and override _design_kwargs().
        - For non-standard arg specs, override get_arg_spec() directly.
        - If none of these fit, subclass MLIROperator instead.
    """

    size: int
    num_aie_columns: int
    num_channels: int
    tile_size: int
    context: AIEContext | None = field(default=None, repr=False)

    kernel_factory: ClassVar[str]
    tile_cap: ClassVar[int] = 4096

    def __post_init__(self) -> None:
        max_multiple = self.num_aie_columns * self.tile_size
        if self.size % max_multiple != 0:
            raise ValueError(
                f"size ({self.size}) must be a multiple of "
                f"num_aie_columns * tile_size ({max_multiple})"
            )
        dev = aie_utils.get_current_device()
        shim_dma_limit = get_shim_dma_limit(dev)
        total_shimdma_channels = self.num_aie_columns * self.num_channels
        if total_shimdma_channels > shim_dma_limit:
            raise ValueError(
                f"num_aie_columns * num_channels ({total_shimdma_channels}) "
                f"exceeds ShimDMA limit of {shim_dma_limit} for this device"
            )
        super().__init__(context=self.context)

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]

    def _design_kwargs(self) -> dict[str, Any]:
        """Subclasses with extra parameters (e.g. alpha) add them here."""
        return {
            "size": self.size,
            "num_columns": self.num_aie_columns,
            "num_channels": self.num_channels,
            "tile_size": self.tile_size,
            "trace_size": 0,
        }

    def get_design(self):
        from iron.operators.channeled_unary_design import channeled_unary_design

        return channeled_unary_design

    def get_design_kwargs(self) -> dict[str, Any]:
        return {
            **self._design_kwargs(),
            "kernel": self.kernel_factory,
            "tile_cap": self.tile_cap,
        }


@dataclass
class BinaryElementwiseOperator(MLIROperator):
    """Base class for binary element-wise AIE operators (two inputs, one output).

    Unlike ChanneledUnaryOperator, binary operators have no explicit num_channels
    parameter — each core uses 2 DMA channels (one per input), so the ShimDMA
    limit is enforced as num_aie_columns * 2 <= 16.

    Subclasses name the mlir-aie kernel factory that writes their compute
    kernel in ``kernel_factory`` (e.g. "add_sized").
    """

    size: int
    tile_size: int
    num_aie_columns: int = 8
    context: AIEContext | None = field(default=None, repr=False)

    kernel_factory: ClassVar[str]
    # Override parent's "c" alias with "col" so binary-elementwise operator names
    # are unambiguous when num_aie_columns and num_channels both appear in the
    # name (the parent ChanneledUnaryOperator uses "c" for num_aie_columns).
    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",  # intentionally overrides parent's "c" alias
    }

    def __post_init__(self) -> None:
        if self.size % (self.num_aie_columns * self.tile_size) != 0:
            raise ValueError(
                f"size ({self.size}) must be a multiple of "
                f"num_aie_columns * tile_size ({self.num_aie_columns * self.tile_size})"
            )
        dev = aie_utils.get_current_device()
        shim_dma_limit = get_shim_dma_limit(dev)
        # Binary operators use 2 ShimDMA channels per column (one per input).
        total_shimdma_channels = self.num_aie_columns * 2
        if total_shimdma_channels > shim_dma_limit:
            raise ValueError(
                f"num_aie_columns ({self.num_aie_columns}) exceeds ShimDMA limit "
                f"of {shim_dma_limit // 2} columns for this device"
            )
        super().__init__(context=self.context)

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]

    def _design_kwargs(self) -> dict[str, Any]:
        """Subclasses with extra parameters (e.g. scalar_factor) add them here."""
        return {
            "num_elements": self.size,
            "num_columns": self.num_aie_columns,
            "tile_size": self.tile_size,
            "trace_size": 0,
        }

    def get_design(self):
        from iron.operators.binary_elementwise_design import binary_elementwise_design

        return binary_elementwise_design

    def get_design_kwargs(self) -> dict[str, Any]:
        return {**self._design_kwargs(), "kernel": self.kernel_factory}
