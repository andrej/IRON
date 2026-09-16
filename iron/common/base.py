# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
from ml_dtypes import bfloat16
import aie.utils as aie_utils
from aie.utils.npukernel import NPUKernel

from .context import AIEContext
from .utils import float_to_name
from .compilation import PythonGeneratedMLIRArtifact, build_compilable


class AIEOperatorBase(ABC):
    """Base class for AIE-accelerated operations"""

    _default_context: ClassVar[AIEContext | None] = None

    def __init__(self, context: AIEContext | None = None) -> None:
        if context is None:
            context = self.get_default_context()
        self.context = context
        self.compilable = None

    @abstractmethod
    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        pass

    @abstractmethod
    def get_callable(self) -> Callable[..., Any]:
        pass

    @abstractmethod
    def compile(self) -> "AIEOperatorBase":
        """Build this operator's artifacts, reusing a cached build when one fits."""

    @classmethod
    def get_default_context(cls) -> AIEContext:
        """Return the process-wide default AIEContext, creating it on first call (lazy singleton)."""
        if AIEOperatorBase._default_context is None:
            AIEOperatorBase._default_context = AIEContext()
        return AIEOperatorBase._default_context

    @property
    def work_dir(self) -> Path:
        """Where the compiler left this operator's intermediate files.

        aiecc writes the lowered module and the runtime-parameter table here, so
        tracing and parameter binding read it back after a build.
        """
        if self.compilable is None or self.compilable._kernel_dir is None:
            raise RuntimeError(f"{type(self).__name__} has not been compiled yet")
        return Path(self.compilable._kernel_dir)

    @property
    def elf_path(self) -> Path:
        """The full ELF this operator dispatches through."""
        if self.compilable is None or self.compilable._elf_path is None:
            raise RuntimeError(
                f"{type(self).__name__} has no full ELF; it was not compiled "
                "with full_elf set"
            )
        return Path(self.compilable._elf_path)


def _serialize_param(v: object) -> str:
    """Convert a parameter value to a filesystem-safe string for operator names."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return float_to_name(v)
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v)
    return str(v)


class MLIROperator(AIEOperatorBase):
    """Base class for AIE-accelerated operations defined by a single MLIR source"""

    _name_aliases: ClassVar[dict[str, str]] = {
        "num_aie_columns": "c",
        "num_channels": "ch",
        "tile_size": "t",
        "size": "sz",
        "scalar_factor": "sf",
        "rows": "r",
        "cols": "n",
    }

    @property
    def operator_dir(self) -> Path:
        return Path(inspect.getfile(type(self))).parent

    def design_key(self) -> str | None:
        """Identifies the design this operator compiles to, for sharing it.

        Two operators returning the same key must produce byte-identical MLIR before
        the fused build prefixes their kernel symbols, and must take the same runtime
        argument shapes. ``None`` means the design is never shared.
        """
        return None

    @property
    def name(self) -> str:
        """Unique name for this operator instance, derived from its parameters.

        For @dataclass subclasses the name is automatically constructed from the
        dataclass fields using ``_name_aliases`` to shorten field names.
        Non-dataclass subclasses must override this property directly.
        """
        if dataclasses.is_dataclass(self):
            aliases = type(self)._name_aliases
            parts = (
                f"{aliases.get(f.name, f.name)}{_serialize_param(getattr(self, f.name))}"
                for f in dataclasses.fields(self)
                if f.repr and getattr(self, f.name) is not None
            )
            base = type(self).__name__ + "_" + "_".join(parts)
        else:
            raise NotImplementedError(
                f"{type(self).__name__} must be a @dataclass or override the name property"
            )
        dev = aie_utils.get_current_device()
        return f"{base}_{dev.resolve().name}"

    @abstractmethod
    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        pass

    @abstractmethod
    def get_kernel_artifacts(self) -> list:
        pass

    def build_compilable(self, *, full_elf: bool = False, aiecc_flags=()):
        return build_compilable(
            self.get_mlir_artifact().generator,
            self.get_kernel_artifacts(),
            use_chess=self.context.compiler == "chess",
            full_elf=full_elf,
            aiecc_flags=aiecc_flags,
        )

    def compile(self) -> "MLIROperator":
        self.compilable = self.build_compilable()
        self.compilable.compile()
        return self

    def get_callable(self) -> Callable[..., Any]:
        xclbin_path, insts_path = self.compilable.get_artifacts()
        npu_kernel = NPUKernel(
            xclbin_path=str(xclbin_path),
            kernel_name="MLIR_AIE",
            insts_path=str(insts_path),
        )
        handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

        def call(*args):
            return aie_utils.DefaultNPURuntime.run(handle, list(args))

        return call


class CompositeOperator(AIEOperatorBase):
    """Base class for composite operators that chain multiple sub-operators"""

    def __init__(self, context: AIEContext | None = None) -> None:
        super().__init__(context)


@dataclass(frozen=True)
class AIERuntimeArgSpec:
    """Specification for a single runtime argument of an AIE operator."""

    direction: str
    shape: tuple[int, ...]
    dtype: np.dtype = dataclasses.field(default_factory=lambda: bfloat16)

    def __post_init__(self) -> None:
        if self.direction not in {"in", "out", "inout"}:
            raise ValueError(
                f"Invalid direction {self.direction!r}: must be one of 'in', 'out', 'inout'"
            )
