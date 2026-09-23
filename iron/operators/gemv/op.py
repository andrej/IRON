# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
)
from iron.common.device_utils import get_kernel_dir


@dataclass
class GEMV(MLIROperator):
    """AIE-accelerated General Matrix-Vector/Vector-Matrix Multiplication layer"""

    M: int
    K: int
    num_aie_columns: int = 1
    tile_size_input: int = 2
    tile_size_output: int | None = None
    num_batches: int = 1
    kernel_vector_size: int = field(default=64, repr=False)
    # Optional fused activation applied to each output tile in the producing core.
    # "none" (default) leaves the output unchanged; "gelu" applies GELU(tanh approx).
    # repr=False keeps operator/artifact names stable for the default path.
    epilogue: str = field(default="none", repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
        "num_batches": "batch",
    }

    def __post_init__(self):
        if self.tile_size_output is None:
            self.tile_size_output = self.tile_size_input

        if not (
            self.tile_size_output % self.tile_size_input == 0
            and self.tile_size_output >= self.tile_size_input
        ):
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        # mv.cc pipelines the k loop on the assumption that it runs at least
        # twice, so VEC_SIZE has to divide K at least twice over. Narrow the
        # vector instead of refusing the shape: llama's attention-scores GEMV
        # has K = head_dim = 64, which the 64-wide default cannot serve.
        while self.kernel_vector_size > 16 and self.K < 2 * self.kernel_vector_size:
            self.kernel_vector_size //= 2
        if self.K % self.kernel_vector_size or self.K < 2 * self.kernel_vector_size:
            raise ValueError(
                f"K ({self.K}) must be a multiple of kernel_vector_size "
                f"({self.kernel_vector_size}) and at least twice as large"
            )
        if self.epilogue not in ("none", "gelu"):
            raise ValueError(
                f"unknown epilogue {self.epilogue!r} (expected 'none' or 'gelu')"
            )
        if self.epilogue == "gelu" and self.tile_size_output % 16 != 0:
            raise ValueError(
                f"gelu epilogue needs tile_size_output % 16 == 0 (got {self.tile_size_output})"
            )

        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # epilogue is repr=False so the default path keeps a stable name, but the fused
        # variant must not share an artifact name with the plain GEMV of the same shape:
        # both would emit the same .mlir/.xclbin, and in a shared build dir a cached unfused
        # build can then satisfy the fused op (running the raw matvec with no activation).
        base = super().name
        if self.epilogue == "none":
            return base
        return f"{base}_epi{self.epilogue}"

    def get_design(self):
        from iron.operators.gemv.design import my_matvec

        return my_matvec

    def get_design_kwargs(self) -> dict[str, Any]:
        # The gelu kernel lives in aie2p/gelu.cc, so the fused epilogue is NPU2-only.
        if self.epilogue == "gelu" and get_kernel_dir() != "aie2p":
            raise NotImplementedError(
                "gemv gelu epilogue is only available on NPU2 (aie2p); "
                f"current kernel dir is {get_kernel_dir()!r}"
            )
        return {
            "cols": self.num_aie_columns,
            "M": self.M,
            "K": self.K,
            "m_input": self.tile_size_input,
            "m_output": self.tile_size_output,
            "num_batches": self.num_batches,
            "kernel_vector_size": self.kernel_vector_size,
            "verbose": getattr(self.context, "mlir_verbose", False),
            "epilogue": self.epilogue,
        }

    def get_arg_spec(self):
        batch_dim = (self.num_batches,) if self.num_batches > 1 else ()
        return [
            AIERuntimeArgSpec("in", batch_dim + (self.M, self.K)),  # matrix
            AIERuntimeArgSpec("in", batch_dim + (self.K,)),  # vector
            AIERuntimeArgSpec("out", batch_dim + (self.M,)),  # output
        ]

    def reference(self, A, B):
        """CPU reference: (optionally batched) matrix-vector product."""
        from iron.operators.gemv.reference import reference

        return reference(A, B)
