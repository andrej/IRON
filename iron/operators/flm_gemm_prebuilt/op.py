# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict

import aie.utils as aie_utils
from aie.utils.npukernel import NPUKernel

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    InstsBinArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    RemoteFileArtifact,
)

from iron.operators.flm_gemm.design import (
    EPILOGUE_MODES,
    K_TILE,
    MIN_K,
    MIN_M,
    pack_b,
)
from iron.operators.flm_gemm_prebuilt.design import N_TILE

# The FastFlowLM revision the overlay is taken from. A commit SHA rather than
# a branch, so the digest below stays valid.
FASTFLOWLM_COMMIT = "f81eba7140decef5e4eda670d02a91b9d6402ee9"
XCLBIN_PATH = "src/xclbins/Gemma4-E4B-IT-NPU2/mm.xclbin"
XCLBIN_URL = (
    f"https://raw.githubusercontent.com/ROCm/FastFlowLM/{FASTFLOWLM_COMMIT}/"
    f"{XCLBIN_PATH}"
)
XCLBIN_SHA256 = "6f1e5507b84d4545536c9b8281002d0e0e10ed241f8593cb4db50eee63876e5f"
XCLBIN_KERNEL_NAME = "MLIR_AIE"


@dataclass
class FLMGEMMPrebuilt(MLIROperator):
    """bf16 GEMM running FastFlowLM's shipped ``mm`` overlay unmodified.

    The overlay is downloaded rather than built: it exists only as a binary
    xclbin. This operator supplies the other half of a dispatch -- the runtime
    parameters and the shim DMA transfers -- so that the shipped kernel can be
    measured against ``FLMGEMM``, the IRON port of it, at the same shapes.

    B must be pre-packed, as for ``FLMGEMM``; use :meth:`pack_B`.
    """

    M: int
    K: int
    N: int
    # "none" | "gelu" | "silu" | "sigmoid", selected through a runtime
    # parameter rather than at compile time as in FLMGEMM.
    epilogue: str = field(default="none", repr=False)
    # Optional (min, max) applied after the activation.
    clamp: tuple[float, float] | None = field(default=None, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {**MLIROperator._name_aliases}

    def __post_init__(self):
        for name, value, unit in (
            ("M", self.M, MIN_M),
            ("K", self.K, MIN_K),
            ("N", self.N, N_TILE),
        ):
            if value % unit != 0:
                raise ValueError(f"{name} ({value}) must be a multiple of {unit}")
        if self.epilogue not in EPILOGUE_MODES:
            raise ValueError(
                f"epilogue must be one of {sorted(EPILOGUE_MODES)}, "
                f"got {self.epilogue!r}"
            )
        if self.clamp is not None and self.clamp[0] > self.clamp[1]:
            raise ValueError(
                f"clamp min ({self.clamp[0]}) must be <= max ({self.clamp[1]})"
            )
        device = aie_utils.get_current_device()
        if device.resolve().name != "npu2" or device.cols < 8:
            raise NotImplementedError(
                "flm_gemm_prebuilt needs the 8 columns of NPU2 (aie2p); got "
                f"{device.resolve().name!r} with {device.cols} columns"
            )

        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # epilogue and clamp are repr=False, but they change the runtime
        # parameters the sequence writes, so the variants must not share an
        # instruction stream.
        base = super().name
        if self.epilogue != "none":
            base = f"{base}_epi{self.epilogue}"
        if self.clamp is not None:
            lo, hi = self.clamp
            tag = f"{lo:g}_{hi:g}".replace("-", "m").replace(".", "p")
            base = f"{base}_clamp{tag}"
        return base

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "flm_gemm_prebuilt",
                (),
                {
                    "dev": aie_utils.get_current_device(),
                    "M": self.M,
                    "K": self.K,
                    "N": self.N,
                    "epilogue": self.epilogue,
                    "clamp": self.clamp,
                },
            ),
        )

    def get_kernel_artifacts(self):
        return []

    def set_up_artifacts(self) -> None:
        mlir_artifact = self.get_mlir_artifact()
        self.insts_artifact = InstsBinArtifact(
            f"{self.name}.bin",
            mlir_input=mlir_artifact,
            dependencies=[mlir_artifact],
        )
        self.xclbin_artifact = RemoteFileArtifact(
            f"flm_mm_{FASTFLOWLM_COMMIT[:8]}.xclbin",
            url=XCLBIN_URL,
            sha256=XCLBIN_SHA256,
        )
        self.add_artifacts([self.insts_artifact, self.xclbin_artifact])

    def get_callable(self) -> Callable[..., Any]:
        npu_kernel = NPUKernel(
            xclbin_path=self.xclbin_artifact.filename,
            kernel_name=XCLBIN_KERNEL_NAME,
            insts_path=self.insts_artifact.filename,
        )
        handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

        def call(*args):
            return aie_utils.DefaultNPURuntime.run(handle, list(args))

        return call

    def pack_B(self, B):
        """Reorder a row-major ``(K, N)`` weight matrix into the order the B
        transfers read. Returns a flat tensor.

        Identical to ``FLMGEMM.pack_B`` at ``tile_n=128``: the port took this
        layout from the overlay.
        """
        return pack_b(B, N_TILE)

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.M, self.K)),  # A
            # B, pre-packed by pack_B -- same element count, different order.
            AIERuntimeArgSpec("in", (self.K, self.N)),  # B (weights)
            AIERuntimeArgSpec("out", (self.M, self.N)),  # C
        ]

    def reference(self, A, B):
        """CPU reference: ``C = epilogue(A @ B)``."""
        from iron.operators.flm_gemm.reference import reference

        return reference(A, B, self.epilogue, self.clamp)
