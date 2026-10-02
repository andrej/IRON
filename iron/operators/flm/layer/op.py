# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from aie.dialects._aie_enum_gen import AIEArch
from aie.iron.kernels import FlmGemma4DecodeGeometry

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.flm.layer.design import (
    LAYER_TYPES,
    MAX_CONTEXT,
    RTP_ADDRESSES,
    arg_sizes,
    layer_kernels,
)


@dataclass
class DecodeLayer(MLIROperator):
    """One Gemma 4 decode layer for one token, as FastFlowLM's engine drives it.

    See README.md for the parameters, the buffers and the dispatch parameters.
    """

    geometry: FlmGemma4DecodeGeometry
    layer_type: str
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.geometry not in RTP_ADDRESSES:
            raise ValueError(
                "geometry must be aie.iron.kernels.FLM_GEMMA4_E2B_DECODE or "
                f"FLM_GEMMA4_E4B_DECODE, not {self.geometry!r}"
            )
        if self.layer_type not in LAYER_TYPES:
            raise ValueError(
                f"layer_type must be one of {LAYER_TYPES}, not {self.layer_type!r}"
            )
        dev = aie_utils.get_current_device()
        if dev.arch != AIEArch.AIE2p:
            raise NotImplementedError("the flm_gemma4_decode kernels are AIE2P only")
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        dev = aie_utils.get_current_device().resolve().name
        return f"FLM_DecodeLayer_{self.geometry.name}_{self.layer_type}_{dev}"

    def get_dispatch_params(self):
        return {"context_len": np.int32, "max_l": np.int32}

    def validate_dispatch_params(self, context_len, max_l):
        """Require 0 < max_l <= MAX_CONTEXT and 0 <= context_len < 2**31 - 1.
        A global or global_skip layer also requires context_len < max_l.

        A global or global_skip layer reads rows 0 to context_len of k and of
        v, rounded up to a multiple of LK rows. v starts at row max_l.
        MAX_CONTEXT is a multiple of LK. The reads of v therefore end at row
        2 * MAX_CONTEXT at most. A global layer also writes row context_len of
        k and of v. A sliding-window layer addresses its ring of SLIDING_WINDOW
        rows modulo the window. The sequence computes context_len + 1 in int32.
        """
        if not 0 < max_l <= MAX_CONTEXT:
            raise ValueError(f"max_l ({max_l}) must be in [1, {MAX_CONTEXT}]")
        if not 0 <= context_len < np.iinfo(np.int32).max:
            raise ValueError(
                f"context_len ({context_len}) must be in "
                f"[0, {np.iinfo(np.int32).max - 1}]"
            )
        if self.layer_type in ("global", "global_skip") and context_len >= max_l:
            raise ValueError(
                f"a {self.layer_type} layer needs context_len ({context_len}) "
                f"< max_l ({max_l})"
            )

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "decode_layer",
                (
                    aie_utils.get_current_device(),
                    self.geometry,
                    RTP_ADDRESSES[self.geometry],
                    self.layer_type,
                ),
                {"kernels": layer_kernels(self.geometry)},
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact.from_extern(fn)
            for fn in layer_kernels(self.geometry).values()
        ]

    def get_arg_spec(self):
        sizes = arg_sizes(self.geometry)
        return [
            AIERuntimeArgSpec("inout", (sizes["x"],), dtype=bfloat16),
            AIERuntimeArgSpec("in", (sizes["proj"],), dtype=bfloat16),
            AIERuntimeArgSpec("in", (sizes["rms"],), dtype=bfloat16),
            AIERuntimeArgSpec("in", (sizes["rope_rms"],), dtype=bfloat16),
            AIERuntimeArgSpec("inout", (sizes["kv"],), dtype=bfloat16),
        ]
