# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
import torch
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from aie.iron.kernels import flm_gemma4
from aie.utils.verify import Tolerance

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
)
from iron.common.utils import float_to_name
from iron.operators.flm.lm_head.design import check_shape
from iron.operators.flm.q4nx import GROUP, K_TILE, M_TILE, packed_bytes


@dataclass
class LMHead(MLIROperator):
    """Softcapped logits from a q4nx vocabulary, with the RMS norm folded in.

    ``dim`` in-features, ``vocab`` out-features, ``softcap`` the tanh bound.
    X holds the token followed by its RMS weight. W holds the q4nx vocabulary.
    Y receives the logits.
    """

    dim: int
    vocab: int
    softcap: float
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        check_shape(aie_utils.get_current_device(), self.dim, self.vocab)
        if not (np.isfinite(self.softcap) and self.softcap > 0):
            raise ValueError(f"softcap ({self.softcap}) must be finite and positive")
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        dev = aie_utils.get_current_device().resolve().name
        # The name keys the build cache. The softcap is part of the runtime
        # sequence. The name therefore includes the softcap.
        cap = float_to_name(float(self.softcap))
        return f"FLM_LMHead_d{self.dim}_v{self.vocab}_c{cap}_{dev}"

    def quantized_size(self) -> int:
        """Bytes of q4nx vocabulary the operator reads."""
        return packed_bytes(self.vocab * self.dim)

    def _kernel(self):
        """The flm_gemma4_q4nx_lm_head kernel for dim, which the design and the
        kernel artifact both take."""
        return flm_gemma4.flm_gemma4_q4nx_lm_head(
            dim=self.dim, m_tile=M_TILE, k_tile=K_TILE, group=GROUP
        )

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "lm_head",
                (
                    aie_utils.get_current_device(),
                    self.dim,
                    self.vocab,
                    self.softcap,
                ),
                {"lm_head_kernel": self._kernel()},
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact.from_extern(self._kernel())]

    def get_arg_spec(self):
        # The runtime sequence's argument order: y, w, x.
        return [
            AIERuntimeArgSpec("out", (self.vocab,), dtype=bfloat16),
            AIERuntimeArgSpec(
                "in",
                (self.quantized_size() // np.dtype(np.uint32).itemsize,),
                dtype=np.uint32,
            ),
            AIERuntimeArgSpec("in", (2 * self.dim,), dtype=bfloat16),
        ]

    def _logits(self, w, x, softcap):
        """The float64 logits of X for W, softcapped at softcap."""
        from iron.operators.flm.lm_head.reference import dequantize, reference

        dev = aie_utils.get_current_device()
        w, x = (_host(a) for a in (w, x))
        weights = dequantize(w, self.dim, self.vocab, dev.cols, len(dev.core_rows))
        return reference(weights, x.astype(np.float64), softcap)

    def reference(self, w, x):
        """CPU reference, in float64: the softcapped logits of X for W."""
        return torch.from_numpy(self._logits(w, x, self.softcap))

    def reference_tolerance(self, tanh_error=True):
        """The error bound of README.md's Numerics section.

        The largest logit before the softcap sets the projection's error.
        The tanh approximation adds TANH_ERROR times the softcap. A test with
        a softcap that keeps each tanh argument near zero checks the
        projection alone with tanh_error=False.
        """
        from iron.operators.flm.lm_head.reference import PROJECTION_ERROR, TANH_ERROR

        def bound(w, x):
            uncapped = self._logits(w, x, 1e30)
            limit = PROJECTION_ERROR * np.abs(uncapped).max()
            if tanh_error:
                limit += TANH_ERROR * self.softcap
            return np.full(self.vocab, limit)

        return Tolerance.bounded(bound, note="README.md's Numerics section")


def _host(a):
    """a as a numpy array. A bf16 torch tensor widens to float32, exactly."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu()
        return (a.float() if a.dtype == torch.bfloat16 else a).numpy()
    return np.asarray(a)
