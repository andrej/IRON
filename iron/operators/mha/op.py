# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

import numpy as np

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
)


@dataclass
class MHA(MLIROperator):
    """AIE-accelerated Multi-Head Attention operator"""

    num_heads: int
    seq_len: int
    d: int
    num_KV_heads: int
    num_of_pipelines: int = field(default=1, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_heads": "h",
        "num_KV_heads": "kv",
        "seq_len": "s",
    }

    def __post_init__(self):
        self.B_q = 64
        self.B_kv = 64
        if self.d != 64:
            raise ValueError(f"Only d=64 is supported in this version, got d={self.d}")
        MLIROperator.__init__(self, context=self.context)

    def get_design(self):
        from iron.operators.mha.design import fused_mha

        return fused_mha

    def get_design_kwargs(self) -> dict[str, Any]:
        return {
            "heads": self.num_heads,
            "S_q": self.seq_len,
            "S_kv": self.seq_len,
            "d": self.d,
            "B_q": self.B_q,
            "B_kv": self.B_kv,
            "num_KV_heads": self.num_KV_heads,
            "number_of_pipelines": self.num_of_pipelines,
            "emulate_bf16_mmul_with_bfp16": True,
            "trace_size": 0,
            "verbose": False,
        }

    def get_arg_spec(self):
        seq_padding = self._calculate_seq_padding(self.seq_len, self.num_of_pipelines)
        # design.py declares Q/O as (heads, S_q_pad, d) and K/V as
        # (num_KV_heads, S_kv_pad * d); num_KV_heads == 0 means plain MHA.
        kv_heads = self.num_KV_heads if self.num_KV_heads else self.num_heads
        q_size = self.num_heads * self.d * seq_padding
        kv_size = kv_heads * self.d * seq_padding
        return [
            AIERuntimeArgSpec("in", (q_size,)),  # Q
            AIERuntimeArgSpec("in", (kv_size,)),  # K
            AIERuntimeArgSpec("in", (kv_size,)),  # V
            AIERuntimeArgSpec("out", (q_size,)),  # O
        ]

    def _calculate_seq_padding(self, seq_len, num_pipeline=1):
        return ((seq_len + 63 * num_pipeline) // (64 * num_pipeline)) * (
            64 * num_pipeline
        )

    def _pad_to_multiple_of_64(self, tensor, seq_dim, num_pipeline=1):
        seq_len = tensor.shape[seq_dim]
        padded_seq_len = self._calculate_seq_padding(seq_len, num_pipeline)
        if padded_seq_len == seq_len:
            return tensor

        pad_size = padded_seq_len - seq_len
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[seq_dim] = (0, pad_size)
        return np.pad(tensor, pad_width)

    def _pack_compact_to_padded(
        self, src: np.ndarray, H: int, S: int, S_pad: int, D: int
    ) -> np.ndarray:
        """Pack compact tensor into padded format."""
        dst = src
        if S != S_pad:
            dst = np.zeros((H, S_pad, D), dtype=src.dtype)
            dst[:H, :S, :D] = src
        return dst

    def _unpack_padded_to_compact(
        self, src: np.ndarray, H: int, S: int, S_pad: int, D: int
    ) -> np.ndarray:
        """Unpack padded tensor back to compact format."""
        if S < S_pad:
            return src[:H, :S, :D]
        return src
