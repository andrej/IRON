# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator, FusedFullELFCallable
from iron.common.utils import torch_to_numpy
from iron.operators.gemv.op import AIEGEMV
from iron.operators.silu.op import AIESiLU
from iron.operators.elementwise_mul.op import AIEElementwiseMul


class SwiGLUDecodeFusedCallable:
    def __init__(self, op):
        self.op = op
        self.fused_callable = op.fused_op.get_callable()

        # Load weights into sub-buffers (no transpose for GEMV — weights are (M, K))
        w_gate_buf = self.fused_callable.get_buffer("W_gate")
        w_gate_buf.view_as_np()[:] = torch_to_numpy(op.weights_1).flatten()

        w_up_buf = self.fused_callable.get_buffer("W_up")
        w_up_buf.view_as_np()[:] = torch_to_numpy(op.weights_2).flatten()

        w_down_buf = self.fused_callable.get_buffer("W_down")
        w_down_buf.view_as_np()[:] = torch_to_numpy(op.weights_3).flatten()

        # Sync input buffer (contains weights) to NPU
        self.fused_callable.input_buffer.to("npu")
        self.fused_callable.scratch_buffer.to("npu")

    def __call__(self, input_buf, output_buf):
        # Copy input to the fused input sub-buffer (write directly to memory view)
        input_sub = self.fused_callable.get_buffer("input")
        np.copyto(
            np.frombuffer(input_sub.memory_view, dtype=input_sub.dtype,
                          count=int(np.prod(input_sub.shape))).reshape(input_sub.shape),
            input_buf.view_as_np().flatten(),
        )

        # Force-sync input buffer to NPU (reset on state to allow re-sync)
        self.fused_callable.input_buffer.on = "cpu"
        self.fused_callable.input_buffer.to("npu")

        # Execute the fused operator
        self.fused_callable()

        # Force-sync output buffer from NPU
        self.fused_callable.output_buffer.on = "npu"
        self.fused_callable.output_buffer.to("cpu")
        output_sub = self.fused_callable.get_buffer("output")
        out_np = np.frombuffer(output_sub.memory_view, dtype=output_sub.dtype,
                               count=int(np.prod(output_sub.shape))).reshape(output_sub.shape)
        np.copyto(output_buf.view_as_np().reshape(out_np.shape), out_np)


class AIESwiGLUDecodeFused:
    """Single-dispatch fused SwiGLU decode operator using FusedMLIROperator.

    Chains GEMV -> SiLU -> ElementwiseMul -> GEMV in a single NPU dispatch.
    """

    def __init__(self, embedding_dim, hidden_dim, context=None):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context = context
        self.weights_1 = None  # W_gate: shape (hidden_dim, embedding_dim)
        self.weights_2 = None  # W_up:   shape (hidden_dim, embedding_dim)
        self.weights_3 = None  # W_down: shape (embedding_dim, hidden_dim)
        self.fused_op = None

    def compile(self):
        # Create a dedicated context for ELF compilation
        elf_ctx = AIEContext()

        # Sub-operators sharing the ELF context
        gemv_1 = AIEGEMV(
            M=self.hidden_dim,
            K=self.embedding_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=self.hidden_dim // 8,
            context=elf_ctx,
        )

        silu = AIESiLU(
            size=self.hidden_dim,
            num_aie_columns=8,
            tile_size=self.hidden_dim // 16,
            context=elf_ctx,
        )

        eltwise_mul = AIEElementwiseMul(
            size=self.hidden_dim,
            num_aie_columns=8,
            tile_size=self.hidden_dim // 8,
            context=elf_ctx,
        )

        gemv_2 = AIEGEMV(
            M=self.embedding_dim,
            K=self.hidden_dim,
            num_aie_columns=8,
            tile_size_input=1,
            tile_size_output=self.embedding_dim // 8,
            context=elf_ctx,
        )

        # GEMV arg_spec: [in(M,K), in(K,), out(M,)] — matrix, vector, output
        # Runlist:
        #   gemv_1(W_gate, input) -> left
        #   gemv_1(W_up, input)   -> right
        #   silu(left)            -> left_activated
        #   eltwise_mul(left_activated, right) -> intermediate
        #   gemv_2(W_down, intermediate) -> output
        runlist = [
            (gemv_1, "W_gate", "input", "left"),
            (gemv_1, "W_up", "input", "right"),
            (silu, "left", "left_activated"),
            (eltwise_mul, "left_activated", "right", "intermediate"),
            (gemv_2, "W_down", "intermediate", "output"),
        ]

        self.fused_op = FusedMLIROperator(
            name="swiglu_decode_fused",
            runlist=runlist,
            input_args=["input", "W_gate", "W_up", "W_down"],
            output_args=["output"],
            context=elf_ctx,
        )

        self.fused_op.compile()
        return self

    def get_callable(self):
        return SwiGLUDecodeFusedCallable(self)
