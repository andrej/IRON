# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator, FusedFullELFCallable
from iron.common.utils import torch_to_numpy
from iron.operators.gemm.op import AIEGEMM
from iron.operators.silu.op import AIESiLU
from iron.operators.elementwise_mul.op import AIEElementwiseMul


class SwiGLUPrefillFusedCallable:
    def __init__(self, op):
        self.op = op
        self.fused_callable = op.fused_op.get_callable()

        # Load weights into sub-buffers (GEMM expects shape (K,N), weights stored as .T)
        w_gate_buf = self.fused_callable.get_buffer("W_gate")
        w_gate_buf.view_as_np()[:] = torch_to_numpy(op.weights_1.T).flatten()

        w_up_buf = self.fused_callable.get_buffer("W_up")
        w_up_buf.view_as_np()[:] = torch_to_numpy(op.weights_2.T).flatten()

        w_down_buf = self.fused_callable.get_buffer("W_down")
        w_down_buf.view_as_np()[:] = torch_to_numpy(op.weights_3.T).flatten()

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


class AIESwiGLUPrefillFused:
    """Single-dispatch fused SwiGLU prefill operator using FusedMLIROperator.

    Chains GEMM -> SiLU -> ElementwiseMul -> GEMM in a single NPU dispatch.
    """

    def __init__(self, seq_len, embedding_dim, hidden_dim, prio_accuracy=False, context=None):
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.prio_accuracy = prio_accuracy
        self.context = context
        self.weights_1 = None  # W_gate: shape (hidden_dim, embedding_dim) (stored as .T)
        self.weights_2 = None  # W_up:   shape (hidden_dim, embedding_dim) (stored as .T)
        self.weights_3 = None  # W_down: shape (embedding_dim, hidden_dim) (stored as .T)
        self.fused_op = None
        # Padded dimensions set during compile
        self.seq_len_padded = None
        self.embedding_dim_padded = None
        self.hidden_dim_padded = None

    def compile(self):
        # Create a dedicated context for ELF compilation
        elf_ctx = AIEContext()

        accuracy_flags = {}
        if self.prio_accuracy:
            accuracy_flags = {
                "emulate_bf16_mmul_with_bfp16": False,
                "prio_accuracy": True,
                "round_conv_even": True,
            }

        # Sub-operators sharing the ELF context
        gemm_1 = AIEGEMM(
            M=self.seq_len,
            K=self.embedding_dim,
            N=self.hidden_dim,
            context=elf_ctx,
            **accuracy_flags,
        )
        self.seq_len_padded = gemm_1.M
        self.embedding_dim_padded = gemm_1.K
        self.hidden_dim_padded = gemm_1.N

        silu = AIESiLU(
            size=gemm_1.M * gemm_1.N,
            num_aie_columns=8,
            tile_size=gemm_1.N // 8,
            context=elf_ctx,
        )

        eltwise_mul = AIEElementwiseMul(
            size=gemm_1.M * gemm_1.N,
            num_aie_columns=8,
            tile_size=gemm_1.N // 8,
            context=elf_ctx,
        )

        gemm_2 = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_dim,
            N=self.embedding_dim,
            context=elf_ctx,
            **accuracy_flags,
        )

        # GEMM arg_spec: [in(M,K), in(K,N), out(M,N)] — A(input), B(weights), C(output)
        # Runlist:
        #   gemm_1(input, W_gate) -> left     (gate projection)
        #   gemm_1(input, W_up)   -> right    (up projection)
        #   silu(left)            -> left_activated
        #   eltwise_mul(left_activated, right) -> intermediate
        #   gemm_2(intermediate, W_down) -> output
        runlist = [
            (gemm_1, "input", "W_gate", "left"),
            (gemm_1, "input", "W_up", "right"),
            (silu, "left", "left_activated"),
            (eltwise_mul, "left_activated", "right", "intermediate"),
            (gemm_2, "intermediate", "W_down", "output"),
        ]

        self.fused_op = FusedMLIROperator(
            name="swiglu_prefill_fused",
            runlist=runlist,
            input_args=["input", "W_gate", "W_up", "W_down"],
            output_args=["output"],
            context=elf_ctx,
        )

        self.fused_op.compile()
        return self

    def get_callable(self):
        return SwiGLUPrefillFusedCallable(self)
