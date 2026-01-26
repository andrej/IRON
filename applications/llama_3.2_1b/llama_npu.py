#!/usr/bin/env python3

import torch
import math
from pathlib import Path
import sys
import numpy as np
import ml_dtypes
import llama_inference_harness as harness
import logging
import time

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from operators.common.context import AIEContext
from operators.common import (
    AIEBuffer
)
from operators.common.utils import torch_to_numpy, numpy_to_torch
from operators.common.base import PatchableSingleXclbinCallable
from operators.common.fusion import FusedMLIROperator, FusedFullELFCallable, load_elf, patch_elf
from operators import (
    AIERMSNorm,
    AIEGEMM,
    AIEGEMV,
    AIEElementwiseAdd
)
from operators.elementwise_mul.op import AIEElementwiseMul
from operators.silu.op import AIESiLU
from operators.rope.op import AIERope
from operators.strided_copy.op import AIEStridedCopy
from operators.repeat.op import AIERepeat
from operators.softmax.op import AIESoftmax
from operators.transpose.op import AIETranspose

logging.basicConfig(level=logging.DEBUG)


# AIE Operator Configuration
# ##########################################################################

aie_ops = None

class AIEPrefillOperations:
    pass

class AIEDecodeOperations:
    pass

class AIELlamaOperators:
    
    def __init__(self, config, prompt_len):
        self.context = AIEContext()
        self.context.build_dir.mkdir(parents=True, exist_ok=True)

        self.prefill = AIEPrefillOperations()
        self.decode = AIEDecodeOperations()

        # RMS Norm
        self.prefill.rms_norm = AIERMSNorm(
            size=prompt_len * config.emb_dim,
            eps=1e-5,
            num_aie_columns=8,
            num_channels=2,
            tile_size=config.emb_dim,
            context=self.context
        ).compile().get_callable()


        # Residual additions
        self.prefill.residual_add = AIEElementwiseAdd(
            size=prompt_len * config.emb_dim,
            tile_size=config.emb_dim
        ).compile().get_callable()
        self.decode.residual_add = AIEElementwiseAdd(
            size=config.emb_dim,
            tile_size=config.emb_dim // 8
        ).compile().get_callable()

        # Final GEMM
        min_N = 64 * 8 * 4  # tile_n * num_aie_columns * partition_N
        config.padded_vocab_size = (config.vocab_size + min_N - 1) // min_N * min_N
        config.vocab_partitions = 4
        self.prefill.gemv_out_head_compilable = AIEGEMM(
            M=prompt_len,
            K=config.emb_dim,
            N=config.padded_vocab_size // config.vocab_partitions,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=True,
            separate_c_tiles=True,
            context=self.context
        ).compile()
        self.prefill.out_head = self.prefill.gemv_out_head_compilable.get_callable()
        self.decode.gemv_out_head = AIEGEMV(
            M=config.vocab_size,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=32,
            context=self.context
        ).compile().get_callable()
        
        # SwiGLU FFN operators
        # Prefill: M=prompt_len, K=emb_dim, N=hidden_dim
        self.prefill.ffn_up_gate = AIEGEMM(
            M=prompt_len,
            K=config.emb_dim,
            N=config.hidden_dim,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=False,  # exceeds stride dimensions otherwise; just transpose weights
            context=self.context
        ).compile().get_callable()
        
        self.prefill.ffn_down = AIEGEMM(
            M=prompt_len,
            K=config.hidden_dim,
            N=config.emb_dim,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=False,  # exceeds stride dimensions otherwise; just transpose weights
            context=self.context
        ).compile().get_callable()
        
        self.prefill.ffn_silu = AIESiLU(
            size=prompt_len * config.hidden_dim,
            tile_size=config.hidden_dim,
            num_aie_columns=8,
            context=self.context
        ).compile().get_callable()
        
        self.prefill.eltwise_mul_ffn = AIEElementwiseMul(
            size=prompt_len * config.hidden_dim,
            tile_size=config.hidden_dim,
            num_aie_columns=8,
            context=self.context
        ).compile().get_callable()
        self.decode.rms_norm = AIERMSNorm(
            size=config.emb_dim,
            eps=1e-5,
            num_aie_columns=1,
            num_channels=2,
            tile_size=config.emb_dim,
            context=self.context
        ).compile().get_callable()        
        
        elf_ctx = AIEContext(build_dir="build_elf")

        # Fused operator for attention projections + RoPE (decode)
        gemv_attn_query_op = AIEGEMV(
            M=config.n_heads * config.head_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.head_dim // 2,
            context=elf_ctx
        )
        
        gemv_attn_key_value_op = AIEGEMV(
            M=config.n_kv_groups * config.head_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.head_dim // 2,
            context=elf_ctx
        )
        
        rope_queries_op = AIERope(
            rows=1 * config.n_heads,
            cols=config.head_dim,
            angle_rows=1,
            context=elf_ctx
        )
        
        rope_keys_op = AIERope(
            rows=1 * config.n_kv_groups,
            cols=config.head_dim,
            angle_rows=1,
            context=elf_ctx
        )
        
        strided_copy_cache_magic = 0xDEADBEE0
        strided_copy_cache_op = AIEStridedCopy(
            input_sizes=(config.n_kv_groups, config.head_dim),
            input_strides=(config.head_dim, 1),
            input_offset=0,
            output_sizes=(1, config.n_kv_groups, config.head_dim),
            output_strides=(0, prompt_len * config.head_dim, 1),
            output_offset=7 * config.head_dim * 2,  # Will be patched at runtime
            input_buffer_size=1 * config.n_kv_groups * config.head_dim,
            output_buffer_size=config.n_kv_groups * prompt_len * config.head_dim,
            num_aie_channels=1,
            output_offset_patch_marker=strided_copy_cache_magic,
            context=elf_ctx
        )
        
        # For decode: per head, (1, head_dim) @ (head_dim, max_context_len)
        # Use GEMV: (max_context_len, head_dim) @ (head_dim,) = (max_context_len,)
        gemv_attn_scores_op = AIEGEMV(
            M=prompt_len,  # max possible context length
            K=config.head_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=prompt_len // 8,
            num_batches=config.n_heads,
            context=elf_ctx
        )
        
        attn_scale_op = AIEElementwiseMul(
            size=config.n_heads * prompt_len,
            tile_size=prompt_len // 8,
            num_aie_columns=8,
            context=elf_ctx
        )
        
        # Softmax operators for attention weights
        softmax_magic = 0xBA5EBA11
        softmax_op = AIESoftmax(
            rows=config.n_heads,
            cols=prompt_len,
            num_aie_columns=1,
            num_channels=1,
            rtp_vector_size=prompt_len,  # Compile with max size
            mask_patch_value=softmax_magic,  # Magic value for patching
            context=elf_ctx
        )
        
        # Fused transpose for all attention heads (decode)
        transpose_values_op = AIETranspose(
            M=prompt_len,
            N=config.head_dim,
            num_aie_columns=2,
            num_channels=1,
            m=256,
            n=32,
            s=8,
            context=elf_ctx
        )
        
        # GEMV for attention context: (head_dim, max_context_len) @ (max_context_len,) = (head_dim,) per head
        gemv_attn_context_op = AIEGEMV(
            M=config.head_dim,
            K=prompt_len,  # max possible context length
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=4,
            num_batches=config.n_heads,
            context=elf_ctx
        )

        gemv_attn_output_op = AIEGEMV(
            M=config.emb_dim,
            K=config.n_heads * config.head_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.emb_dim // 8,
            context=elf_ctx
        )
        
        rms_norm_op = AIERMSNorm(
            size=config.emb_dim,
            eps=1e-5,
            num_aie_columns=1,
            num_channels=2,
            tile_size=config.emb_dim,
            context=elf_ctx
        )
        
        gemv_ffn_up_gate_op = AIEGEMV(
            M=config.hidden_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.hidden_dim // 8,
            context=elf_ctx
        )
        
        gemv_ffn_down_op = AIEGEMV(
            M=config.emb_dim,
            K=config.hidden_dim,
            num_aie_columns=8,
            tile_size_input=1,
            tile_size_output=config.emb_dim // 8,
            context=elf_ctx
        )
        
        silu_ffn_op = AIESiLU(
            size=config.hidden_dim,
            tile_size=config.hidden_dim // 8,
            num_aie_columns=8,
            context=elf_ctx
        )
        
        eltwise_mul_ffn_op = AIEElementwiseMul(
            size=config.hidden_dim,
            tile_size=config.hidden_dim // 8,
            num_aie_columns=8,
            context=elf_ctx
        )
        
        residual_add_op = AIEElementwiseAdd(
            size=config.emb_dim,
            tile_size=config.emb_dim // 8,
            context=elf_ctx
        )
        
        repeat_interleave_op = AIERepeat(
            rows=config.n_kv_groups,
            cols=prompt_len * config.head_dim,  # Max context length
            repeat=config.n_heads // config.n_kv_groups,
            transfer_size=config.head_dim,
            context=elf_ctx
        )
        
        cache_buffer_size = config.n_kv_groups * prompt_len * config.head_dim * 2  # * 2 for bfloat16
        values_per_head_buffer_size = prompt_len * config.head_dim * 2  # * 2 for bfloat16
        values_buffer_size = config.n_heads * values_per_head_buffer_size

        runlist = []
        for layer_idx in range(config.n_layers):       
            runlist.extend(
                [
                    (rms_norm_op,             "x", f"W_norm1_{layer_idx}", "x_norm") # Step 1: RMS normalization
                ] + [
                    # <grouped query attention>
                    (gemv_attn_query_op,     f"W_attn_query_{layer_idx}", "x_norm", "queries"),
                    (gemv_attn_key_value_op, f"W_attn_key_{layer_idx}", "x_norm", "keys"),
                    (gemv_attn_key_value_op, f"W_attn_value_{layer_idx}", "x_norm", "values"),
                    (rope_queries_op,        "queries", "rope_angles", "queries"),
                    (rope_keys_op,           "keys", "rope_angles", "keys"),
                    (strided_copy_cache_op,  "keys", "keys_cache"),
                    (strided_copy_cache_op,  "values", "values_cache"),
                    (repeat_interleave_op,   f"keys_cache_{layer_idx}", "attn_scores_keys"),
                    (repeat_interleave_op,   f"values_cache_{layer_idx}", "attn_scores_values"),
                    (gemv_attn_scores_op,    "attn_scores_keys", "queries", "attn_scores"),
                    (attn_scale_op,          "attn_scores", "attn_scale_factor", "attn_scores"),
                    (softmax_op,             "attn_scores", "attn_weights")
                ] + [
                    (transpose_values_op,
                        f"attn_scores_values[{h * values_per_head_buffer_size}:{(h + 1) * values_per_head_buffer_size}]",
                        f"attn_scores_values_transposed[{h * values_per_head_buffer_size}:{(h + 1) * values_per_head_buffer_size}]"
                    )
                    for h in range(config.n_heads)
                ] + [
                    (gemv_attn_context_op,   "attn_scores_values_transposed", "attn_weights", "attn_context"),
                    (gemv_attn_output_op,    f"W_attn_output_decode_{layer_idx}", "attn_context", "attn_output")
                    # </grouped query attention>
                ] + [
                    # <post attention block>
                    (residual_add_op,        "x", "attn_output", "x"),
                    (rms_norm_op,            "x", f"W_norm2_{layer_idx}", "x_norm"),
                    (gemv_ffn_up_gate_op,    f"W_ffn_gate_{layer_idx}", "x_norm", "ffn_gate"),
                    (gemv_ffn_up_gate_op,    f"W_ffn_up_{layer_idx}", "x_norm", "ffn_up"),
                    (silu_ffn_op,            "ffn_gate", "ffn_gate"),
                    (eltwise_mul_ffn_op,     "ffn_gate", "ffn_up", "ffn_hidden"),
                    (gemv_ffn_down_op,       f"W_ffn_down_{layer_idx}", "ffn_hidden", "ffn_output"),
                    (residual_add_op,        "x", "ffn_output", "x"),
                    # </post attention block>
                ]
            )
            
        self.decode.attn_fused_op = FusedMLIROperator(
            "attn_fused_op",
            runlist,
            input_args=[
                f"W_norm1_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"W_attn_query_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"W_attn_key_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"W_attn_value_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"W_norm2_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"W_ffn_gate_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"W_ffn_up_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"W_ffn_down_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"keys_cache_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                f"values_cache_{layer_idx}" for layer_idx in range(config.n_layers)
            ] + [
                "rope_angles",
                "attn_scale_factor",
            ],
            output_args=[],
            buffer_sizes={
                **{
                    "keys_cache_{layer_idx}": cache_buffer_size
                    for layer_idx in range(config.n_layers)
                },
                **{
                    "values_cache_{layer_idx}": cache_buffer_size
                    for layer_idx in range(config.n_layers)
                },
                **{
                    "attn_scores_values": values_buffer_size,
                    "attn_scores_values_transposed": values_buffer_size
                }
            },
            context=elf_ctx
        ).compile()

        self.decode.attn_fused_elf_data = load_elf(self.decode.attn_fused_op)
        
        def get_patch_locs(elf_data, magic):
            return [i for i, x in enumerate(elf_data) if magic & 0xFFFFFFFF == x]

        # Extract patch offsets for strided_copy operations in fused operator
        _, keys_cache_offs, _ = self.decode.attn_fused_op.get_layout_for_buffer("keys_cache")
        _, values_cache_offs, _ = self.decode.attn_fused_op.get_layout_for_buffer("values_cache")
        keys_patches = {
            l: keys_cache_offs
            for l in get_patch_locs(self.decode.attn_fused_elf_data, (keys_cache_offs + strided_copy_cache_magic * 2))
        }
        values_patches = {
            l: values_cache_offs
            for l in get_patch_locs(self.decode.attn_fused_elf_data, (values_cache_offs + strided_copy_cache_magic * 2))
        }
        no_offset_patches = {
            l: 0
            for l in get_patch_locs(self.decode.attn_fused_elf_data, (strided_copy_cache_magic * 2))
        }
        self.decode.attn_fused_patch_locations = {**keys_patches, **values_patches, **no_offset_patches}
        assert len(self.decode.attn_fused_patch_locations) == 4 * config.n_layers + 2

        self.decode.softmax_patch_offsets = get_patch_locs(self.decode.attn_fused_elf_data, softmax_magic)
        assert len(self.decode.softmax_patch_offsets) == config.n_layers + 1

        patch_operators_for_decode(self.decode, config, 128)
        
        # Attention score scaling operators
        # FIXME: Using elementwise mul is very wasteful (of bandwidth) here since it's the same scalar factor for all values; need a kernel that allows scalar multiplication of a vector; maybe use AXPY
        self.prefill.attn_scale = AIEElementwiseMul(
            size=config.n_heads * prompt_len * prompt_len,
            tile_size=prompt_len,
            num_aie_columns=8,
            context=self.context
        ).compile().get_callable()
        
        # RoPE operators
        # For queries: (seq_len, num_heads * head_dim) = (seq_len, 2048)
        # For keys: (seq_len, num_kv_groups * head_dim) = (seq_len, 512)
        # angle_rows=1 because all rows use the same angle row (angles are per position)
        self.prefill.rope_queries = AIERope(
            rows=prompt_len * config.n_heads,
            cols=config.head_dim,
            angle_rows=prompt_len,
            context=self.context
        ).compile().get_callable()
        
        self.prefill.rope_keys = AIERope(
            rows=prompt_len * config.n_kv_groups,
            cols=config.head_dim,
            angle_rows=prompt_len,
            context=self.context
        ).compile().get_callable()
        
        self.decode.rope_queries = AIERope(
            rows=1 * config.n_heads,
            cols=config.head_dim,
            angle_rows=1,
            context=self.context
        ).compile().get_callable()
        
        self.decode.rope_keys = AIERope(
            rows=1 * config.n_kv_groups,
            cols=config.head_dim,
            angle_rows=1,
            context=self.context
        ).compile().get_callable()
        
        # Attention projection operators
        # Query projection: (seq_len, emb_dim) -> (seq_len, n_heads * head_dim)
        self.prefill.attn_query = AIEGEMM(
            M=prompt_len,
            K=config.emb_dim,
            N=config.n_heads * config.head_dim,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=False,
            context=self.context
        ).compile().get_callable()
        
        self.decode.gemv_attn_query = AIEGEMV(
            M=config.n_heads * config.head_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.head_dim // 2,
            context=self.context
        ).compile().get_callable()
        
        # Key projection: (seq_len, emb_dim) -> (seq_len, n_kv_groups * head_dim)
        self.prefill.attn_key = AIEGEMM(
            M=prompt_len,
            K=config.emb_dim,
            N=config.n_kv_groups * config.head_dim,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=False,
            context=self.context
        ).compile().get_callable()
        
        self.decode.gemv_attn_key_value = AIEGEMV(
            M=config.n_kv_groups * config.head_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.head_dim // 2,
            context=self.context
        ).compile().get_callable()
        
        # Value projection: (seq_len, emb_dim) -> (seq_len, n_kv_groups * head_dim)
        self.prefill.attn_value = AIEGEMM(
            M=prompt_len,
            K=config.emb_dim,
            N=config.n_kv_groups * config.head_dim,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=False,
            context=self.context
        ).compile().get_callable()
        
        # Attention score computation: Q @ K^T per head
        # For prefill: (seq_len, head_dim) @ (head_dim, seq_len) = (seq_len, seq_len) per head
        self.prefill.attn_scores = AIEGEMM(
            M=prompt_len,
            K=config.head_dim,
            N=prompt_len,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=False,
            context=self.context
        ).compile().get_callable()
        
        # Transpose values from (max_context_len, head_dim) to (head_dim, max_context_len) per head
        self.decode.transpose_values = AIETranspose(
            M=prompt_len,
            N=config.head_dim,
            num_aie_columns=2,
            num_channels=1,
            m=256,
            n=32,
            s=8,
            context=self.context
        ).compile().get_callable()
        


# Allocate buffers shared with NPU
# ##########################################################################

aie_buffers = None

class AIEPrefillBuffers:
    def __init__(self, prompt_len, emb_dim, hidden_dim, n_heads, n_kv_groups, head_dim):
        self.x = AIEBuffer(shape=(prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        self.x_norm = AIEBuffer(shape=(prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        self.attn_output = AIEBuffer(shape=(prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_output = AIEBuffer(shape=(prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        # SwiGLU intermediate buffers
        self.ffn_gate = AIEBuffer(shape=(prompt_len, hidden_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_up = AIEBuffer(shape=(prompt_len, hidden_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_hidden = AIEBuffer(shape=(prompt_len, hidden_dim), dtype=ml_dtypes.bfloat16)
        # Attention buffers: queries and keys serve as both projection output and RoPE input/output
        self.queries = AIEBuffer(shape=(prompt_len * n_heads, head_dim), dtype=ml_dtypes.bfloat16)
        self.keys = AIEBuffer(shape=(prompt_len * n_kv_groups, head_dim), dtype=ml_dtypes.bfloat16)
        self.values = AIEBuffer(shape=(prompt_len, n_kv_groups * head_dim), dtype=ml_dtypes.bfloat16)
        self.rope_angles = AIEBuffer(shape=(prompt_len, head_dim), dtype=ml_dtypes.bfloat16)
        # Attention score computation buffers (per-head) - parent buffers with subbuffers
        # Parent buffer for all heads' queries: (n_heads, prompt_len, head_dim) stored contiguously
        self.attn_scores_queries_all = AIEBuffer(shape=(n_heads * prompt_len, head_dim), dtype=ml_dtypes.bfloat16)
        self.attn_scores_queries_per_head = [
            self.attn_scores_queries_all.subbuffer(
                length=prompt_len * head_dim,
                offset=h * prompt_len * head_dim,
                shape=(prompt_len, head_dim)
            )
            for h in range(n_heads)
        ]
        # Parent buffer for all KV groups' keys: (n_kv_groups, head_dim, prompt_len) stored contiguously
        self.attn_scores_keys_all = AIEBuffer(shape=(n_kv_groups * head_dim, prompt_len), dtype=ml_dtypes.bfloat16)
        self.attn_scores_keys_per_kv_group = [
            self.attn_scores_keys_all.subbuffer(
                length=head_dim * prompt_len,
                offset=g * head_dim * prompt_len,
                shape=(head_dim, prompt_len)
            )
            for g in range(n_kv_groups)
        ]
        # Parent buffer for all heads' scores: (n_heads * prompt_len, prompt_len)
        self.attn_scores = AIEBuffer(shape=(n_heads * prompt_len, prompt_len), dtype=ml_dtypes.bfloat16)
        self.attn_scores_per_head = [
            self.attn_scores.subbuffer(
                length=prompt_len * prompt_len,
                offset=h * prompt_len * prompt_len,
                shape=(prompt_len, prompt_len)
            )
            for h in range(n_heads)
        ]
        # Attention score scaling buffer (pre-initialized with 1/sqrt(head_dim))
        scale_factor = 1.0 / math.sqrt(head_dim)
        self.attn_scale_factor = AIEBuffer(shape=(n_heads * prompt_len, prompt_len), dtype=ml_dtypes.bfloat16)
        self.attn_scale_factor.view_as_torch()[:] = scale_factor
        self.attn_scale_factor.to("npu")
        # Attention weights buffer (output of softmax)
        self.attn_weights = AIEBuffer(shape=(n_heads * prompt_len, prompt_len), dtype=ml_dtypes.bfloat16)

class AIEDecodeBuffers:
    def __init__(self, emb_dim, hidden_dim, n_heads, n_kv_groups, head_dim, max_context_len):
        self.x = AIEBuffer(shape=(1, emb_dim), dtype=ml_dtypes.bfloat16)
        self.x_norm = AIEBuffer(shape=(1, emb_dim), dtype=ml_dtypes.bfloat16)
        self.attn_output = AIEBuffer(shape=(1, emb_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_output = AIEBuffer(shape=(1, emb_dim), dtype=ml_dtypes.bfloat16)
        # SwiGLU intermediate buffers
        self.ffn_gate = AIEBuffer(shape=(1, hidden_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_up = AIEBuffer(shape=(1, hidden_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_hidden = AIEBuffer(shape=(1, hidden_dim), dtype=ml_dtypes.bfloat16)
        # Attention buffers: queries and keys serve as both projection output and RoPE input/output
        self.queries = AIEBuffer(shape=(1 * n_heads, head_dim), dtype=ml_dtypes.bfloat16)
        self.keys = AIEBuffer(shape=(1 * n_kv_groups, head_dim), dtype=ml_dtypes.bfloat16)
        self.values = AIEBuffer(shape=(1, n_kv_groups * head_dim), dtype=ml_dtypes.bfloat16)
        self.rope_angles = AIEBuffer(shape=(1, head_dim), dtype=ml_dtypes.bfloat16)
        # Attention score computation buffers (batched)
        self.attn_scores_keys = AIEBuffer(shape=(n_heads, max_context_len, head_dim), dtype=ml_dtypes.bfloat16)
        self.attn_scores_values = AIEBuffer(shape=(n_heads, max_context_len, head_dim), dtype=ml_dtypes.bfloat16)
        self.attn_scores_values_transposed = AIEBuffer(shape=(n_heads, head_dim, max_context_len), dtype=ml_dtypes.bfloat16)
        # Create per-head subbuffers for transpose operations (to avoid allocating in hot path)
        self.attn_scores_values_per_head = [
            self.attn_scores_values.subbuffer(
                length=max_context_len * head_dim,
                offset=h * max_context_len * head_dim,
                shape=(max_context_len, head_dim)
            )
            for h in range(n_heads)
        ]
        self.attn_scores_values_transposed_per_head = [
            self.attn_scores_values_transposed.subbuffer(
                length=head_dim * max_context_len,
                offset=h * head_dim * max_context_len,
                shape=(head_dim, max_context_len)
            )
            for h in range(n_heads)
        ]
        self.attn_context = AIEBuffer(shape=(n_heads, head_dim), dtype=ml_dtypes.bfloat16)
        self.attn_context_concat = AIEBuffer(shape=(n_heads * head_dim,), dtype=ml_dtypes.bfloat16)
        self.attn_scores = AIEBuffer(shape=(n_heads, max_context_len), dtype=ml_dtypes.bfloat16)
        # Attention score scaling buffer (pre-initialized with 1/sqrt(head_dim))
        scale_factor = 1.0 / math.sqrt(head_dim)
        self.attn_scale_factor = AIEBuffer(shape=(n_heads, max_context_len), dtype=ml_dtypes.bfloat16)
        self.attn_scale_factor.view_as_torch()[:] = scale_factor
        self.attn_scale_factor.to("npu")
        self.attn_weights = AIEBuffer(shape=(n_heads, max_context_len), dtype=ml_dtypes.bfloat16)

class AIELlamaBuffers:
    def __init__(self, config, prompt_len):
        # Vector of the current token(s) being processed through the pipeline
        self.prefill = AIEPrefillBuffers(prompt_len, config.emb_dim, config.hidden_dim, config.n_heads, config.n_kv_groups, config.head_dim)
        self.decode = AIEDecodeBuffers(config.emb_dim, config.hidden_dim, config.n_heads, config.n_kv_groups, config.head_dim, prompt_len)

        # Per-layer KV cache buffers on NPU (used by strided copy for transpose and concatenate)
        self.keys_cache = [
            AIEBuffer(shape=(config.n_kv_groups, prompt_len, config.head_dim), dtype=ml_dtypes.bfloat16)
            for _ in range(config.n_layers)
        ]
        self.values_cache = [
            AIEBuffer(shape=(config.n_kv_groups, prompt_len, config.head_dim), dtype=ml_dtypes.bfloat16)
            for _ in range(config.n_layers)
        ]

        # Transformer block layer-wise RMS norm
        self.W_norm1 = []
        self.W_norm2 = []
        # Attention projection weights
        self.W_attn_query_prefill = []
        self.W_attn_query_decode = []
        self.W_attn_key_prefill = []
        self.W_attn_key_decode = []
        self.W_attn_value_prefill = []
        self.W_attn_value_decode = []
        self.W_attn_output_decode = []
        # SwiGLU FFN weights
        self.W_ffn_gate_prefill = []
        self.W_ffn_up_prefill = []
        self.W_ffn_down_prefill = []
        self.W_ffn_gate_decode = []
        self.W_ffn_up_decode = []
        self.W_ffn_down_decode = []
        for layer_idx in range(config.n_layers):
            self.W_norm1.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.input_layernorm.weight']).to("npu")
            )
            self.W_norm2.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.post_attention_layernorm.weight']).to("npu")
            )
            self.W_attn_query_decode.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.self_attn.q_proj.weight']).to("npu")
            )
            self.W_attn_query_prefill.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.self_attn.q_proj.weight'].T).to("npu")
            )
            self.W_attn_key_decode.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.self_attn.k_proj.weight']).to("npu")
            )
            self.W_attn_key_prefill.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.self_attn.k_proj.weight'].T).to("npu")
            )
            self.W_attn_value_decode.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.self_attn.v_proj.weight']).to("npu")
            )
            self.W_attn_value_prefill.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.self_attn.v_proj.weight'].T).to("npu")
            )
            self.W_attn_output_decode.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.self_attn.o_proj.weight']).to("npu")
            )
            self.W_ffn_gate_decode.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.mlp.gate_proj.weight']).to("npu")
            )
            self.W_ffn_up_decode.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.mlp.up_proj.weight']).to("npu")
            )
            self.W_ffn_down_decode.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.mlp.down_proj.weight']).to("npu")
            )
            self.W_ffn_gate_prefill.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.mlp.gate_proj.weight'].T).to("npu")
            )
            self.W_ffn_up_prefill.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.mlp.up_proj.weight'].T).to("npu")
            )
            self.W_ffn_down_prefill.append(
                AIEBuffer.from_torch(config.weights[f'model.layers.{layer_idx}.mlp.down_proj.weight'].T).to("npu")
            )

        # Final RMS norm weights
        self.W_final_norm = AIEBuffer.from_torch(config.weights['model.norm.weight']).to("npu")
        # Final linear layer
        self.W_out_head = AIEBuffer.from_torch(config.weights['model.embed_tokens.weight']).to("npu")  # unpadded/unpartitioned, used by GEMV
        W_out_head_parts = aie_ops.prefill.gemv_out_head_compilable.partition_B(
            torch_to_numpy(config.weights['model.embed_tokens.weight']), 
            config.vocab_partitions
        )
        self.W_out_head_parts = [
            AIEBuffer.from_np(W_out_head_part).to("npu") 
            for W_out_head_part in W_out_head_parts
        ] # partitioned, padded parts of weight, used by GEMM
        self.prefill.logits = AIEBuffer(shape=(config.vocab_partitions, prompt_len, config.padded_vocab_size // config.vocab_partitions)).to("npu")
        self.prefill.logits_parts = [
            self.prefill.logits.subbuffer(
                length=prompt_len * (config.padded_vocab_size // config.vocab_partitions),
                offset=i * prompt_len * (config.padded_vocab_size // config.vocab_partitions),
                shape=(prompt_len, config.padded_vocab_size // config.vocab_partitions),
            )
            for i in range(config.vocab_partitions)
        ]
        self.decode.logits = AIEBuffer(shape=(config.vocab_size,))


# Prefill
# ##########################################################################

def grouped_query_attention_forward_prefill(
    config,
    x, 
    keys_cache,
    values_cache,
    layer_idx,
    mask=None,
):
    batch, seq_len, emb_dim = x.shape
    num_preceding_tokens = keys_cache.shape[2]

    # Step 1: Linear projections
    aie_ops.prefill.attn_query(aie_buffers.prefill.x_norm, aie_buffers.W_attn_query_prefill[layer_idx], aie_buffers.prefill.queries)
    aie_ops.prefill.attn_key(aie_buffers.prefill.x_norm, aie_buffers.W_attn_key_prefill[layer_idx], aie_buffers.prefill.keys)
    aie_ops.prefill.attn_value(aie_buffers.prefill.x_norm, aie_buffers.W_attn_value_prefill[layer_idx], aie_buffers.prefill.values)
    
    # Step 2: Apply RoPE to queries and keys
    aie_ops.prefill.rope_queries(aie_buffers.prefill.queries, aie_buffers.prefill.rope_angles, aie_buffers.prefill.queries)
    aie_ops.prefill.rope_keys(aie_buffers.prefill.keys, aie_buffers.prefill.rope_angles, aie_buffers.prefill.keys)
    
    # Read results from NPU
    queries = aie_buffers.prefill.queries.to("cpu").view_as_torch()[:seq_len * config.n_heads, :]
    keys = aie_buffers.prefill.keys.to("cpu").view_as_torch()[:seq_len * config.n_kv_groups, :]
    values = aie_buffers.prefill.values.to("cpu").view_as_torch()[:seq_len, :]  # (seq_len, n_kv_groups * head_dim)
    queries = queries.view(batch, seq_len, config.n_heads, config.head_dim)
    keys = keys.unsqueeze(0).view(batch, seq_len, config.n_kv_groups, config.head_dim)
    values = values.unsqueeze(0).view(batch, seq_len, config.n_kv_groups, config.head_dim) # (batch, seq_len, num_kv_groups, head_dim)

    # Step 3: Transpose for attention computation
    # As a result of the attention projections, the queries, keys and values for each head are interspersed with each other.
    # Transpose so that heads are consecutive for attention computation: 
    # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
    queries = queries.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    keys = keys.transpose(1, 2)        # (batch, num_kv_groups, seq_len, head_dim)
    values = values.transpose(1, 2)    # (batch, num_kv_groups, seq_len, head_dim)

    # Step 4: Combine newly computed keys/values for most recent token with cache; these values are used as the updated cache and will be returned to use in the next iteration.
    keys_cache = torch.cat([keys_cache, keys], dim=2)
    values_cache = torch.cat([values_cache, values], dim=2)
    keys = keys_cache
    values = values_cache
    
    # Step 5: Repeat keys and values for grouped attention -- multiple queries get the same key/value
    group_size = config.n_heads // config.n_kv_groups
    values = values.repeat_interleave(group_size, dim=1)
    context_len = keys.shape[2]
    
    # Step 6: Compute attention scores using NPU (per-head)
    # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, context_len)
    # -> (batch, num_heads, seq_len, context_len)
    
    queries_buf = aie_buffers.prefill.attn_scores_queries_all.view_as_torch().view(
        config.n_heads, -1, config.head_dim
    )
    queries_buf[:, :seq_len, :] = queries.squeeze(0)[:, :seq_len, :] # (num_heads, seq_len, head_dim)
    keys_buf = aie_buffers.prefill.attn_scores_keys_all.view_as_torch().view(
        config.n_kv_groups, config.head_dim, -1
    )
    keys_buf[:, :, :context_len] = keys.squeeze(0).transpose(-2, -1) # (num_kv_groups, head_dim, context_len)
    
    # Transfer parent buffers to NPU once
    aie_buffers.prefill.attn_scores_queries_all.to("npu")
    aie_buffers.prefill.attn_scores_keys_all.to("npu")
    aie_buffers.prefill.attn_scores.to("npu")
    
    # Execute GEMM for each head using sub-buffers
    for h in range(config.n_heads):
        kv_group = h // group_size
        aie_ops.prefill.attn_scores(
            aie_buffers.prefill.attn_scores_queries_per_head[h],
            aie_buffers.prefill.attn_scores_keys_per_kv_group[kv_group],
            aie_buffers.prefill.attn_scores_per_head[h]
        )
    
    # Read back all results at once from parent buffer and apply scaling on NPU
    aie_ops.prefill.attn_scale(aie_buffers.prefill.attn_scores, aie_buffers.prefill.attn_scale_factor, aie_buffers.prefill.attn_scores)
    aie_buffers.prefill.attn_scores.to("cpu")
    # Buffer is (n_heads * max_seq_len, max_seq_len), view as (n_heads, max_seq_len, max_seq_len) then slice
    max_seq_len = aie_buffers.prefill.attn_scores.shape[0] // config.n_heads
    scores = aie_buffers.prefill.attn_scores.view_as_torch().view(config.n_heads, max_seq_len, max_seq_len).unsqueeze(0)[:, :, :seq_len, :context_len]
    
    # Step 7: Apply mask
    # This ensures causality, so that tokens in the future cannot attend to tokens in the past.
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Step 8: Apply softmax on CPU
    scores = torch.softmax(scores.to(torch.float32), dim=-1).to(torch.bfloat16)
    attention_weights = scores
    
    # Step 9: Compute attention output
    # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
    # -> (batch, num_heads, seq_len, head_dim)
    context = torch.matmul(attention_weights, values)
    
    # Step 10: Concatenate heads and project
    # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads * head_dim)
    context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    
    output = torch.nn.functional.linear(context, config.weights[f'model.layers.{layer_idx}.self_attn.o_proj.weight'])
    
    return output, keys_cache, values_cache


def swiglu_ffn_forward_prefill(layer_idx):
    # Step 1: Gate projection
    aie_ops.prefill.ffn_up_gate(aie_buffers.prefill.x_norm, aie_buffers.W_ffn_gate_prefill[layer_idx], aie_buffers.prefill.ffn_gate)
    
    # Step 2: Up projection
    aie_ops.prefill.ffn_up_gate(aie_buffers.prefill.x_norm, aie_buffers.W_ffn_up_prefill[layer_idx], aie_buffers.prefill.ffn_up)
    
    # Step 3: Apply SiLU activation
    aie_ops.prefill.ffn_silu(aie_buffers.prefill.ffn_gate, aie_buffers.prefill.ffn_gate)
    
    # Step 4: Element-wise multiplication
    aie_ops.prefill.eltwise_mul_ffn(aie_buffers.prefill.ffn_gate, aie_buffers.prefill.ffn_up, aie_buffers.prefill.ffn_hidden)
    
    # Step 5: Down projection
    aie_ops.prefill.ffn_down(aie_buffers.prefill.ffn_hidden, aie_buffers.W_ffn_down_prefill[layer_idx], aie_buffers.prefill.ffn_output)


def transformer_block_forward_prefill(
    config,
    seq_len,
    layer_idx,
    attn_keys_cache,
    attn_values_cache,
    attn_mask
):
    # Step 1: RMS normalization
    aie_ops.prefill.rms_norm(aie_buffers.prefill.x, aie_buffers.W_norm1[layer_idx], aie_buffers.prefill.x_norm)
    aie_buffers.prefill.x_norm.to("cpu")
    x_norm = aie_buffers.prefill.x_norm.view_as_torch().unsqueeze(0)[:, :seq_len, :]

    # Step 2: Attention
    attn_output, attn_keys, attn_values = grouped_query_attention_forward_prefill(
        config,
        x_norm,
        attn_keys_cache,
        attn_values_cache,
        layer_idx,
        attn_mask,
    )
    
    # Step 3: Residual
    aie_buffers.prefill.attn_output.view_as_torch().unsqueeze(0)[0, :seq_len, :] = attn_output
    aie_ops.prefill.residual_add(aie_buffers.prefill.x, aie_buffers.prefill.attn_output, aie_buffers.prefill.x)
    x = aie_buffers.prefill.x.to("cpu").view_as_torch().unsqueeze(0)[:, :seq_len, :]
    
    # Step 4: Post-norm
    aie_buffers.prefill.x.view_as_torch().unsqueeze(0)[0, :seq_len, :] = x
    aie_ops.prefill.rms_norm(aie_buffers.prefill.x, aie_buffers.W_norm2[layer_idx], aie_buffers.prefill.x_norm)
    aie_buffers.prefill.x_norm.to("cpu")
    x_norm = aie_buffers.prefill.x_norm.view_as_torch().unsqueeze(0)[:, :seq_len, :]
    
    # Step 5: Feed-forward network
    swiglu_ffn_forward_prefill(layer_idx)
    
    # Step 6: Residual
    aie_ops.prefill.residual_add(aie_buffers.prefill.x, aie_buffers.prefill.ffn_output, aie_buffers.prefill.x)
    
    return attn_keys, attn_values


def llama_forward_pass_prefill(
    config,
    state
):
    batch, seq_len = state.token_ids.shape
    
    # Step 1: RoPE angles
    num_preceding_tokens = state.attn_keys_caches[0].shape[2]
    angles_slice = config.angles[num_preceding_tokens : num_preceding_tokens + seq_len]
    aie_buffers.prefill.rope_angles.view_as_torch()[:seq_len, :] = angles_slice

    # Step 2: Token embedding
    tok_emb_weight = config.weights['model.embed_tokens.weight']
    x = torch.nn.functional.embedding(state.token_ids, tok_emb_weight)
    attn_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
        diagonal=1
    )
    aie_buffers.prefill.x.view_as_torch().unsqueeze(0)[0, :seq_len, :] = x

    # Step 3: Transformer blocks
    for layer_idx in range(config.n_layers):
        state.attn_keys_caches[layer_idx], state.attn_values_caches[layer_idx] = transformer_block_forward_prefill(
            config,
            seq_len,
            layer_idx,
            state.attn_keys_caches[layer_idx],
            state.attn_values_caches[layer_idx],
            attn_mask=attn_mask,
        )

    # Step 4: Final normalization
    aie_ops.prefill.rms_norm(aie_buffers.prefill.x, aie_buffers.W_final_norm, aie_buffers.prefill.x)
    
    # Step 5: Output projection
    for i in range(config.vocab_partitions):
        aie_ops.prefill.out_head(aie_buffers.prefill.x, aie_buffers.W_out_head_parts[i], aie_buffers.prefill.logits_parts[i])
    aie_buffers.prefill.logits.to("cpu")
    logits_padded_partitioned = aie_buffers.prefill.logits.view_as_torch()
    logits_padded = logits_padded_partitioned.transpose(0, 1).contiguous().view(-1, config.padded_vocab_size)
    logits = logits_padded.unsqueeze(0)[:,:seq_len,:config.vocab_size]

    # Step 6: Initialize per-layer NPU cache buffers with current cache state for decode phase
    for layer_idx in range(config.n_layers):
        cache_len = state.attn_keys_caches[layer_idx].shape[2]
        aie_buffers.keys_cache[layer_idx].view_as_torch()[:, :cache_len, :] = state.attn_keys_caches[layer_idx].squeeze(0)
        aie_buffers.values_cache[layer_idx].view_as_torch()[:, :cache_len, :] = state.attn_values_caches[layer_idx].squeeze(0)
        aie_buffers.keys_cache[layer_idx].to("npu")
        aie_buffers.values_cache[layer_idx].to("npu")

    return logits, state


# Decode
# ##########################################################################

def patch_operators_for_decode(ops, config, num_preceding_tokens):
    context_len = num_preceding_tokens + 1
    
    # Patch fused operator for strided copy cache offset
    output_offset = num_preceding_tokens * config.head_dim
    offset_val = output_offset * 2  # Multiply by 2 for bfloat16 byte offset
    strided_copy_patches = { 
        i: (base + offset_val, 0xFFFFFFFF)
        for i, base in ops.attn_fused_patch_locations.items()
    }
    softmax_patches = {
        i: (context_len, 0xFFFFFFFF)
        for i in ops.softmax_patch_offsets
    }
    patches = {**strided_copy_patches, **softmax_patches}
    patched_elf_data = ops.attn_fused_elf_data.copy()
    patch_elf(patched_elf_data, patches)

    ops.attn_fused = FusedFullELFCallable(
        ops.attn_fused_op,
        elf_data=patched_elf_data
    )


def llama_forward_pass_decode(config, state):
    batch, seq_len = state.token_ids.shape
    assert seq_len == 1 

    #patch_operators_for_decode(aie_ops.decode, config, state.num_preceding_tokens)

    # Step 1: Prefill RoPE angle look-up tables
    angles_slice = config.angles[state.num_preceding_tokens : state.num_preceding_tokens + seq_len]
    aie_buffers.decode.rope_angles.view_as_torch()[:] = angles_slice

    # Step 2: Token embedding (on CPU)
    tok_emb_weight = config.weights['model.embed_tokens.weight']
    x = torch.nn.functional.embedding(state.token_ids, tok_emb_weight)
    aie_buffers.decode.x.view_as_torch().unsqueeze(0)[0, :seq_len, :] = x

    # Step 3: Transformer blocks
    transformer_blocks_forward_decode(
        config,
        state.num_preceding_tokens,
    )
    aie_ops.decode.rms_norm(aie_buffers.decode.x, aie_buffers.W_final_norm, aie_buffers.decode.x) # Step 4: Final normalization
    aie_ops.decode.gemv_out_head(aie_buffers.W_out_head, aie_buffers.decode.x, aie_buffers.decode.logits)  # Step 5: Output projection

    # Read outputs from NPU to CPU
    aie_buffers.decode.logits.to("cpu")
    logits = aie_buffers.decode.logits.view_as_torch().view(1, 1, config.vocab_size)

    return logits, state


def transformer_blocks_forward_decode(config, num_preceding_tokens):
    fused_op = aie_ops.decode.attn_fused
    #fused_op.input_buffer.view_as_torch().to("cpu")[:] = 0
    #fused_op.output_buffer.view_as_torch().to("cpu")[:] = 0
    #fused_op.scratch_buffer.view_as_torch().to("cpu")[:] = 0

    for layer_idx in range(config.n_layers):
        #fused_op.get_buffer(f"W_norm1_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_norm1[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_attn_query_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_attn_query_decode[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_attn_key_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_attn_key_decode[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_attn_value_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_attn_value_decode[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_attn_output_decode_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_attn_output_decode[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_norm2_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_norm2[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_ffn_gate_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_ffn_gate_decode[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_ffn_up_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_ffn_up_decode[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"W_ffn_down_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.W_ffn_down_decode[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"keys_cache_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.keys_cache[layer_idx].to("cpu").view_as_torch().flatten()
        #fused_op.get_buffer(f"values_cache_{layer_idx}").to("cpu").view_as_torch()[:] = aie_buffers.values_cache[layer_idx].to("cpu").view_as_torch().flatten()
        pass

    fused_op.get_buffer("x").to("cpu").view_as_torch()[:] = aie_buffers.decode.x.to("cpu").view_as_torch().flatten()
    #fused_op.get_buffer("rope_angles").to("cpu").view_as_torch()[:] = aie_buffers.decode.rope_angles.to("cpu").view_as_torch().flatten()
    #fused_op.get_buffer("attn_scale_factor").to("cpu").view_as_torch()[:] = aie_buffers.decode.attn_scale_factor.to("cpu").view_as_torch().flatten()

    fused_op()
    
    for layer_idx in range(config.n_layers):
        #aie_buffers.keys_cache[layer_idx].to("cpu").view_as_torch().flatten()[:] = fused_op.get_buffer(f"keys_cache_{layer_idx}").to("cpu").view_as_torch().flatten()
        #aie_buffers.values_cache[layer_idx].to("cpu").view_as_torch().flatten()[:] = fused_op.get_buffer(f"values_cache_{layer_idx}").to("cpu").view_as_torch().flatten()
        pass
    aie_buffers.decode.x.to("cpu").view_as_torch()[:] = fused_op.get_buffer("x").to("cpu").view_as_torch()[:]


# Main
# ##########################################################################

def llama_forward_pass(
    config,
    state
):
    batch, seq_len = state.token_ids.shape
    if seq_len > 1:
        ret = llama_forward_pass_prefill(config, state)
        state.num_preceding_tokens = state.token_ids.shape[1]
        return ret
    else:
        ret = llama_forward_pass_decode(config, state)
        state.num_preceding_tokens += 1
        return ret


def main():
    global aie_ops, aie_buffers
    max_seq_len = 2048
    prompt = "The capital of France is "
    #with open('prompt.txt', 'r') as f:
    #    prompt = f.read()
    #prompt = prompt[:max_seq_len]

    config, state = harness.init(prompt=prompt)

    aie_ops = AIELlamaOperators(config, max_seq_len)
    aie_buffers = AIELlamaBuffers(config, max_seq_len)

    print(prompt, end='', flush=True)
    harness.generate(config, state, llama_forward_pass, use_kv_cache=True)

if __name__ == "__main__":
    main()
