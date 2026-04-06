# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16


def apply_rope(x, lut):
    """Apply Rotary Position Embedding using pre-computed cos/sin LUT.

    x: (rows, cols) — rows are heads or positions
    lut: (angle_rows, cols) — interleaved [cos_0, sin_0, cos_1, sin_1, ...] values
         LUT format matches the NPU rope kernel: even indices = cos, odd indices = sin
         If angle_rows=1, the same angles are broadcast to all rows.
    Returns: (rows, cols) with RoPE applied (two-halves method)
    """
    rows, cols = x.shape
    angle_rows = lut.shape[0]
    half = cols // 2

    # Extract cos and sin from interleaved LUT (even=cos, odd=sin)
    if angle_rows == 1:
        cos = lut[:, ::2].expand(rows, -1)   # (rows, half)
        sin = lut[:, 1::2].expand(rows, -1)  # (rows, half)
    else:
        cos = lut[:, ::2]   # (angle_rows, half)
        sin = lut[:, 1::2]  # (angle_rows, half)

    x1 = x[:, :half]   # (rows, half)
    x2 = x[:, half:]   # (rows, half)

    # Two-halves RoPE: matches rope_kernel_two_halves
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out


def generate_golden_reference(
    num_heads,
    num_kv_groups,
    head_dim,
    embedding_dim,
    max_seq_len,
    token_position,
    seed=42,
):
    """Generate golden reference for attention decode with KV cache.

    Parameters:
        num_heads (H): number of query attention heads
        num_kv_groups (G): number of KV heads (for GQA; if G=H it's standard MHA)
        head_dim (d): dimension per head
        embedding_dim (E): model embedding dimension (typically H*d)
        max_seq_len (S): maximum sequence length (KV cache size)
        token_position (t): position of the current token (0-indexed); the KV cache
            will be pre-filled with random data for positions [0..t-1], and position t
            will be computed by this function.
        seed: random seed

    Returns:
        dict with keys: input, rope_angles, W_query, W_key, W_value, W_output,
            attn_scale_factor, keys_cache_init, values_cache_init, queries, keys, values,
            queries_rope, keys_rope, keys_cache, values_cache, attn_scores_keys,
            attn_scores_values, attn_scores, attn_weights, attn_scores_values_T,
            attn_context, attn_output
    """
    torch.manual_seed(seed)
    H, G, d, E, S = num_heads, num_kv_groups, head_dim, embedding_dim, max_seq_len
    t = token_position
    group_size = H // G  # number of query heads per KV head

    val_range = 0.5

    # Inputs
    x = torch.randn(E, dtype=torch.float32).to(torch.bfloat16) * val_range

    # RoPE angles for position t: shape (1, d)
    # LUT format: interleaved [cos_0, sin_0, cos_1, sin_1, ...] matching the NPU kernel.
    # The kernel (rope_kernel_two_halves) reads even indices as cos and odd indices as sin.
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    angles_half = freqs * t  # (d//2,) — raw angle values for each frequency
    cos_vals = torch.cos(angles_half)  # (d//2,)
    sin_vals = torch.sin(angles_half)  # (d//2,)
    rope_angles = torch.zeros(1, d, dtype=torch.float32)
    rope_angles[0, ::2] = cos_vals   # even positions = cos
    rope_angles[0, 1::2] = sin_vals  # odd positions = sin
    rope_angles = rope_angles.to(torch.bfloat16)

    # Weight matrices (stored as (output_rows, input_cols) for GEMV)
    W_query = torch.randn(H * d, E, dtype=torch.bfloat16) * val_range
    W_key = torch.randn(G * d, E, dtype=torch.bfloat16) * val_range
    W_value = torch.randn(G * d, E, dtype=torch.bfloat16) * val_range
    W_output = torch.randn(E, H * d, dtype=torch.bfloat16) * val_range

    # Attention scale factor: 1/sqrt(d), broadcast to shape (H*S,)
    scale = 1.0 / (d ** 0.5)
    attn_scale_factor = torch.full((H * S,), scale, dtype=torch.bfloat16)

    # Pre-fill KV cache for positions [0..t-1] with random data
    # Keys cache: (G, S, d) — G KV groups, S positions, d head_dim
    keys_cache_init = torch.randn(G, S, d, dtype=torch.bfloat16) * val_range
    values_cache_init = torch.randn(G, S, d, dtype=torch.bfloat16) * val_range
    # Positions >= t are garbage (not yet filled), so zero them out for clarity
    keys_cache_init[:, t:, :] = 0.0
    values_cache_init[:, t:, :] = 0.0

    # ---- Step 1-3: Q/K/V projections ----
    queries = W_query @ x  # (H*d,)
    keys = W_key @ x  # (G*d,)
    values = W_value @ x  # (G*d,)

    # ---- Step 4-5: RoPE ----
    queries_2d = queries.reshape(H, d)  # (H, d)
    keys_2d = keys.reshape(G, d)  # (G, d)
    queries_rope_2d = apply_rope(queries_2d, rope_angles)  # (H, d)
    keys_rope_2d = apply_rope(keys_2d, rope_angles)  # (G, d)
    queries_rope = queries_rope_2d.flatten()  # (H*d,)
    keys_rope = keys_rope_2d.flatten()  # (G*d,)

    # ---- Step 6-7: KV cache update ----
    keys_cache = keys_cache_init.clone()
    values_cache = values_cache_init.clone()
    keys_cache[:, t, :] = keys_rope_2d  # insert at position t
    values_cache[:, t, :] = values.reshape(G, d)  # insert at position t

    context_len = t + 1  # number of valid positions in cache

    # ---- Step 8-9: GQA broadcast (repeat interleave along head dim) ----
    # keys_cache: (G, S, d) → (H, S, d) by repeating each KV group group_size times
    # In llama: repeat_interleave_op repeats rows (the G*S*d flat buffer) group_size times
    # Flatten cache to (G, S*d) for repeat, then to (H, S*d)
    keys_cache_flat = keys_cache.reshape(G, S * d)  # (G, S*d)
    values_cache_flat = values_cache.reshape(G, S * d)  # (G, S*d)
    # Repeat interleave along dim 0: each row repeated group_size times
    attn_scores_keys_flat = keys_cache_flat.repeat_interleave(group_size, dim=0)  # (H, S*d)
    attn_scores_values_flat = values_cache_flat.repeat_interleave(group_size, dim=0)  # (H, S*d)
    # Reshape to (H, S, d)
    attn_scores_keys = attn_scores_keys_flat.reshape(H, S, d)
    attn_scores_values = attn_scores_values_flat.reshape(H, S, d)

    # ---- Step 10: Score computation ----
    # GEMV: attn_scores_keys(H, S, d) @ queries(H, d) → attn_scores(H, S)
    # Per head: scores[h] = attn_scores_keys[h] @ queries_rope_2d[h]
    queries_per_head = queries_rope_2d  # (H, d)
    # attn_scores_keys[h] is (S, d) — we compute (S, d) @ (d,) = (S,)
    attn_scores = torch.einsum("hsd,hd->hs", attn_scores_keys, queries_per_head)  # (H, S)

    # ---- Step 11: Scale ----
    attn_scores_scaled = attn_scores * scale  # (H, S)

    # ---- Step 12: Softmax (only over valid positions) ----
    # Mask out positions beyond context_len
    attn_scores_masked = attn_scores_scaled.clone()
    attn_scores_masked[:, context_len:] = float("-inf")
    attn_weights = torch.nn.functional.softmax(attn_scores_masked.float(), dim=-1).to(
        torch.bfloat16
    )  # (H, S)

    # ---- Step 13: Transpose values per head ----
    # attn_scores_values is (H, S, d); we need (H, d, S) per head for context GEMV
    # Per head: transpose (S, d) → (d, S)
    attn_scores_values_T = attn_scores_values.transpose(1, 2)  # (H, d, S)

    # ---- Step 14: Context computation ----
    # GEMV: attn_scores_values_T(H, d, S) @ attn_weights(H, S) → attn_context(H, d)
    attn_context = torch.einsum("hds,hs->hd", attn_scores_values_T, attn_weights)  # (H, d)

    # ---- Step 15: Output projection ----
    attn_context_flat = attn_context.flatten()  # (H*d,)
    attn_output = W_output @ attn_context_flat  # (E,)

    return {
        "input": x,
        "rope_angles": rope_angles,
        "W_query": W_query,
        "W_key": W_key,
        "W_value": W_value,
        "W_output": W_output,
        "attn_scale_factor": attn_scale_factor,
        "keys_cache_init": keys_cache_init,
        "values_cache_init": values_cache_init,
        "queries": queries,
        "keys": keys,
        "values": values,
        "queries_rope": queries_rope,
        "keys_rope": keys_rope,
        "keys_cache": keys_cache,
        "values_cache": values_cache,
        "attn_scores_keys": attn_scores_keys,
        "attn_scores_values": attn_scores_values,
        "attn_scores": attn_scores,
        "attn_scores_scaled": attn_scores_scaled,
        "attn_weights": attn_weights,
        "attn_scores_values_T": attn_scores_values_T,
        "attn_context": attn_context,
        "attn_output": attn_output,
        "context_len": context_len,
    }
