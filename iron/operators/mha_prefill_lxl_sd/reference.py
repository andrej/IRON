# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16


def apply_rope(x, lut):
    """Apply Rotary Position Embedding using pre-computed cos/sin LUT.

    x: (rows, cols) — rows are (positions * heads) interleaved
    lut: (angle_rows, cols) — interleaved [cos_0, sin_0, cos_1, sin_1, ...]
         If angle_rows < rows, each angle row is reused for
         (rows // angle_rows) consecutive input rows (block repetition).
    Returns: (rows, cols) with RoPE applied (two-halves method)
    """
    rows, cols = x.shape
    angle_rows = lut.shape[0]
    half = cols // 2

    cos = lut[:, ::2]   # (angle_rows, half)
    sin = lut[:, 1::2]  # (angle_rows, half)

    if angle_rows < rows:
        # Block repetition: each angle row repeats for consecutive input rows
        repeats = rows // angle_rows
        cos = cos.repeat_interleave(repeats, dim=0)  # (rows, half)
        sin = sin.repeat_interleave(repeats, dim=0)  # (rows, half)

    x1 = x[:, :half]
    x2 = x[:, half:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out


def generate_golden_reference(
    num_heads,
    num_kv_groups,
    head_dim,
    embedding_dim,
    seq_len,
    seed=42,
):
    """Generate golden reference for fused attention prefill.

    Parameters:
        num_heads (H): number of query attention heads
        num_kv_groups (G): number of KV heads (G=H for MHA, G<H for GQA)
        head_dim (d): dimension per head
        embedding_dim (E): model embedding dimension
        seq_len (S): sequence length (= max_seq_len for prefill)
        seed: random seed

    Returns:
        dict with all intermediate and final tensors
    """
    torch.manual_seed(seed)
    H, G, d, E, S = num_heads, num_kv_groups, head_dim, embedding_dim, seq_len
    group_size = H // G

    val_range = 0.5

    # Input: (S, E)
    x = torch.randn(S, E, dtype=torch.float32).to(torch.bfloat16) * val_range

    # RoPE angles for all S positions: shape (S, d)
    # LUT format: interleaved [cos_0, sin_0, cos_1, sin_1, ...]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    rope_angles = torch.zeros(S, d, dtype=torch.float32)
    for pos in range(S):
        angles_half = freqs * pos
        rope_angles[pos, ::2] = torch.cos(angles_half)
        rope_angles[pos, 1::2] = torch.sin(angles_half)
    rope_angles = rope_angles.to(torch.bfloat16)

    # Weight matrices (transposed for GEMM: input @ W → output)
    # Q proj: (S, E) @ (E, H*d) → (S, H*d)
    W_query = torch.randn(E, H * d, dtype=torch.bfloat16) * val_range
    # K proj: (S, E) @ (E, G*d) → (S, G*d)
    W_key = torch.randn(E, G * d, dtype=torch.bfloat16) * val_range
    # V proj: (S, E) @ (E, G*d) → (S, G*d)
    W_value = torch.randn(E, G * d, dtype=torch.bfloat16) * val_range
    # Output proj: (S, H*d) @ (H*d, E) → (S, E)
    W_output = torch.randn(H * d, E, dtype=torch.bfloat16) * val_range

    # Scale factor: 1/sqrt(d), broadcast to (H*S, S)
    scale = 1.0 / (d ** 0.5)
    attn_scale_factor = torch.full((H * S * S,), scale, dtype=torch.bfloat16)

    # Causal mask: (H*S, S) — 0 for valid positions, -inf for future positions
    # Row (h*S + i) attends to positions 0..i, so mask col j > i with -inf
    causal_mask = torch.zeros(H * S, S, dtype=torch.bfloat16)
    for h in range(H):
        for i in range(S):
            for j in range(S):
                if j > i:
                    causal_mask[h * S + i, j] = torch.tensor(float("-inf")).to(
                        torch.bfloat16
                    )

    # ---- Step 1-3: Q/K/V projections ----
    queries_raw = x.float() @ W_query.float()  # (S, H*d)
    queries_raw = queries_raw.to(torch.bfloat16)
    keys_raw = x.float() @ W_key.float()  # (S, G*d)
    keys_raw = keys_raw.to(torch.bfloat16)
    values_raw = x.float() @ W_value.float()  # (S, G*d)
    values_raw = values_raw.to(torch.bfloat16)

    # ---- Step 4-5: RoPE ----
    # Q proj output is (S, H*d), viewed as (S*H, d) with heads interleaved:
    # row layout: [pos0_head0, pos0_head1, ..., pos0_headH-1, pos1_head0, ...]
    # RoPE angle_rows=S: row i uses angle row (i % S) = position index
    queries_for_rope = queries_raw.reshape(S * H, d)
    queries_roped = apply_rope(queries_for_rope, rope_angles)  # (S*H, d)

    keys_for_rope = keys_raw.reshape(S * G, d)
    keys_roped = apply_rope(keys_for_rope, rope_angles)  # (S*G, d)

    # ---- Step 6: Deinterleave Q: (S*H, d) → (H, S, d) ----
    # Current layout: [pos0_h0, pos0_h1, ..., pos0_{H-1}, pos1_h0, ...]
    # = (S, H, d) in memory; reshape and transpose to (H, S, d)
    queries_deinterleaved = queries_roped.reshape(S, H, d).transpose(0, 1).contiguous()  # (H, S, d)

    # ---- Step 7: Deinterleave K: (S*G, d) → (G, S, d) then transpose to (G, d, S) ----
    keys_deinterleaved = keys_roped.reshape(S, G, d).transpose(0, 1).contiguous()  # (G, S, d)
    keys_transposed = keys_deinterleaved.transpose(1, 2).contiguous()  # (G, d, S)

    # ---- Step 8: Deinterleave V: (S, G*d) → (G, S, d) ----
    values_deinterleaved = values_raw.reshape(S, G, d).transpose(0, 1).contiguous()  # (G, S, d)

    # ---- Step 9: GQA repeat ----
    if group_size > 1:
        # Repeat keys and values: (G, ...) → (H, ...)
        # Flatten to (G, d*S) / (G, S*d), repeat, reshape
        keys_for_scores = keys_transposed.reshape(G, d * S).repeat_interleave(
            group_size, dim=0
        ).reshape(H, d, S)
        values_for_context = values_deinterleaved.reshape(G, S * d).repeat_interleave(
            group_size, dim=0
        ).reshape(H, S, d)
    else:
        keys_for_scores = keys_transposed  # (H, d, S)
        values_for_context = values_deinterleaved  # (H, S, d)

    # ---- Step 10: Score GEMM per head ----
    # Q_head(S, d) @ K_head(d, S) → scores(S, S)
    attn_scores = torch.zeros(H, S, S, dtype=torch.bfloat16)
    for h in range(H):
        attn_scores[h] = (
            queries_deinterleaved[h].float() @ keys_for_scores[h].float()
        ).to(torch.bfloat16)

    # ---- Step 11: Scale ----
    attn_scores_scaled = (attn_scores.float() * scale).to(torch.bfloat16)
    # ---- Step 12: Causal mask (add -inf) ----
    attn_scores_masked = attn_scores_scaled.reshape(H * S, S).float() + causal_mask.float()
    attn_scores_masked = attn_scores_masked.to(torch.bfloat16)

    # ---- Step 13: Softmax ----
    attn_weights = torch.nn.functional.softmax(
        attn_scores_masked.float().reshape(H, S, S), dim=-1
    ).to(torch.bfloat16)  # (H, S, S)

    # ---- Step 14: Context GEMM per head ----
    # weights(S, S) @ values(S, d) → context(S, d)
    attn_context = torch.zeros(H, S, d, dtype=torch.bfloat16)
    for h in range(H):
        attn_context[h] = (
            attn_weights[h].float() @ values_for_context[h].float()
        ).to(torch.bfloat16)

    # ---- Step 15: Re-interleave context: (H, S, d) → (S, H*d) ----
    context_interleaved = attn_context.transpose(0, 1).contiguous().reshape(S, H * d)

    # ---- Step 16: Output projection ----
    attn_output = (context_interleaved.float() @ W_output.float()).to(torch.bfloat16)

    return {
        "input": x,
        "rope_angles": rope_angles,
        "W_query": W_query,
        "W_key": W_key,
        "W_value": W_value,
        "W_output": W_output,
        "attn_scale_factor": attn_scale_factor,
        "causal_mask": causal_mask,
        "queries_raw": queries_raw,
        "keys_raw": keys_raw,
        "values_raw": values_raw,
        "queries_roped": queries_roped,
        "keys_roped": keys_roped,
        "queries_deinterleaved": queries_deinterleaved,
        "keys_deinterleaved": keys_deinterleaved,
        "keys_transposed": keys_transposed,
        "values_deinterleaved": values_deinterleaved,
        "keys_for_scores": keys_for_scores,
        "values_for_context": values_for_context,
        "attn_scores": attn_scores,
        "attn_scores_scaled": attn_scores_scaled,
        "attn_scores_masked": attn_scores_masked,
        "attn_weights": attn_weights,
        "attn_context": attn_context,
        "context_interleaved": context_interleaved,
        "attn_output": attn_output,
    }
