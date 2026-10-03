# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The q4nx weight format that FastFlowLM quantizes Gemma 4 into.

A block holds M_TILE out-features by K_TILE in-features in BLOCK_BYTES bytes:
bf16 scales, then bf16 minima, both at ``[k // GROUP, m]``, then 4-bit codes
at nibble ``(m // 16) * 16 * K_TILE + k * 16 + m % 16``, low nibble first. A
weight is ``min + scale * code``.

The functions here take and return blocks along the last axes. Each operator
orders the blocks of a matrix as its cores consume them.
"""

import numpy as np

from iron.operators.flm.aie2p_math_emulation import bf16_to_f32, to_bf16

# Replaced by aie.iron.kernels.quant.Q4NX_M_TILE, Q4NX_K_TILE, Q4NX_GROUP and
# Q4NX_BLOCK_BYTES at the next wheel bump.
M_TILE, K_TILE, GROUP = 32, 256, 32

BITS_PER_WEIGHT = 4 + 2 * 16 // GROUP
BLOCK_BYTES = M_TILE * K_TILE * BITS_PER_WEIGHT // 8

# The bytes of a block's scales and minima.
_PARAM_BYTES = 2 * 2 * M_TILE * K_TILE // GROUP
# The out-features of one run of codes.
_RUN = 16


def packed_bytes(n_weights: int) -> int:
    """Bytes holding n_weights in q4nx packing."""
    return n_weights * BITS_PER_WEIGHT // 8


# Replaced by aie.iron.kernels.quant.q4nx_unpack at the next wheel bump.
def unpack(blocks):
    """Blocks ``(..., BLOCK_BYTES)`` to ``(codes, scales, mins)``.

    codes is uint8 ``(..., M_TILE, K_TILE)``. scales and mins are float32
    ``(..., M_TILE, K_TILE // GROUP)``.
    """
    blocks = np.ascontiguousarray(blocks, np.uint8)
    lead = blocks.shape[:-1]
    params = bf16_to_f32(np.ascontiguousarray(blocks[..., :_PARAM_BYTES]).view("<u2"))
    params = params.reshape(lead + (2, K_TILE // GROUP, M_TILE)).swapaxes(-1, -2)
    packed = blocks[..., _PARAM_BYTES:]
    codes = np.empty(lead + (M_TILE * K_TILE,), np.uint8)
    codes[..., 0::2] = packed & 0xF
    codes[..., 1::2] = packed >> 4
    codes = codes.reshape(lead + (M_TILE // _RUN, K_TILE, _RUN)).swapaxes(-1, -2)
    codes = codes.reshape(lead + (M_TILE, K_TILE))
    return codes, params[..., 0, :, :], params[..., 1, :, :]


def pack(codes, scales, mins):
    """The blocks ``(..., BLOCK_BYTES)`` that unpack splits into codes, scales
    and mins. scales and mins must be bf16 values."""
    codes = np.asarray(codes, np.uint8)
    lead = codes.shape[:-2]
    params = np.stack([scales, mins], -3).swapaxes(-1, -2)
    bits = to_bf16(params, "rne")
    if not np.array_equal(bf16_to_f32(bits), params):
        raise ValueError("q4nx scales and minima must be bf16 values")
    c = codes.reshape(lead + (M_TILE // _RUN, _RUN, K_TILE)).swapaxes(-1, -2)
    c = c.reshape(lead + (M_TILE * K_TILE,))
    packed = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)
    params = bits.reshape(lead + (_PARAM_BYTES // 2,)).view(np.uint8)
    return np.concatenate([params, packed], -1)


def dequantize(blocks, dtype=np.float32):
    """Blocks ``(..., BLOCK_BYTES)`` to their weights ``(..., M_TILE, K_TILE)``,
    computed in dtype."""
    codes, scales, mins = unpack(blocks)
    group = np.arange(K_TILE) // GROUP
    return mins[..., group].astype(dtype) + scales[..., group].astype(dtype) * codes


def random_blocks(rng, n_blocks, scales, mins, rounding):
    """n_blocks random blocks ``(n_blocks, BLOCK_BYTES)``.

    scales and mins are the ``(low, high)`` ranges of uniform draws. rounding
    is "floor" or "rne", the rounding of the draws to bf16. The codes are
    uniform.
    """
    shape = (n_blocks, M_TILE * K_TILE // GROUP)
    params = np.concatenate([rng.uniform(*scales, shape), rng.uniform(*mins, shape)], 1)
    codes = rng.integers(0, 256, (n_blocks, M_TILE * K_TILE // 2), dtype=np.uint8)
    return np.concatenate([to_bf16(params, rounding).view(np.uint8), codes], 1)
