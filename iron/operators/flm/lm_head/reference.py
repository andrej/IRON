# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for :class:`iron.operators.flm.LMHead`, in float64."""

import numpy as np

from iron.operators.flm import q4nx
from iron.operators.flm.q4nx import BLOCK_BYTES, K_TILE, M_TILE


def dequantize(w, dim, vocab, cols, rows):
    """The whole q4nx vocabulary buffer to its (vocab, dim) weights.

    Out-feature ``n`` is ``((round * cols + col) * rows + row) * M_TILE + m``.
    Its k-th block has block index
    ``((round * cols + col) * k_blocks + k) * rows + row``.
    """
    blocks = np.asarray(w).view(np.uint8).reshape(-1, dim // K_TILE, rows, BLOCK_BYTES)
    out = np.empty((vocab, dim), np.float64)
    span = rows * M_TILE
    # One (round, col) slice at a time bounds the float64 temporaries.
    for i, blocks_i in enumerate(blocks):
        weights = q4nx.dequantize(blocks_i, np.float64).transpose(1, 2, 0, 3)
        out[i * span : (i + 1) * span] = weights.reshape(span, dim)
    return out


def reference(weights, x, softcap, eps=1e-6):
    """Softcapped logits of the RMS-normalized token.

    ``weights`` is the (vocab, dim) matrix from ``dequantize``. ``x`` has the
    layout of LMHead's X.
    """
    x = np.asarray(x, np.float64)
    dim = weights.shape[1]
    token, rms_w = x[:dim], x[dim : 2 * dim]
    normed = token / np.sqrt(np.mean(token**2) + eps) * rms_w
    return softcap * np.tanh(weights @ normed / softcap)
