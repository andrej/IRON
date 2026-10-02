# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants of mlir-aie's flm_gemma4 kernels that the host references use."""

# The epsilon of aie_kernels/flm_gemma4/rms_norm.h.
RMS_EPS = 1e-6

# The block of a bf16 projection, from aie_kernels/flm_gemma4/decode_bf16_proj.h:
# BF16_M out-features by BF16_K in-features.
BF16_M, BF16_K = 32, 256
