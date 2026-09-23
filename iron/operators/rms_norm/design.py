# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    kernels,
)
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


@iron.jit
def my_rms_norm(
    *,
    num_elements: CompileTime[int],
    num_columns: CompileTime[int],
    num_channels: CompileTime[int],
    tile_size: CompileTime[int],
    trace_size: CompileTime[int],
    epsilon: CompileTime[float] = 1e-5,
):
    per_tile_elements = 8192 if tile_size > 8192 else tile_size
    total_cores = num_columns * num_channels
    per_core_elements = num_elements // total_cores
    if num_elements % total_cores != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {total_cores}."
        )
    N_div_n = per_core_elements // per_tile_elements
    chunk = num_elements // num_columns // num_channels  # For offset calculation
    dtype = bfloat16

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    fifodepth = 1 if tile_size > 4096 else 2

    # AIE-array data movement with object fifos
    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # AIE Core Function declaration
    rms_norm_kernel = kernels.rms_norm_eps(tile_size=per_tile_elements)

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_out, rms_norm_kernel):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            rms_norm_kernel(elem_in1, elem_out, per_tile_elements, epsilon)
            of_in1.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                rms_norm_kernel,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels.
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    def sequence(A, C, of_in1s_prods, of_outs_conss):

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = TaskGroup()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                of_in1s_prods[i * num_channels + j].fill(
                    A,
                    taps[i * num_channels + j],
                    group=tg,
                )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                of_outs_conss[i * num_channels + j].drain(
                    C,
                    taps[i * num_channels + j],
                    wait=True,  # wait for the transfer to complete and data to be available
                    group=tg,
                )
        tg.finish()

    rt = Runtime(
        sequence,
        [
            tensor_ty,
            tensor_ty,
            [of.prod() for of in of_in1s],
            [of.cons() for of in of_outs],
        ],
    )
    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt, workers=my_workers).resolve_program()
