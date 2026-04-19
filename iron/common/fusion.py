# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ml_dtypes
import ctypes
from pathlib import Path
from . import compilation as comp
from .base import AIEOperatorBase, MLIROperator
from .utils import XRTSubBuffer
import aie.utils as aie_utils
from aie.utils.npukernel import NPUKernel
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from .xclbin_patching import (
    parse_instr_offsets_json,
    parse_write32_offsets_json,
    patch_load_pdi,
    patch_rtp,
    find_magic_value_offsets,
    patch_magic_value,
)

# Fused Operator
# ##########################################################################


class FusedMLIROperator(AIEOperatorBase):
    """Operator that fuses multiple MLIROperators into one."""

    def __init__(
        self, name, runlist, input_args, output_args, buffer_sizes=None, *args, **kwargs
    ):
        if not all(
            isinstance(op, MLIROperator) and all(isinstance(buf, str) for buf in bufs)
            for op, *bufs in runlist
        ):
            raise TypeError(
                "runlist entries must be (MLIROperator, *str) tuples; "
                "each operator must be an MLIROperator and each buffer name must be a str"
            )
        super().__init__(*args, **kwargs)
        self.runlist = runlist
        self.name = name
        self.input_args = input_args
        self.output_args = output_args
        self.explicit_buffer_sizes = (
            buffer_sizes or {}
        )  # Optional dict: buffer_name -> size_in_bytes

    def get_kernel_artifacts(self):
        """Collect all kernel artifacts from child operators.

        Returns:
            List of KernelObjectArtifact instances from all unique child operators,
            with filenames and symbol prefixes disambiguated per operator index.
        """
        kernel_artifacts = []
        seen: dict[int, object] = {}
        unique_operators = [
            seen.setdefault(id(op), op) for op, *_ in self.runlist if id(op) not in seen
        ]
        for idx, op in enumerate(unique_operators):
            objs = op.get_kernel_artifacts()
            for obj in objs:
                obj.filename = f"op{idx}_{obj.filename}"
                obj.prefix_symbols = f"op{idx}_"
            kernel_artifacts.extend(objs)
        return kernel_artifacts

    def get_mlir_artifact(self):
        """Build and return the fused MLIR source artifact.

        Constructs the operator MLIR map and run-list, then wraps them in a
        ``FusedMLIRSource`` artifact.  Buffer layout attributes
        (``subbuffer_layout``, ``buffer_sizes``, ``slice_info``) must already
        be set by ``set_up_artifacts()`` before this method is called.

        Returns:
            A ``FusedMLIRSource`` artifact ready for compilation.
        """
        # Build operator_mlir_map: {op_name -> PythonGeneratedMLIRArtifact}
        operator_mlir_map = {}
        comp_runlist = []
        op_names = {}  # id(op) -> op_name

        seen2: dict[int, object] = {}
        unique_operators = [
            seen2.setdefault(id(op), op)
            for op, *_ in self.runlist
            if id(op) not in seen2
        ]
        for idx, op in enumerate(unique_operators):
            mlir_artifact = op.get_mlir_artifact()
            if len(op.get_kernel_artifacts()) > 0:
                mlir_artifact.generator.kwargs["func_prefix"] = f"op{idx}_"
            op_name = f"op{idx}_{op.__class__.__name__}"
            op_names[id(op)] = op_name
            operator_mlir_map[op_name] = mlir_artifact

        for op, *bufs in self.runlist:
            comp_runlist.append((op_names[id(op)], *bufs))

        filename = self.name + "_fused.mlir"
        fused_artifact = comp.FusedMLIRSource(
            filename,
            operator_mlir_map=operator_mlir_map,
            runlist=comp_runlist,
            subbuffer_layout=self.subbuffer_layout,
            buffer_sizes=self.buffer_sizes,
            slice_info=self.slice_info,
        )

        return fused_artifact

    def _calculate_buffer_layout(self):
        args = {}  # base_buffer_name -> args_spec
        sliced_buffers = (
            {}
        )  # full_buffer_name (with slice) -> (base_name, start, end, args_spec)

        # Collect all buffer specs from operators
        for op, *bufs in self.runlist:
            args_specs = op.get_arg_spec()
            if len(args_specs) != len(bufs):
                raise ValueError(
                    f"Number of buffers ({len(bufs)}) must match operator argument "
                    f"specification ({len(args_specs)}) for operator {op!r}"
                )
            for i, buf_name in enumerate(bufs):
                args_spec = args_specs[i]

                # Parse slice notation: "buffer_name[start:end]"
                if "[" in buf_name and buf_name.endswith("]"):
                    base_name = buf_name[: buf_name.index("[")]
                    slice_part = buf_name[buf_name.index("[") + 1 : -1]
                    start, end = map(int, slice_part.split(":"))
                    sliced_buffers[buf_name] = (base_name, start, end, args_spec)
                    # Track that base buffer exists (size will be set later)
                    if (
                        base_name not in args
                        and base_name not in self.explicit_buffer_sizes
                    ):
                        raise ValueError(
                            f"Sliced buffer '{buf_name}' requires explicit size for base buffer '{base_name}' in buffer_sizes parameter"
                        )
                else:
                    # Regular buffer (no slice)
                    if buf_name not in args:
                        args[buf_name] = args_spec
                    else:
                        if np.prod(args[buf_name].shape) != np.prod(args_spec.shape):
                            raise ValueError(
                                f"Buffer '{buf_name}' has conflicting sizes between operators: "
                                f"{args[buf_name].shape} vs {args_spec.shape}"
                            )

        # Verify all input/output args are present (either as regular or sliced buffers)
        all_buffer_names = set(args.keys()) | set(sliced_buffers.keys())
        for arg in self.input_args:
            # Check if it's a base buffer name in explicit_buffer_sizes
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Input argument {arg} not found in runlist buffers")
        for arg in self.output_args:
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Output argument {arg} not found in runlist buffers")

        # Determine buffer types and create layout
        subbuffer_layout = {}
        slice_info = {}  # full_buffer_name -> (base_name, start, end)

        def add_buffers(buffer_type, args_list):
            offset = 0
            for arg in args_list:
                if arg in self.explicit_buffer_sizes:
                    # Explicit size specified - this is a parent buffer for slices
                    length = self.explicit_buffer_sizes[arg]
                    subbuffer_layout[arg] = (buffer_type, offset, length)
                    offset += length
                elif arg in args:
                    # Regular buffer with inferred size
                    arg_spec = args[arg]
                    length = int(
                        np.prod(arg_spec.shape) * np.dtype(arg_spec.dtype).itemsize
                    )
                    subbuffer_layout[arg] = (buffer_type, offset, length)
                    offset += length
                # Note: sliced buffers are handled separately, not in args_list
            return offset  # == total length

        # Add sliced buffer entries to layout (they reference parent buffers)
        for buf_name, (base_name, start, end, args_spec) in sliced_buffers.items():
            slice_info[buf_name] = (base_name, start, end)

        input_buffer_size = add_buffers("input", self.input_args)
        output_buffer_size = add_buffers("output", self.output_args)
        scratch_args = [
            arg
            for arg in args
            if arg not in self.input_args and arg not in self.output_args
        ]
        # Also include explicit buffers that are only used for slicing
        for explicit_buf in self.explicit_buffer_sizes:
            if (
                explicit_buf not in self.input_args
                and explicit_buf not in self.output_args
                and explicit_buf not in scratch_args
            ):
                scratch_args.append(explicit_buf)
        scratch_buffer_size = add_buffers("scratch", scratch_args)

        buffer_sizes = (input_buffer_size, output_buffer_size, scratch_buffer_size)
        return subbuffer_layout, buffer_sizes, slice_info

    def set_up_artifacts(self):
        """Set up the artifact dependency graph for this fused operator.

        Computes the buffer layout first, then builds the xclbin and
        instructions artifacts and registers them via ``add_artifacts()``.
        """
        # Calculate buffer layout before building mlir artifact (used by get_mlir_artifact)
        self.subbuffer_layout, self.buffer_sizes, self.slice_info = (
            self._calculate_buffer_layout()
        )
        operator_name = self.name
        mlir_artifact = self.get_mlir_artifact()
        kernel_objects = self.get_kernel_artifacts()

        xclbin_artifact = comp.XclbinArtifact(
            f"{operator_name}.xclbin",
            mlir_input=mlir_artifact,
            dependencies=[mlir_artifact] + kernel_objects,
        )
        insts_artifact = comp.InstsBinArtifact(
            f"{operator_name}.bin",
            mlir_input=mlir_artifact,
            dependencies=[mlir_artifact],
        )
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_elf_artifacts(self):
        """Set up the artifact dependency graph for the full-ELF flow.

        Computes the buffer layout first, then builds the fused MLIR artifact
        and full-ELF artifact and registers them via ``add_artifacts()``.
        """
        self.subbuffer_layout, self.buffer_sizes, self.slice_info = (
            self._calculate_buffer_layout()
        )
        operator_name = self.name
        mlir_artifact = self.get_mlir_artifact()
        kernel_objects = self.get_kernel_artifacts()
        full_elf_artifact = comp.FullElfArtifact(
            f"{operator_name}.elf",
            mlir_input=mlir_artifact,
            dependencies=[mlir_artifact] + kernel_objects,
        )
        self.add_artifacts([full_elf_artifact])

    def compile_elf(self, dry_run=False):
        """Compile using the full-ELF flow instead of the default xclbin flow.

        Returns self for chaining.
        """
        import iron.common.compilation as comp_mod

        self.set_up_elf_artifacts()
        comp_mod.compile(
            self.context.compilation_rules,
            self.artifacts,
            self.context.build_dir,
            dry_run=dry_run,
        )
        return self

    def get_arg_spec(self):
        raise NotImplementedError(
            "FusedMLIROperator does not expose a unified arg spec; "
            "use get_layout_for_buffer() to inspect individual buffer layouts"
        )

    def get_callable(self):
        """Return a callable that executes the fused operator on the NPU.

        Returns:
            A ``FusedXclbinCallable`` wrapping this operator.
        """
        return FusedXclbinCallable(self)

    def get_elf_callable(self):
        """Return a callable that executes the fused operator using the full-ELF flow.

        Returns:
            A ``FusedFullELFCallable`` wrapping this operator.
        """
        return FusedFullELFCallable(self)

    def get_layout_for_buffer(self, buffer_name):
        """Return the (buffer_type, offset, length) layout for a named buffer.

        Sliced buffers are resolved recursively to their parent's absolute
        offset.

        Args:
            buffer_name: Name of the buffer, optionally with slice notation.

        Returns:
            Tuple of (buf_type, offset_bytes, length_bytes).
        """
        if buffer_name in self.slice_info:
            buf_name, start, end = self.slice_info[buffer_name]
            buf_type, parent_start, parent_end = self.get_layout_for_buffer(buf_name)
            return buf_type, parent_start + start, parent_start + end

        buf_type, offset, length = self.subbuffer_layout[buffer_name]
        return buf_type, offset, length


class FusedXclbinCallable:
    """Callable that runs a fused operator using the xclbin flow.

    Loads the xclbin, registers it with XRT, reads the instruction binary,
    parses offsets JSON for PDI and RTP patching, and allocates shared
    input/output/scratch buffers.
    """

    def __init__(self, op):
        self.op = op

        # Locate compiled artifacts
        xclbin_path = op.xclbin_artifact.filename
        insts_path = op.insts_artifact.filename

        # Derive offsets JSON and PDI paths from the insts.bin location
        insts_p = Path(insts_path)
        offsets_json_path = insts_p.parent / (insts_p.stem + "_offsets.json")
        self.offsets_json_path = str(offsets_json_path)

        # Find PDI files in the build directory (generated by --aie-generate-pdi)
        # Sort by operator index (op0, op1, ..., op18) then main.pdi last,
        # matching the compiler's device iteration order for pdi_id assignment.
        build_dir = insts_p.parent
        import re
        def _pdi_sort_key(p):
            m = re.match(r'op(\d+)_', p.name)
            return (0, int(m.group(1))) if m else (1, 0)
        self.pdi_files = sorted(build_dir.glob("*.pdi"), key=_pdi_sort_key)

        # Read base instruction binary as uint32 array
        with open(insts_path, "rb") as f:
            self._base_insts = np.frombuffer(f.read(), dtype=np.uint32).copy()

        # Parse offsets JSON for PDI and write32 (RTP) entries
        self._load_pdi_infos = parse_instr_offsets_json(self.offsets_json_path)
        self._write32_infos = parse_write32_offsets_json(self.offsets_json_path)

        # Load xclbin and register with XRT via DefaultNPURuntime
        npu_kernel = NPUKernel(
            xclbin_path=xclbin_path,
            kernel_name="MLIR_AIE",
            insts_path=insts_path,
        )
        handle = aie_utils.DefaultNPURuntime.load(npu_kernel)
        self._kernel_handle = handle

        # Prepare patched instructions (patch PDI once per PDI file).
        # The compiler assigns 1-based pdi_ids sequentially by device order,
        # matching the sorted PDI file order (op0_*.pdi → pdi_id=1, etc.).
        self._patched_insts = self._base_insts.copy()
        for idx, pdi_file in enumerate(self.pdi_files):
            pdi_id = idx + 1  # compiler uses 1-based IDs
            filtered_infos = [
                info for info in self._load_pdi_infos if info.pdi_id == pdi_id
            ]
            self._patched_insts = patch_load_pdi(
                self._patched_insts, filtered_infos, str(pdi_file)
            )

        # Create the instruction buffer object and fix up LOAD_PDI addresses
        # from relative byte offsets to absolute device addresses.
        # The C++ test_utils does this after creating the bo; the Python
        # xclbin_patching.patch_load_pdi only sets relative offsets.
        self._fixup_load_pdi_addresses()

        # Allocate shared input/output/scratch buffers
        input_buffer_size, output_buffer_size, scratch_buffer_size = op.buffer_sizes
        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize

        self.input_buffer = XRTTensor(
            (max(input_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )
        self.output_buffer = XRTTensor(
            (max(output_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )
        self.scratch_buffer = XRTTensor(
            (max(scratch_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )

        self._buffer_cache = {}

    def get_buffer(self, buffer_name):
        """Return a sub-buffer view for the named buffer."""
        if buffer_name in self._buffer_cache:
            return self._buffer_cache[buffer_name]

        buf_type, offset, length = self.op.get_layout_for_buffer(buffer_name)

        if buf_type == "input":
            main_buffer = self.input_buffer
        elif buf_type == "output":
            main_buffer = self.output_buffer
        elif buf_type == "scratch":
            main_buffer = self.scratch_buffer
        else:
            raise ValueError(
                f"Unknown buffer type '{buf_type}' for buffer '{buffer_name}'"
            )

        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
        sub_buffer = XRTSubBuffer(
            parent_bo=main_buffer.buffer_object(),
            offset_bytes=offset,
            size_bytes=length,
            shape=(length // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )

        self._buffer_cache[buffer_name] = sub_buffer
        return sub_buffer

    def _fixup_load_pdi_addresses(self):
        """Create an XRT buffer object for instructions and convert LOAD_PDI
        address fields from relative byte offsets to absolute device addresses.

        This mirrors what the C++ test_utils code does:
          1. Create a bo from the instruction data
          2. Get the bo's device address
          3. For each LOAD_PDI entry, convert address = bo_addr + relative_offset
          4. Write the fixed-up data back into the bo
        """
        import pyxrt as _pyxrt

        group_id = self._kernel_handle.kernel.group_id(1)
        self._insts_tensor = XRTTensor(
            self._patched_insts,
            flags=_pyxrt.bo.cacheable,
            group_id=group_id,
        )
        bo = self._insts_tensor.buffer_object()
        bo_addr = bo.address()

        # The _insts_tensor._data is the memory-mapped numpy view of the bo.
        # We work on this view so changes go directly into the bo.
        self._insts_bo_data = self._insts_tensor.data

        # Fix up each LOAD_PDI address field from relative to absolute
        # in both the bo data and _patched_insts (the baseline copy)
        for info in self._load_pdi_infos:
            addr_idx = info.address_field_offset_bytes // 4
            rel_offset = int(self._insts_bo_data[addr_idx])
            abs_addr = bo_addr + rel_offset
            lo = np.uint32(abs_addr & 0xFFFFFFFF)
            hi = np.uint32(abs_addr >> 32)
            self._insts_bo_data[addr_idx] = lo
            self._insts_bo_data[addr_idx + 1] = hi
            self._patched_insts[addr_idx] = lo
            self._patched_insts[addr_idx + 1] = hi

        # Sync the fixed-up instructions to the device
        self._insts_tensor.to("npu")

        # Store the bo on the kernel handle so the runtime reuses it
        self._kernel_handle.insts = self._patched_insts
        self._kernel_handle.insts_bo = bo

    def patch_rtp(self, name, value):
        """Patch a named RTP parameter in the instruction stream."""
        patch_rtp(self._patched_insts, self._write32_infos, name, value)

    def patch_magic(self, value, magic=0xDEADBEE0):
        """Replace magic sentinel values in the instruction stream."""
        return patch_magic_value(self._patched_insts, value, magic)

    def reload_insts(self):
        """Reload the patched instruction stream into the kernel handle.

        Copies the patched instruction data into the memory-mapped bo
        and syncs to device. The LOAD_PDI address fields are already
        absolute (set during __init__), and the per-token patches only
        touch non-address fields (StridedCopy offsets, softmax mask
        width), so no re-fixup is needed.
        """
        # Copy patched instruction words into the bo's mapped memory
        np.copyto(self._insts_bo_data, self._patched_insts)
        # Sync to device
        self._insts_tensor.to("npu")

    def __call__(self):
        self.input_buffer.to("npu")

        buffers = [
            self.input_buffer,
            self.output_buffer,
            self.scratch_buffer,
        ]
        aie_utils.DefaultNPURuntime.run(self._kernel_handle, buffers)

        self.output_buffer.to("cpu")


# Full-ELF Flow
# ##########################################################################


def load_elf(op):
    assert isinstance(op.artifacts[0], comp.FullElfArtifact)
    with open(op.artifacts[0].filename, "rb") as f:
        elf_data = np.frombuffer(f.read(), dtype=np.uint32)
    return elf_data


def patch_elf(elf_data, patches):
    for i, patch in patches.items():
        val, mask = patch
        val = np.uint64(val)
        mask = np.uint64(mask)
        elf_data[i] = np.uint32((elf_data[i] & ~mask) | (val & mask))
    return elf_data


class FullELFCallable:
    def __init__(
        self,
        elf_data,
        device_name="main",
        sequence_name="sequence",
    ):
        import pyxrt

        self._pyxrt = pyxrt
        self.device_name = device_name
        self.sequence_name = sequence_name
        self.reload_elf(elf_data)

    def __call__(self, *args):
        run = self._pyxrt.run(self.xrt_kernel)
        for i, arg in enumerate(args):
            assert isinstance(arg, self._pyxrt.bo), f"Argument {i} is not a pyxrt.bo"
            run.set_arg(i, arg)
        run.start()
        ret_code = run.wait()
        if ret_code != self._pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"Kernel execution failed with return code {ret_code}")

    def reload_elf(self, elf_data):
        pyxrt = self._pyxrt
        elf_data_u8 = elf_data.view(dtype=np.uint8)
        ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
        ctypes.pythonapi.PyCapsule_New.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        capsule = ctypes.pythonapi.PyCapsule_New(elf_data_u8.ctypes.data, None, None)
        xrt_elf = pyxrt.elf(capsule, elf_data.nbytes)
        xrt_context = pyxrt.hw_context(aie_utils.DefaultNPURuntime._device, xrt_elf)
        self.xrt_kernel = pyxrt.ext.kernel(
            xrt_context, f"{self.device_name}:{self.sequence_name}"
        )


class FusedFullELFCallable(FullELFCallable):
    def __init__(self, op, elf_data=None):
        if elf_data is None:
            elf_data = load_elf(op)
        super().__init__(elf_data)

        self.op = op
        input_buffer_size, output_buffer_size, scratch_buffer_size = op.buffer_sizes
        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize

        self.input_buffer = XRTTensor(
            (max(input_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )
        self.output_buffer = XRTTensor(
            (max(output_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )
        self.scratch_buffer = XRTTensor(
            (max(scratch_buffer_size, itemsize) // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )

        self._buffer_cache = {}

    def get_buffer(self, buffer_name):
        if buffer_name in self._buffer_cache:
            return self._buffer_cache[buffer_name]

        buf_type, offset, length = self.op.get_layout_for_buffer(buffer_name)

        if buf_type == "input":
            main_buffer = self.input_buffer
        elif buf_type == "output":
            main_buffer = self.output_buffer
        elif buf_type == "scratch":
            main_buffer = self.scratch_buffer
        else:
            raise ValueError(
                f"Unknown buffer type '{buf_type}' for buffer '{buffer_name}'"
            )

        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
        sub_buffer = XRTSubBuffer(
            parent_bo=main_buffer.buffer_object(),
            offset_bytes=offset,
            size_bytes=length,
            shape=(length // itemsize,),
            dtype=ml_dtypes.bfloat16,
        )

        self._buffer_cache[buffer_name] = sub_buffer
        return sub_buffer

    def __call__(self):
        self.input_buffer.to("npu")
        super().__call__(
            self.input_buffer.buffer_object(),
            self.output_buffer.buffer_object(),
            self.scratch_buffer.buffer_object(),
        )
        self.output_buffer.to("cpu")
