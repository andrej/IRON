"""
Pure Python reimplementation of the xclbin-flow patching utilities from
mlir-aie/runtime_lib/test_lib/test_utils.{h,cpp}.

Provides functions to:
  - Parse the insts_offsets.json produced by the compiler
  - Patch LOAD_PDI instructions (append PDI binary, fix address/size fields)
  - Patch write32 / RTP instructions by name
  - Scan for magic sentinel values in the instruction uint32 array
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Data classes mirroring the C++ structs
# ---------------------------------------------------------------------------


@dataclass
class LoadPdiPatchInfo:
    """Mirrors test_utils::LoadPdiPatchInfo."""

    load_pdi_offset_bytes: int
    address_field_offset_bytes: int
    size_field_offset_bytes: int
    pdi_id: int


@dataclass
class Write32PatchInfo:
    """Mirrors test_utils::Write32PatchInfo."""

    name: str
    offset_bytes: int
    value_field_offset_bytes: int


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_offsets_json(json_path: str | Path) -> dict:
    """Read the full offsets JSON file and return the parsed dict."""
    with open(json_path, "r") as f:
        return json.load(f)


def parse_instr_offsets_json(json_path: str | Path) -> list[LoadPdiPatchInfo]:
    """Parse *load_pdi* entries from the offsets JSON.

    Equivalent to ``test_utils::parse_instr_offsets_json()``.
    """
    data = parse_offsets_json(json_path)
    results: list[LoadPdiPatchInfo] = []
    for entry in data.get("instructions", []):
        if entry.get("type") != "load_pdi":
            continue
        results.append(
            LoadPdiPatchInfo(
                load_pdi_offset_bytes=entry["offset_bytes"],
                address_field_offset_bytes=entry["address_field_offset_bytes"],
                size_field_offset_bytes=entry["size_field_offset_bytes"],
                pdi_id=entry["pdi_id"],
            )
        )
    return results


def parse_write32_offsets_json(json_path: str | Path) -> list[Write32PatchInfo]:
    """Parse *write32* (RTP) entries from the offsets JSON.

    Equivalent to ``test_utils::parse_write32_offsets_json()``.
    """
    data = parse_offsets_json(json_path)
    results: list[Write32PatchInfo] = []
    for entry in data.get("instructions", []):
        if entry.get("type") != "write32":
            continue
        results.append(
            Write32PatchInfo(
                name=entry.get("name", ""),
                offset_bytes=entry["offset_bytes"],
                value_field_offset_bytes=entry["value_field_offset_bytes"],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Patching functions
# ---------------------------------------------------------------------------


def patch_load_pdi(
    instr_v: np.ndarray,
    patch_infos: Sequence[LoadPdiPatchInfo],
    pdi_path: str | Path,
) -> np.ndarray:
    """Append PDI binary data to *instr_v* and patch address/size fields.

    Equivalent to ``test_utils::patch_load_pdi()``.

    Parameters
    ----------
    instr_v : np.ndarray
        1-D uint32 instruction array (will NOT be mutated).
    patch_infos : sequence of LoadPdiPatchInfo
        Entries produced by :func:`parse_instr_offsets_json`.
    pdi_path : path-like
        Path to the ``.pdi`` binary file.

    Returns
    -------
    np.ndarray
        New uint32 array with the PDI data appended and address/size fields
        patched.
    """
    pdi_data = Path(pdi_path).read_bytes()
    pdi_size = len(pdi_data)

    # Record the byte offset where PDI data will be appended
    pdi_append_offset = np.uint32(instr_v.size * 4)

    # Pad to 4-byte alignment
    padded_size = (pdi_size + 3) & ~3
    pdi_padded = pdi_data + b"\x00" * (padded_size - pdi_size)

    # Convert PDI bytes to uint32 words and append
    pdi_words = np.frombuffer(pdi_padded, dtype=np.uint32)
    instr_v = np.concatenate([instr_v, pdi_words])

    # Patch each LOAD_PDI instruction's address and size fields
    pdi_size_u32 = np.uint32(pdi_size)
    for info in patch_infos:
        addr_idx = info.address_field_offset_bytes // 4
        size_idx = info.size_field_offset_bytes // 4
        # Address field (lower 32 bits)
        instr_v[addr_idx] = pdi_append_offset
        # Upper 32 bits = 0
        instr_v[addr_idx + 1] = np.uint32(0)
        # Size field
        instr_v[size_idx] = pdi_size_u32

    return instr_v


def patch_rtp(
    instr_v: np.ndarray,
    infos: Sequence[Write32PatchInfo],
    name: str,
    value: int,
) -> None:
    """Patch an RTP parameter by name in the instruction array.

    Equivalent to ``test_utils::patch_rtp()``.

    Parameters
    ----------
    instr_v : np.ndarray
        1-D uint32 instruction array (mutated in-place).
    infos : sequence of Write32PatchInfo
        Entries produced by :func:`parse_write32_offsets_json`.
    name : str
        The RTP parameter name to patch.
    value : int
        The uint32 value to write.

    Raises
    ------
    KeyError
        If no entry with the given *name* is found.
    """
    for info in infos:
        if info.name == name:
            instr_v[info.value_field_offset_bytes // 4] = np.uint32(value)
            return
    raise KeyError(f"RTP entry not found: {name!r}")


def find_magic_value_offsets(
    instr_v: np.ndarray,
    magic: int = 0xDEADBEE0,
) -> list[int]:
    """Scan the uint32 instruction array for a magic sentinel value.

    Returns a list of uint32 word indices where the magic value appears.
    This is used for the short-term StridedCopy DMA-offset patching approach
    where the compiler emits a placeholder (0xDEADBEE0) that must be replaced
    at runtime with the actual DMA address offset.

    Parameters
    ----------
    instr_v : np.ndarray
        1-D uint32 instruction array.
    magic : int
        The sentinel value to search for (default ``0xDEADBEE0``).

    Returns
    -------
    list[int]
        Word indices (not byte offsets) where the magic value was found.
    """
    magic_u32 = np.uint32(magic)
    (indices,) = np.where(instr_v == magic_u32)
    return indices.tolist()


def patch_magic_value(
    instr_v: np.ndarray,
    value: int,
    magic: int = 0xDEADBEE0,
) -> int:
    """Replace all occurrences of a magic sentinel with the given value.

    Parameters
    ----------
    instr_v : np.ndarray
        1-D uint32 instruction array (mutated in-place).
    value : int
        The replacement uint32 value.
    magic : int
        The sentinel value to replace (default ``0xDEADBEE0``).

    Returns
    -------
    int
        Number of replacements made.
    """
    indices = find_magic_value_offsets(instr_v, magic)
    for idx in indices:
        instr_v[idx] = np.uint32(value)
    return len(indices)
