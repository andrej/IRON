"""Tests for iron.common.xclbin_patching.

Verifies the pure-Python reimplementation matches the behaviour of the C++
functions in mlir-aie/runtime_lib/test_lib/test_utils.{h,cpp}.
"""

import json
import os
import struct
import tempfile

import numpy as np
import pytest

from iron.common.xclbin_patching import (
    LoadPdiPatchInfo,
    Write32PatchInfo,
    find_magic_value_offsets,
    parse_instr_offsets_json,
    parse_write32_offsets_json,
    patch_load_pdi,
    patch_magic_value,
    patch_rtp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _make_offsets_json(tmpdir: str, instructions: list[dict]) -> str:
    path = os.path.join(tmpdir, "insts_offsets.json")
    _write_json({"instructions": instructions}, path)
    return path


# ---------------------------------------------------------------------------
# parse_instr_offsets_json
# ---------------------------------------------------------------------------


class TestParseInstrOffsetsJson:
    def test_basic(self, tmp_path):
        json_path = _make_offsets_json(
            str(tmp_path),
            [
                {
                    "type": "load_pdi",
                    "offset_bytes": 16,
                    "pdi_id": 1,
                    "address_field_offset_bytes": 24,
                    "size_field_offset_bytes": 20,
                },
                {
                    "type": "write32",
                    "name": "rtp_foo",
                    "offset_bytes": 224,
                    "value_field_offset_bytes": 240,
                },
            ],
        )
        infos = parse_instr_offsets_json(json_path)
        assert len(infos) == 1
        assert infos[0].load_pdi_offset_bytes == 16
        assert infos[0].pdi_id == 1
        assert infos[0].address_field_offset_bytes == 24
        assert infos[0].size_field_offset_bytes == 20

    def test_multiple_load_pdi(self, tmp_path):
        json_path = _make_offsets_json(
            str(tmp_path),
            [
                {
                    "type": "load_pdi",
                    "offset_bytes": 16,
                    "pdi_id": 1,
                    "address_field_offset_bytes": 24,
                    "size_field_offset_bytes": 20,
                },
                {
                    "type": "load_pdi",
                    "offset_bytes": 316,
                    "pdi_id": 2,
                    "address_field_offset_bytes": 324,
                    "size_field_offset_bytes": 320,
                },
            ],
        )
        infos = parse_instr_offsets_json(json_path)
        assert len(infos) == 2
        assert infos[0].pdi_id == 1
        assert infos[1].pdi_id == 2

    def test_empty(self, tmp_path):
        json_path = _make_offsets_json(str(tmp_path), [])
        assert parse_instr_offsets_json(json_path) == []


# ---------------------------------------------------------------------------
# parse_write32_offsets_json
# ---------------------------------------------------------------------------


class TestParseWrite32OffsetsJson:
    def test_basic(self, tmp_path):
        json_path = _make_offsets_json(
            str(tmp_path),
            [
                {
                    "type": "load_pdi",
                    "offset_bytes": 16,
                    "pdi_id": 1,
                    "address_field_offset_bytes": 24,
                    "size_field_offset_bytes": 20,
                },
                {
                    "type": "write32",
                    "name": "rtp_alpha",
                    "offset_bytes": 224,
                    "value_field_offset_bytes": 240,
                },
                {
                    "type": "write32",
                    "name": "rtp_beta",
                    "offset_bytes": 276,
                    "value_field_offset_bytes": 292,
                },
            ],
        )
        infos = parse_write32_offsets_json(json_path)
        assert len(infos) == 2
        assert infos[0].name == "rtp_alpha"
        assert infos[0].offset_bytes == 224
        assert infos[0].value_field_offset_bytes == 240
        assert infos[1].name == "rtp_beta"

    def test_no_name_defaults_empty(self, tmp_path):
        json_path = _make_offsets_json(
            str(tmp_path),
            [
                {
                    "type": "write32",
                    "offset_bytes": 100,
                    "value_field_offset_bytes": 116,
                }
            ],
        )
        infos = parse_write32_offsets_json(json_path)
        assert len(infos) == 1
        assert infos[0].name == ""


# ---------------------------------------------------------------------------
# patch_rtp
# ---------------------------------------------------------------------------


class TestPatchRtp:
    def test_patch_by_name(self):
        instr = np.zeros(100, dtype=np.uint32)
        infos = [
            Write32PatchInfo(name="rtp_alpha", offset_bytes=224, value_field_offset_bytes=240),
            Write32PatchInfo(name="rtp_beta", offset_bytes=276, value_field_offset_bytes=292),
        ]
        patch_rtp(instr, infos, "rtp_alpha", 42)
        assert instr[240 // 4] == 42
        # rtp_beta should be untouched
        assert instr[292 // 4] == 0

    def test_patch_second_entry(self):
        instr = np.zeros(100, dtype=np.uint32)
        infos = [
            Write32PatchInfo(name="a", offset_bytes=0, value_field_offset_bytes=4),
            Write32PatchInfo(name="b", offset_bytes=8, value_field_offset_bytes=12),
        ]
        patch_rtp(instr, infos, "b", 0xCAFEBABE)
        assert instr[12 // 4] == 0xCAFEBABE

    def test_not_found_raises(self):
        instr = np.zeros(10, dtype=np.uint32)
        infos = [
            Write32PatchInfo(name="x", offset_bytes=0, value_field_offset_bytes=4),
        ]
        with pytest.raises(KeyError, match="RTP entry not found"):
            patch_rtp(instr, infos, "nonexistent", 1)


# ---------------------------------------------------------------------------
# patch_load_pdi
# ---------------------------------------------------------------------------


class TestPatchLoadPdi:
    def test_append_and_patch(self, tmp_path):
        # Create a small PDI file (16 bytes = 4 uint32 words, already aligned)
        pdi_content = struct.pack("<4I", 0x11111111, 0x22222222, 0x33333333, 0x44444444)
        pdi_path = tmp_path / "test.pdi"
        pdi_path.write_bytes(pdi_content)

        # Instruction array: 20 words (80 bytes)
        # address field at byte 24 → index 6
        # size field at byte 20 → index 5
        instr = np.zeros(20, dtype=np.uint32)
        patch_infos = [
            LoadPdiPatchInfo(
                load_pdi_offset_bytes=16,
                address_field_offset_bytes=24,
                size_field_offset_bytes=20,
                pdi_id=1,
            )
        ]

        result = patch_load_pdi(instr, patch_infos, str(pdi_path))

        # Array should now be 20 + 4 = 24 words
        assert result.shape[0] == 24
        # PDI data should be appended at the end
        assert result[20] == 0x11111111
        assert result[21] == 0x22222222
        assert result[22] == 0x33333333
        assert result[23] == 0x44444444
        # Address field (index 6) = byte offset where PDI was appended = 20 * 4 = 80
        assert result[6] == 80
        # Upper 32 bits of address (index 7) = 0
        assert result[7] == 0
        # Size field (index 5) = 16 bytes
        assert result[5] == 16

    def test_unaligned_pdi(self, tmp_path):
        # PDI file with 5 bytes (needs padding to 8)
        pdi_content = b"\x01\x02\x03\x04\x05"
        pdi_path = tmp_path / "unaligned.pdi"
        pdi_path.write_bytes(pdi_content)

        instr = np.zeros(10, dtype=np.uint32)
        patch_infos = [
            LoadPdiPatchInfo(
                load_pdi_offset_bytes=0,
                address_field_offset_bytes=24,
                size_field_offset_bytes=20,
                pdi_id=1,
            )
        ]

        result = patch_load_pdi(instr, patch_infos, str(pdi_path))
        # 5 bytes padded to 8 = 2 words appended
        assert result.shape[0] == 12
        # Size field should be the original unpadded size
        assert result[20 // 4] == 5

    def test_multiple_patch_infos(self, tmp_path):
        pdi_content = struct.pack("<2I", 0xAAAAAAAA, 0xBBBBBBBB)
        pdi_path = tmp_path / "multi.pdi"
        pdi_path.write_bytes(pdi_content)

        # Two LOAD_PDI entries pointing to different address/size field locations
        instr = np.zeros(100, dtype=np.uint32)
        patch_infos = [
            LoadPdiPatchInfo(
                load_pdi_offset_bytes=16,
                address_field_offset_bytes=24,
                size_field_offset_bytes=20,
                pdi_id=1,
            ),
            LoadPdiPatchInfo(
                load_pdi_offset_bytes=316,
                address_field_offset_bytes=324,
                size_field_offset_bytes=320,
                pdi_id=2,
            ),
        ]

        result = patch_load_pdi(instr, patch_infos, str(pdi_path))
        pdi_offset = 100 * 4  # 400 bytes

        # Both entries should point to the same PDI append location
        assert result[24 // 4] == pdi_offset
        assert result[324 // 4] == pdi_offset
        assert result[20 // 4] == 8
        assert result[320 // 4] == 8


# ---------------------------------------------------------------------------
# Magic value scanning/patching
# ---------------------------------------------------------------------------


class TestMagicValue:
    def test_find_magic(self):
        instr = np.zeros(20, dtype=np.uint32)
        instr[5] = np.uint32(0xDEADBEE0)
        instr[15] = np.uint32(0xDEADBEE0)
        indices = find_magic_value_offsets(instr)
        assert indices == [5, 15]

    def test_find_no_magic(self):
        instr = np.zeros(10, dtype=np.uint32)
        assert find_magic_value_offsets(instr) == []

    def test_custom_magic(self):
        instr = np.zeros(10, dtype=np.uint32)
        instr[3] = np.uint32(0xBA5EBA11)
        indices = find_magic_value_offsets(instr, magic=0xBA5EBA11)
        assert indices == [3]

    def test_patch_magic(self):
        instr = np.zeros(20, dtype=np.uint32)
        instr[5] = np.uint32(0xDEADBEE0)
        instr[15] = np.uint32(0xDEADBEE0)
        count = patch_magic_value(instr, 0x12345678)
        assert count == 2
        assert instr[5] == 0x12345678
        assert instr[15] == 0x12345678

    def test_patch_magic_ba5eba11(self):
        instr = np.zeros(10, dtype=np.uint32)
        instr[7] = np.uint32(0xBA5EBA11)
        count = patch_magic_value(instr, 256, magic=0xBA5EBA11)
        assert count == 1
        assert instr[7] == 256


# ---------------------------------------------------------------------------
# Integration test with real JSON files from mlir-aie build
# ---------------------------------------------------------------------------


class TestWithRealJson:
    LOADPDI_JSON = "/workspace/mlir-aie/build/test/npu-xrt/loadpdi_xclbin/insts_offsets.json"
    RTP_JSON = "/workspace/mlir-aie/build/test/Targets/NPU/rtp_offsets_test.json"
    RECONFIG_JSON = "/workspace/mlir-aie/build/test/npu-xrt/reconfigure_loadpdi_xclbin/insts_offsets.json"

    @pytest.mark.skipif(
        not os.path.exists(LOADPDI_JSON), reason="build artifacts not present"
    )
    def test_parse_loadpdi_json(self):
        infos = parse_instr_offsets_json(self.LOADPDI_JSON)
        assert len(infos) == 1
        assert infos[0].pdi_id == 1

    @pytest.mark.skipif(
        not os.path.exists(LOADPDI_JSON), reason="build artifacts not present"
    )
    def test_parse_write32_from_loadpdi_json(self):
        infos = parse_write32_offsets_json(self.LOADPDI_JSON)
        assert len(infos) == 2

    @pytest.mark.skipif(
        not os.path.exists(RTP_JSON), reason="build artifacts not present"
    )
    def test_parse_rtp_json(self):
        infos = parse_write32_offsets_json(self.RTP_JSON)
        assert len(infos) == 1
        assert infos[0].name == "rtp2"
        assert infos[0].value_field_offset_bytes == 32

    @pytest.mark.skipif(
        not os.path.exists(RECONFIG_JSON), reason="build artifacts not present"
    )
    def test_parse_reconfig_json(self):
        load_pdis = parse_instr_offsets_json(self.RECONFIG_JSON)
        write32s = parse_write32_offsets_json(self.RECONFIG_JSON)
        assert len(load_pdis) == 2
        assert len(write32s) == 4
        assert load_pdis[0].pdi_id == 1
        assert load_pdis[1].pdi_id == 2
