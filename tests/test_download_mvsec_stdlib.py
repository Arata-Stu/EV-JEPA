from __future__ import annotations

import importlib.util
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "download"
    / "download_mvsec.py"
)
SPEC = importlib.util.spec_from_file_location("download_mvsec", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
download_mvsec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_mvsec
SPEC.loader.exec_module(download_mvsec)


def _npy_bytes(shape: tuple[int, ...], descriptor: str) -> bytes:
    header = repr(
        {"descr": descriptor, "fortran_order": False, "shape": shape}
    ).encode("latin1")
    padding = (-(10 + len(header) + 1)) % 16
    header += b" " * padding + b"\n"
    prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header))
    item_size = int(descriptor[-1])
    return prefix + header + b"\x00" * (math.prod(shape) * item_size)


def _corrupt_stored_member(path: Path, member_name: str) -> None:
    """Flip one payload byte while leaving the central-directory CRC unchanged."""

    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(member_name)
        header_offset = member.header_offset
    with path.open("r+b") as handle:
        handle.seek(header_offset)
        local_header = handle.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", local_header)
        if fields[0] != 0x04034B50:
            raise AssertionError("invalid synthetic local ZIP header")
        filename_length, extra_length = fields[-2:]
        payload_offset = header_offset + 30 + filename_length + extra_length
        handle.seek(payload_offset + member.file_size - 1)
        original = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original[0] ^ 0x01]))


class MVSECDownloadStdlibTests(unittest.TestCase):
    def test_profile_totals_and_membership(self) -> None:
        stage1 = download_mvsec.artifacts_for_profile("stage1", False)
        self.assertEqual(
            [artifact.filename for artifact in stage1],
            [
                "outdoor_day1_data.hdf5",
                "outdoor_day1_gt.hdf5",
                "outdoor_day1_gt_flow_dist.npz",
                "outdoor_day2_data.hdf5",
                "outdoor_day2_gt.hdf5",
                "outdoor_day2_gt_flow_dist.npz",
            ],
        )
        self.assertEqual(
            download_mvsec.profile_total_bytes("stage1"), 102_646_291_553
        )
        self.assertEqual(
            download_mvsec.profile_total_bytes("stage1-ood"), 122_227_194_379
        )
        self.assertEqual(
            download_mvsec.profile_total_bytes("stage1", True), 102_647_581_591
        )
        self.assertEqual(
            download_mvsec.profile_total_bytes("stage1-ood", True),
            122_229_776_899,
        )
        flow_artifacts = [artifact for artifact in stage1 if artifact.kind == "flow_npz"]
        self.assertEqual(
            [(artifact.file_id, artifact.expected_bytes) for artifact in flow_artifacts],
            [
                ("1XjJnriPh3k0FJo11or7X02myWARqVt7S", 7_389_716_086),
                ("1RIP-Fp0s7z9QtJTbsyqn_EMEiNwA7l1Y", 17_555_972_270),
            ],
        )

    def test_help_requires_no_download_dependency(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--plan-only", completed.stdout)
        self.assertIn("stage1-ood", completed.stdout)

    def test_plan_only_does_not_create_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "new" / "mvsec"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--root",
                    str(root),
                    "--profile",
                    "stage1-ood",
                    "--include-calibration",
                    "--plan-only",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("total_bytes: 122229776899", completed.stdout)
            self.assertIn("publisher_checksum: unavailable", completed.stdout)
            self.assertFalse(root.exists())

    def test_unsafe_roots_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "filesystem root"):
            download_mvsec.validate_root(Path(Path.cwd().anchor))
        with self.assertRaisesRegex(ValueError, "absolute"):
            download_mvsec.validate_root("relative/mvsec")
        with self.assertRaisesRegex(ValueError, "workspace"):
            download_mvsec.validate_root(Path.cwd())

    def test_remaining_bytes_account_for_one_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "datasets" / "mvsec"
            artifact = download_mvsec.Artifact(
                scene="scene",
                filename="sample.hdf5",
                file_id="fixed-id",
                expected_bytes=100,
                kind="data_hdf5",
            )
            final_path = download_mvsec.output_path(root, artifact)
            final_path.parent.mkdir(parents=True)
            part_path = final_path.with_name(final_path.name + ".part")
            part_path.write_bytes(b"x" * 40)
            self.assertEqual(
                download_mvsec.remaining_download_bytes(root, (artifact,)), 60
            )

    def test_flow_npz_schema_is_validated_without_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_flow.npz"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("timestamps.npy", _npy_bytes((2,), "<f8"))
                archive.writestr(
                    "x_flow_dist.npy", _npy_bytes((2, 260, 346), "<f4")
                )
                archive.writestr(
                    "y_flow_dist.npy", _npy_bytes((2, 260, 346), "<f4")
                )
            self.assertEqual(
                download_mvsec._validate_flow_npz(path),
                "npz-stored-npy-headers-crc",
            )

    def test_flow_npz_rejects_integer_or_mismatched_flow_dtypes(self) -> None:
        invalid_descriptors = (("<i4", "<i4"), ("<f4", "<f8"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            for number, (x_descriptor, y_descriptor) in enumerate(
                invalid_descriptors
            ):
                path = Path(temporary_directory) / f"invalid_{number}.npz"
                with zipfile.ZipFile(
                    path, "w", compression=zipfile.ZIP_STORED
                ) as archive:
                    archive.writestr("timestamps.npy", _npy_bytes((1,), "<f8"))
                    archive.writestr(
                        "x_flow_dist.npy",
                        _npy_bytes((1, 260, 346), x_descriptor),
                    )
                    archive.writestr(
                        "y_flow_dist.npy",
                        _npy_bytes((1, 260, 346), y_descriptor),
                    )
                with self.assertRaisesRegex(
                    download_mvsec.MVSECDownloadError, "floating dtypes"
                ):
                    download_mvsec._validate_flow_npz(path)

    def test_flow_npz_member_crc_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "corrupted_flow.npz"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("timestamps.npy", _npy_bytes((1,), "<f8"))
                archive.writestr(
                    "x_flow_dist.npy", _npy_bytes((1, 260, 346), "<f4")
                )
                archive.writestr(
                    "y_flow_dist.npy", _npy_bytes((1, 260, 346), "<f4")
                )
            _corrupt_stored_member(path, "x_flow_dist.npy")
            with self.assertRaisesRegex(
                download_mvsec.MVSECDownloadError,
                "CRC failure|invalid MVSEC flow NPZ",
            ):
                download_mvsec._validate_flow_npz(path)


if __name__ == "__main__":
    unittest.main()
