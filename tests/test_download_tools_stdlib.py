from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "download" / "archive_tool.py"
)
SPEC = importlib.util.spec_from_file_location("archive_tool", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
archive_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive_tool)


class DownloadToolTests(unittest.TestCase):
    def test_verify_and_resumable_extract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("train/sequence/events.bin", b"events")
            metadata_path = root / "sample.zip.verified.json"

            verified = archive_tool.verify_file(
                archive_path, "archive", metadata_path, archive_path.stat().st_size, None
            )
            checked = archive_tool.check_metadata(
                archive_path, "archive", metadata_path, archive_path.stat().st_size, None
            )
            self.assertEqual(verified["sha256"], checked["sha256"])

            expected_crc32 = f"{zlib.crc32(archive_path.read_bytes()):08x}"
            checked_crc = archive_tool.check_metadata(
                archive_path,
                "archive",
                metadata_path,
                archive_path.stat().st_size,
                None,
                expected_crc32,
            )
            self.assertEqual(checked_crc["crc32"], expected_crc32)
            with self.assertRaisesRegex(ValueError, "CRC32 mismatch"):
                archive_tool.check_metadata(
                    archive_path,
                    "archive",
                    metadata_path,
                    archive_path.stat().st_size,
                    None,
                    "00000000",
                )

            output = root / "raw"
            state = root / "state"
            archive_tool.extract_archive(archive_path, output, state)
            archive_tool.extract_archive(archive_path, output, state)
            extracted = output / "train" / "sequence" / "events.bin"
            self.assertEqual(extracted.read_bytes(), b"events")

            extracted.unlink()
            archive_tool.extract_archive(archive_path, output, state)
            self.assertEqual(extracted.read_bytes(), b"events")

            second_output = root / "second-raw"
            archive_tool.extract_archive(archive_path, second_output, state)
            self.assertEqual(
                (second_output / "train" / "sequence" / "events.bin").read_bytes(),
                b"events",
            )

            state_path = archive_tool._state_path(archive_path, output, state)
            with state_path.open("ab") as handle:
                handle.write(b'{"truncated":')
            archive_tool.extract_archive(archive_path, output, state)
            self.assertTrue(state_path.read_bytes().endswith(b"\n"))

    def test_extraction_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape", b"no")
            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                archive_tool.extract_archive(archive_path, root / "raw", root / "state")

    def test_http_identity_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            previous_headers = root / "previous.headers"
            current_headers = root / "current.headers"
            previous_headers.write_text(
                "HTTP/2 200\r\ncontent-length: 10\r\netag: \"one\"\r\n\r\n",
                encoding="latin-1",
            )
            current_headers.write_text(
                "HTTP/2 200\r\ncontent-length: 10\r\netag: \"two\"\r\n\r\n",
                encoding="latin-1",
            )
            previous = root / "previous.json"
            current = root / "current.json"
            archive_tool.write_http_identity(previous_headers, previous)
            archive_tool.write_http_identity(current_headers, current)
            with self.assertRaisesRegex(ValueError, "etag mismatch"):
                archive_tool.compare_http_identity(previous, current)

            length_only_headers = root / "length-only.headers"
            length_only_headers.write_text(
                "HTTP/2 200\r\ncontent-length: 10\r\n\r\n", encoding="latin-1"
            )
            length_only = root / "length-only.json"
            archive_tool.write_http_identity(length_only_headers, length_only)
            with self.assertRaisesRegex(ValueError, "unsafe resume"):
                archive_tool.compare_http_identity(length_only, length_only)
            archive_tool.compare_http_identity(
                length_only, length_only, allow_weak=True
            )

    def test_m3ed_plan_enforces_official_test_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_list = root / "dataset_list.yaml"
            dataset_list.write_text(
                "- file: calibration\n"
                "  filetype: camera_calib\n"
                "- file: train_recording\n"
                "  filetype: data\n"
                "  is_test_file: false\n"
                "  notes: >\n"
                "    A multiline description without a colon must not be\n"
                "    interpreted as another dataset-list field.\n"
                "- file: test_recording\n"
                "  filetype: data\n"
                "  is_test_file: true\n",
                encoding="utf-8",
            )
            sequence_list = root / "sequences.txt"
            sequence_list.write_text("test_recording\n", encoding="utf-8")
            archive_tool.validate_m3ed_plan(dataset_list, sequence_list, "test")
            with self.assertRaisesRegex(ValueError, "train/test boundary"):
                archive_tool.validate_m3ed_plan(dataset_list, sequence_list, "train")

    def test_prophesee_dat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split = root / "train"
            split.mkdir()
            dat_path = split / "recording_td.dat"
            dat_path.write_bytes(
                b"% Height 240\n% Width 304\n" + bytes((0, 8)) + b"\x00" * 8
            )
            (split / "recording_bbox.npy").write_bytes(b"placeholder")
            archive_tool.validate_prophesee(root, 304, 240, "train")
            script = MODULE_PATH.with_name("download_prophesee_gen1_dat.sh")
            completed = subprocess.run(
                [
                    str(script),
                    "--root",
                    str(root / "state"),
                    "--split",
                    "train",
                    "--extracted-root",
                    str(split),
                ],
                capture_output=True,
                text=True,
                check=False,
                env={"PATH": os.environ["PATH"], "PYTHON_BIN": sys.executable},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_rvt_genx_hdf5_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for dataset, suffix in (("gen4", "_td.h5"), ("gen1", "_td.dat.h5")):
                dataset_root = root / dataset
                split = dataset_root / "train"
                split.mkdir(parents=True)
                (split / "recording_bbox.npy").write_bytes(b"placeholder")
                (split / f"recording{suffix}").write_bytes(
                    archive_tool.HDF5_MAGIC + b"placeholder"
                )
                archive_tool.validate_rvt_genx(dataset_root, dataset, "train")
                script = MODULE_PATH.with_name(f"download_{dataset}.sh")
                completed = subprocess.run(
                    [
                        str(script),
                        "--extracted-root",
                        str(dataset_root),
                        "--split",
                        "train",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    env={"PATH": os.environ["PATH"], "PYTHON_BIN": sys.executable},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            (root / "gen4" / "train" / "recording_td.h5").unlink()
            with self.assertRaisesRegex(ValueError, "bbox file.*no event HDF5"):
                archive_tool.validate_rvt_genx(root / "gen4", "gen4", "train")

            second_bbox = root / "gen4" / "train" / "paired_bbox.npy"
            second_h5 = root / "gen4" / "train" / "paired_td.h5"
            second_bbox.write_bytes(b"placeholder")
            second_h5.write_bytes(archive_tool.HDF5_MAGIC + b"placeholder")
            archive_tool.validate_rvt_genx(
                root / "gen4", "gen4", "train", allow_orphan_bboxes=True
            )

    def test_normalized_archive_member_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "collision.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("folder/file", b"one")
                archive.writestr("folder//file", b"two")
            with self.assertRaisesRegex(ValueError, "duplicate ZIP member"):
                archive_tool.extract_archive(archive_path, root / "raw", root / "state")

    def test_existing_output_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "archive.zip"
            Path(str(output) + ".lock").mkdir()
            common = MODULE_PATH.with_name("_common.sh")
            command = (
                f'source "{common}"; '
                f'_download_acquire_lock "{output}"'
            )
            completed = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True, check=False
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("output is locked", completed.stderr)

    def test_private_url_plan_rejects_casefolded_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            url_file = root / "private.urls"
            url_file.write_text(
                "Archive.zip https://example.invalid/one\n"
                "archive.zip https://example.invalid/two\n",
                encoding="utf-8",
            )
            common = MODULE_PATH.with_name("_common.sh")
            command = (
                f'source "{common}"; '
                f'download_validate_private_url_file "{url_file}"'
            )
            completed = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True, check=False
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate filename", completed.stderr)


if __name__ == "__main__":
    unittest.main()
