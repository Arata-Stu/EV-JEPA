from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    PROJECT_ROOT / "scripts/preprocess/preprocess_dsec.sh",
    PROJECT_ROOT / "scripts/preprocess/preprocess_gen1.sh",
    PROJECT_ROOT / "scripts/preprocess/preprocess_mvsec.sh",
)


MVSEC_WRAPPER = PROJECT_ROOT / "scripts/preprocess/preprocess_mvsec.sh"
STAGE1_PATHS = (
    "outdoor_day/outdoor_day1_data.hdf5",
    "outdoor_day/outdoor_day1_gt.hdf5",
    "outdoor_day/outdoor_day1_gt_flow_dist.npz",
    "outdoor_day/outdoor_day2_data.hdf5",
    "outdoor_day/outdoor_day2_gt.hdf5",
    "outdoor_day/outdoor_day2_gt_flow_dist.npz",
)
NIGHT_PATHS = (
    "outdoor_night/outdoor_night1_data.hdf5",
    "outdoor_night/outdoor_night1_gt.hdf5",
)


def _populate(root: Path, paths: tuple[str, ...] = STAGE1_PATHS) -> None:
    for relative_path in paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _fake_python(root: Path) -> tuple[Path, Path]:
    executable = root / "fake-python"
    log = root / "fake-python.log"
    executable.write_text(
        "#!/bin/sh\n"
        "{\n"
        "  printf 'CALL'\n"
        "  for argument in \"$@\"; do printf '\\t%s' \"$argument\"; done\n"
        "  printf '\\n'\n"
        "} >> \"$MVSEC_FAKE_PYTHON_LOG\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, log


def _calls(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [line.split("\t")[1:] for line in log.read_text().splitlines()]


class PreprocessWrapperStdlibTests(unittest.TestCase):
    def test_native_resolution_preprocess_wrappers_have_valid_bash_syntax(self) -> None:
        for wrapper in WRAPPERS:
            subprocess.run(["bash", "-n", str(wrapper)], check=True)

    def test_native_resolution_preprocess_wrappers_expose_help(self) -> None:
        for wrapper in WRAPPERS:
            completed = subprocess.run(
                ["bash", str(wrapper), "--help"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertNotIn("--spatial-downsample", completed.stdout)

    def test_native_resolution_preprocess_wrappers_fix_coordinate_conversion(self) -> None:
        for wrapper in WRAPPERS:
            source = wrapper.read_text(encoding="utf-8")
            self.assertIn("--spatial-downsample 1", source)
            self.assertIn("--spatial-downsample-method coordinate", source)
            self.assertNotIn("area_accumulate", source)

    def _run_mvsec(
        self,
        root: Path,
        fake_python: Path,
        log: Path,
        *,
        include_night: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            str(MVSEC_WRAPPER),
            "--raw-root",
            str(root),
            "--bundle-root",
            str(root.parent / "processed bundle"),
            "--python-bin",
            str(fake_python),
            "--plan-only",
        ]
        if include_night:
            command.append("--include-night")
        environment = os.environ.copy()
        environment["MVSEC_FAKE_PYTHON_LOG"] = str(log)
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_mvsec_gui_direct_layout_is_used_and_zip_residue_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "GUI MVSEC"
            _populate(root)
            archives = (
                root / "indoor_flying-20260905T004245Z-1-002.zip",
                root / "outdoor_day-20260905T004317Z-1-005.zip",
            )
            for archive in archives:
                archive.write_bytes(b"GUI archive residue")
            fake_python, log = _fake_python(temporary_root)

            completed = self._run_mvsec(root, fake_python, log)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                f"Resolved MVSEC raw root (direct layout): {root.resolve()}",
                completed.stdout,
            )
            calls = _calls(log)
            self.assertEqual(len(calls), 3)
            for call in calls:
                input_index = call.index("--input")
                self.assertEqual(call[input_index + 1], str(root.resolve()))
            for archive in archives:
                self.assertEqual(archive.read_bytes(), b"GUI archive residue")

    def test_mvsec_downloader_container_layout_resolves_to_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "managed"
            dataset_root = root / "raw"
            _populate(dataset_root)
            fake_python, log = _fake_python(temporary_root)

            completed = self._run_mvsec(root, fake_python, log)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                f"Resolved MVSEC raw root (downloader layout): "
                f"{dataset_root.resolve()}",
                completed.stdout,
            )
            calls = _calls(log)
            self.assertEqual(len(calls), 3)
            for call in calls:
                input_index = call.index("--input")
                self.assertEqual(call[input_index + 1], str(dataset_root.resolve()))

    def test_mvsec_ambiguous_direct_and_downloader_layouts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "ambiguous"
            _populate(root)
            _populate(root / "raw")
            fake_python, log = _fake_python(temporary_root)

            completed = self._run_mvsec(root, fake_python, log)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("ambiguous MVSEC raw root", completed.stderr)
            self.assertIn(str(root / "outdoor_day"), completed.stderr)
            self.assertIn(str(root / "raw" / "outdoor_day"), completed.stderr)
            self.assertEqual(_calls(log), [])

    def test_mvsec_plan_reports_all_missing_stage1_files_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "incomplete"
            (root / "outdoor_day").mkdir(parents=True)
            # One present file proves that only the missing subset is reported.
            _populate(root, (STAGE1_PATHS[0],))
            fake_python, log = _fake_python(temporary_root)

            completed = self._run_mvsec(root, fake_python, log)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("required MVSEC Stage 1 files are missing", completed.stderr)
            self.assertNotIn(STAGE1_PATHS[0], completed.stderr)
            for relative_path in STAGE1_PATHS[1:]:
                self.assertIn(str(root / relative_path), completed.stderr)
            self.assertEqual(_calls(log), [])

    def test_mvsec_gui_hdf5_bundle_reports_only_the_two_missing_flow_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "GUI MVSEC"
            hdf5_paths = tuple(
                path for path in STAGE1_PATHS if path.endswith(".hdf5")
            )
            _populate(root, hdf5_paths)
            fake_python, log = _fake_python(temporary_root)

            completed = self._run_mvsec(root, fake_python, log)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                str(root / "outdoor_day/outdoor_day1_gt_flow_dist.npz"),
                completed.stderr,
            )
            self.assertIn(
                str(root / "outdoor_day/outdoor_day2_gt_flow_dist.npz"),
                completed.stderr,
            )
            self.assertNotIn("_data.hdf5", completed.stderr)
            self.assertNotIn("_gt.hdf5", completed.stderr)
            self.assertIn(
                "https://daniilidis-group.github.io/mvsec/download/",
                completed.stderr,
            )
            self.assertEqual(_calls(log), [])

    def test_mvsec_night_files_are_required_only_with_include_night(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "direct"
            _populate(root)
            fake_python, log = _fake_python(temporary_root)

            without_night = self._run_mvsec(root, fake_python, log)
            self.assertEqual(without_night.returncode, 0, without_night.stderr)
            self.assertEqual(len(_calls(log)), 3)

            log.unlink()
            missing_night = self._run_mvsec(
                root, fake_python, log, include_night=True
            )
            self.assertNotEqual(missing_night.returncode, 0)
            for relative_path in NIGHT_PATHS:
                self.assertIn(str(root / relative_path), missing_night.stderr)
            self.assertEqual(_calls(log), [])

            _populate(root, NIGHT_PATHS)
            with_night = self._run_mvsec(root, fake_python, log, include_night=True)
            self.assertEqual(with_night.returncode, 0, with_night.stderr)
            self.assertEqual(len(_calls(log)), 4)

    def test_mvsec_raw_root_symlink_is_resolved_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            physical_root = temporary_root / "physical"
            _populate(physical_root)
            linked_root = temporary_root / "linked"
            linked_root.symlink_to(physical_root, target_is_directory=True)
            fake_python, log = _fake_python(temporary_root)

            completed = self._run_mvsec(linked_root, fake_python, log)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = _calls(log)
            self.assertEqual(len(calls), 3)
            for call in calls:
                input_index = call.index("--input")
                self.assertEqual(call[input_index + 1], str(physical_root.resolve()))


if __name__ == "__main__":
    unittest.main()
