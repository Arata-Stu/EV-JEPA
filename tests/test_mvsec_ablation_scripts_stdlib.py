from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER = PROJECT_ROOT / "scripts/experiments/mvsec_ablation_config.py"
TRAIN = PROJECT_ROOT / "scripts/experiments/train_mvsec_ablation.sh"
EVAL = PROJECT_ROOT / "scripts/experiments/eval_mvsec_ablation.sh"
VISUALIZE = PROJECT_ROOT / "scripts/experiments/visualize_mvsec_ablation.sh"
JEPA_TEMPLATE = (
    PROJECT_ROOT / "configs/pretrain/recurrent_future_convlstm_vits_mvsec.yaml"
)
CMAX_TEMPLATE = (
    PROJECT_ROOT / "configs/pretrain/recurrent_future_convlstm_vits_mvsec_cmax.yaml"
)


def run(*command: str | Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


class MVSECAblationHelperTests(unittest.TestCase):
    def test_all_matrix_is_unique_and_contains_independent_suites(self) -> None:
        result = run(sys.executable, HELPER, "matrix", "--suite", "all", "--seeds", "0")
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 25)
        self.assertEqual(len({row[0] for row in rows}), 25)
        names = {row[1] for row in rows}
        self.assertIn("jepa", names)
        self.assertIn("jepa_cmax_w0p05_tsig_w0p02", names)
        self.assertIn("jepa_cmax_w0p05_ref_future", names)
        self.assertIn("jepa_cmax_w0p05_scales1", names)
        self.assertIn("jepa_frame_support_sigreg_off", names)
        self.assertIn("jepa_ra_w0p01_g0p5", names)
        self.assertIn("jepa_ls_w0p001", names)
        self.assertIn("jepa_ra_w0p01_g1_ls_w0p01", names)
        self.assertIn("jepa_cmax_w0p05_ra_w0p01_g1_ls_w0p01", names)

    def test_latent_matrices_are_explicit_two_by_two_designs(self) -> None:
        latent = run(
            sys.executable, HELPER, "matrix", "--suite", "latent", "--seeds", "3"
        )
        latent_rows = [line.split("\t") for line in latent.stdout.splitlines()]
        self.assertEqual(
            [(row[1], row[9], row[10], row[13]) for row in latent_rows],
            [
                ("jepa", "0", "1", "0"),
                ("jepa_ra_w0p01_g1", "0.01", "1", "0"),
                ("jepa_ls_w0p01", "0", "1", "0.01"),
                ("jepa_ra_w0p01_g1_ls_w0p01", "0.01", "1", "0.01"),
            ],
        )
        latent_cmax = run(
            sys.executable,
            HELPER,
            "matrix",
            "--suite",
            "latent_cmax",
            "--seeds",
            "3",
        )
        cmax_rows = [line.split("\t") for line in latent_cmax.stdout.splitlines()]
        self.assertEqual(
            [(row[3], row[9], row[13]) for row in cmax_rows],
            [("0", "0", "0"), ("0", "0.01", "0.01"),
             ("0.05", "0", "0"), ("0.05", "0.01", "0.01")],
        )
        self.assertTrue(
            all(row[11] == "0.000001" and row[14] == "0.000001" for row in cmax_rows)
        )
        self.assertTrue(
            all(row[12] == "per_clip_mean_supported_patch_rate" for row in cmax_rows)
        )
        selected = run(
            sys.executable,
            HELPER,
            "lookup",
            "--run-id",
            "mvsec_jepa_cmax_w0p05_ra_w0p01_g1_ls_w0p01__seed3",
        )
        selected_row = selected.stdout.rstrip("\n").split("\t")
        self.assertEqual(selected_row[3], "0.05")
        self.assertEqual(selected_row[9:15], cmax_rows[-1][9:15])

    def test_rate_and_straightening_sweeps_use_the_prespecified_scales(self) -> None:
        rate = run(
            sys.executable,
            HELPER,
            "matrix",
            "--suite",
            "rate_alignment",
            "--seeds",
            "0",
        )
        rate_rows = [line.split("\t") for line in rate.stdout.splitlines()]
        self.assertEqual(
            [(row[9], row[10], row[13]) for row in rate_rows],
            [("0", "1", "0"), ("0.001", "1", "0"),
             ("0.01", "1", "0"), ("0.05", "1", "0")],
        )
        gamma = run(
            sys.executable,
            HELPER,
            "matrix",
            "--suite",
            "rate_gamma",
            "--seeds",
            "0",
        )
        gamma_rows = [line.split("\t") for line in gamma.stdout.splitlines()]
        self.assertEqual(
            [(row[9], row[10]) for row in gamma_rows],
            [("0.01", "0.5"), ("0.01", "1"), ("0.01", "2")],
        )
        straightening = run(
            sys.executable,
            HELPER,
            "matrix",
            "--suite",
            "straightening",
            "--seeds",
            "0",
        )
        straightening_rows = [
            line.split("\t") for line in straightening.stdout.splitlines()
        ]
        self.assertEqual(
            [row[13] for row in straightening_rows],
            ["0", "0.001", "0.01", "0.05"],
        )

    def test_matrix_rejects_duplicate_seed(self) -> None:
        result = run(
            sys.executable,
            HELPER,
            "matrix",
            "--suite",
            "core",
            "--seeds",
            "0,0",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate seed", result.stderr)

    def test_manifest_validator_rejects_recording_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for camera in ("left", "right"):
                event_path = root / f"{camera}.h5"
                event_path.touch()
                rows.append(
                    {
                        "sequence_id": f"mvsec__outdoor_day2__{camera}",
                        "source_recording_id": "mvsec__outdoor_day2",
                        "dataset": "mvsec",
                        "split": "train",
                        "camera": camera,
                        "height": 260,
                        "width": 346,
                        "source_height": 260,
                        "source_width": 346,
                        "coordinate_frame": "distorted",
                        "spatial_downsample": 1,
                        "path": event_path.name,
                    }
                )
            manifest = root / "train.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = run(
                sys.executable,
                HELPER,
                "manifest",
                "--path",
                manifest,
                "--expected-recording",
                "outdoor_day2",
                "--expected-split",
                "train",
                "--expected-cameras",
                "left,right",
                "--require-artifacts",
            )
            self.assertIn('"rows": 2', result.stdout)
            rows[1]["source_recording_id"] = "mvsec__outdoor_day1"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            mismatched_source = run(
                sys.executable,
                HELPER,
                "manifest",
                "--path",
                manifest,
                "--expected-recording",
                "outdoor_day2",
                "--expected-split",
                "train",
                "--expected-cameras",
                "left,right",
                check=False,
            )
            self.assertNotEqual(mismatched_source.returncode, 0)
            self.assertIn("source_recording_id", mismatched_source.stderr)
            rows[1]["sequence_id"] = "mvsec__outdoor_day1__right"
            rows[1]["source_recording_id"] = "mvsec__outdoor_day2"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            leaked = run(
                sys.executable,
                HELPER,
                "manifest",
                "--path",
                manifest,
                "--expected-recording",
                "outdoor_day2",
                "--expected-split",
                "train",
                "--expected-cameras",
                "left,right",
                check=False,
            )
            self.assertNotEqual(leaked.returncode, 0)
            self.assertIn("sequence_id", leaked.stderr)

    def test_snapshot_index_lists_only_declared_local_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "sample.npz"
            snapshot.write_bytes(b"fixed-snapshot-payload")
            snapshot_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "schema": "event-window-jepa-mvsec-visualization-index-v1",
                        "samples": [
                            {
                                "path": str(snapshot),
                                "kind": "flow",
                                "bytes": snapshot.stat().st_size,
                                "sha256": snapshot_sha256,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = run(
                sys.executable,
                HELPER,
                "snapshot-index",
                "--path",
                index,
                "--limit",
                "1",
            )
            self.assertEqual(
                result.stdout.strip(),
                (
                    f"0\t{snapshot.resolve()}\tflow\t{snapshot.stat().st_size}"
                    f"\t{snapshot_sha256}"
                ),
            )

            snapshot.write_bytes(b"tampered-snapshot-payload")
            tampered = run(
                sys.executable,
                HELPER,
                "snapshot-index",
                "--path",
                index,
                check=False,
            )
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("snapshot byte size changed", tampered.stderr)

            snapshot.write_bytes(b"!" * len(b"fixed-snapshot-payload"))
            same_size_tampered = run(
                sys.executable,
                HELPER,
                "snapshot-index",
                "--path",
                index,
                check=False,
            )
            self.assertNotEqual(same_size_tampered.returncode, 0)
            self.assertIn("snapshot SHA-256 changed", same_size_tampered.stderr)

    def test_completion_marker_verifies_artifacts_identity_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint-bytes")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "checkpoint": {"sha256": checkpoint_sha},
                        "runtime": {
                            "epochs": 1.0,
                            "seed": 7,
                            "weight_decay": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            marker = root / "complete.json"
            common = (
                "--path",
                marker,
                "--kind",
                "evaluation",
                "--identity",
                "run_id=mvsec_jepa__seed0",
                "--artifact",
                f"checkpoint={checkpoint}",
                "--artifact",
                f"report={report}",
                "--report-field",
                "checkpoint.sha256=@sha256:checkpoint",
                "--report-field",
                "runtime.epochs=1",
                "--report-field",
                "runtime.seed=7",
                "--report-field",
                "runtime.weight_decay=0",
            )
            run(sys.executable, HELPER, "completion", "--action", "record", *common)
            verified = run(
                sys.executable, HELPER, "completion", "--action", "verify", *common
            )
            self.assertIn("verified", verified.stdout)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["artifacts"]["checkpoint"]["sha256"], checkpoint_sha
            )
            equivalent_numeric_spelling = tuple(
                "runtime.epochs=1.0"
                if item == "runtime.epochs=1"
                else "runtime.weight_decay=0.0"
                if item == "runtime.weight_decay=0"
                else item
                for item in common
            )
            numeric_verified = run(
                sys.executable,
                HELPER,
                "completion",
                "--action",
                "verify",
                *equivalent_numeric_spelling,
            )
            self.assertIn("verified", numeric_verified.stdout)
            changed_identity = run(
                sys.executable,
                HELPER,
                "completion",
                "--action",
                "verify",
                "--path",
                marker,
                "--kind",
                "evaluation",
                "--identity",
                "run_id=mvsec_jepa__seed1",
                "--artifact",
                f"checkpoint={checkpoint}",
                "--artifact",
                f"report={report}",
                "--report-field",
                "checkpoint.sha256=@sha256:checkpoint",
                "--report-field",
                "runtime.epochs=1",
                "--report-field",
                "runtime.seed=7",
                "--report-field",
                "runtime.weight_decay=0",
                check=False,
            )
            self.assertNotEqual(changed_identity.returncode, 0)
            self.assertIn("does not match", changed_identity.stderr)
            report.write_text(
                json.dumps(
                    {
                        "checkpoint": {"sha256": checkpoint_sha},
                        "runtime": {
                            "epochs": 1.0,
                            "seed": 7,
                            "weight_decay": 0.25,
                        },
                    }
                ),
                encoding="utf-8",
            )
            changed_report = run(
                sys.executable,
                HELPER,
                "completion",
                "--action",
                "verify",
                *common,
                check=False,
            )
            self.assertNotEqual(changed_report.returncode, 0)
            self.assertIn("report contract mismatch", changed_report.stderr)

    def test_comparison_set_identifier_is_ordered_and_deterministic(self) -> None:
        first = run(
            sys.executable,
            HELPER,
            "set-id",
            "--item",
            "a=/reports/a.json",
            "--item",
            "b=/reports/b.json",
        ).stdout.strip()
        repeated = run(
            sys.executable,
            HELPER,
            "set-id",
            "--item",
            "a=/reports/a.json",
            "--item",
            "b=/reports/b.json",
        ).stdout.strip()
        reversed_order = run(
            sys.executable,
            HELPER,
            "set-id",
            "--item",
            "b=/reports/b.json",
            "--item",
            "a=/reports/a.json",
        ).stdout.strip()
        self.assertRegex(first, r"^[0-9a-f]{16}$")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, reversed_order)

    def _render(
        self,
        output: Path,
        *,
        frame: str,
        cmax: str,
        reference: str,
        rate: str = "0",
        gamma: str = "1",
        straightening: str = "0",
    ) -> None:
        template = CMAX_TEMPLATE if cmax != "0" else JEPA_TEMPLATE
        run_id = (
            "mvsec_jepa_cmax_w0p05_ref_past__seed7"
            if cmax != "0"
            else "mvsec_jepa_frame_support_sigreg_off__seed7"
        )
        run(
            sys.executable,
            HELPER,
            "render",
            "--template",
            template,
            "--output",
            output,
            "--run-id",
            run_id,
            "--manifest",
            output.parent / "data/train.jsonl",
            "--run-output",
            output.parent / "run",
            "--seed",
            "7",
            "--cmax-weight",
            cmax,
            "--temporal-sigreg-weight",
            "0",
            "--frame-sigreg-weight",
            frame,
            "--rate-alignment-weight",
            rate,
            "--rate-alignment-gamma",
            gamma,
            "--rate-alignment-eps",
            "0.000001",
            "--rate-alignment-normalization",
            "per_clip_mean_supported_patch_rate",
            "--latent-straightening-weight",
            straightening,
            "--latent-straightening-eps",
            "0.000001",
            "--sequence-length",
            "8",
            "--cmax-reference-mode",
            reference,
            "--cmax-temporal-scales",
            "1,2",
            "--epochs",
            "20",
            "--warmup-epochs",
            "2",
            "--samples-per-epoch",
            "64",
            "--batch-size",
            "4",
            "--workers",
            "0",
            "--precision",
            "fp32",
        )

    def test_renderer_records_metadata_and_unregularized_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.yaml"
            self._render(output, frame="0", cmax="0", reference="both")
            text = output.read_text(encoding="utf-8")
            metadata_line = next(
                line for line in text.splitlines() if line.startswith("# ablation_metadata: ")
            )
            metadata = json.loads(metadata_line.partition(": ")[2])
            self.assertEqual(metadata["frame_sigreg_weight"], "0")
            self.assertTrue(metadata["allow_unregularized"])
            self.assertEqual(metadata["seed"], 7)
            self.assertEqual(
                metadata["exposure_policy"],
                "equal_supervised_frames_variable_updates",
            )
            self.assertIn("  allow_unregularized: true\n", text)
            self.assertIn("  frame_sigreg_weight: 0\n", text)
            self.assertIn("  epochs: 20\n", text)
            self.assertIn("  precision: fp32\n", text)

    def test_renderer_records_all_rate_and_straightening_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latent.yaml"
            self._render(
                output,
                frame="0.02",
                cmax="0",
                reference="both",
                rate="0.01",
                gamma="0.5",
                straightening="0.01",
            )
            text = output.read_text(encoding="utf-8")
            metadata_line = next(
                line for line in text.splitlines() if line.startswith("# ablation_metadata: ")
            )
            metadata = json.loads(metadata_line.partition(": ")[2])
            self.assertEqual(metadata["rate_alignment_weight"], "0.01")
            self.assertEqual(metadata["rate_alignment_gamma"], "0.5")
            self.assertEqual(metadata["rate_alignment_eps"], "0.000001")
            self.assertEqual(
                metadata["rate_alignment_normalization"],
                "per_clip_mean_supported_patch_rate",
            )
            self.assertEqual(metadata["latent_straightening_weight"], "0.01")
            self.assertEqual(metadata["latent_straightening_eps"], "0.000001")
            self.assertIn("  rate_alignment_weight: 0.01\n", text)
            self.assertIn("  rate_alignment_gamma: 0.5\n", text)
            self.assertIn("  latent_straightening_weight: 0.01\n", text)

    def test_renderer_is_idempotent_but_refuses_different_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cmax.yaml"
            self._render(output, frame="0.02", cmax="0.05", reference="past")
            first = output.read_bytes()
            self._render(output, frame="0.02", cmax="0.05", reference="past")
            self.assertEqual(output.read_bytes(), first)
            result = run(
                sys.executable,
                HELPER,
                "render",
                "--template",
                CMAX_TEMPLATE,
                "--output",
                output,
                "--run-id",
                "mvsec_jepa_cmax_w0p05_ref_past__seed7",
                "--manifest",
                output.parent / "different/train.jsonl",
                "--run-output",
                output.parent / "run",
                "--seed",
                "7",
                "--cmax-weight",
                "0.05",
                "--temporal-sigreg-weight",
                "0",
                "--frame-sigreg-weight",
                "0.02",
                "--rate-alignment-weight",
                "0",
                "--rate-alignment-gamma",
                "1",
                "--rate-alignment-eps",
                "0.000001",
                "--rate-alignment-normalization",
                "per_clip_mean_supported_patch_rate",
                "--latent-straightening-weight",
                "0",
                "--latent-straightening-eps",
                "0.000001",
                "--sequence-length",
                "8",
                "--cmax-reference-mode",
                "past",
                "--cmax-temporal-scales",
                "1,2",
                "--epochs",
                "20",
                "--warmup-epochs",
                "2",
                "--samples-per-epoch",
                "64",
                "--batch-size",
                "4",
                "--workers",
                "0",
                "--precision",
                "fp32",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(output.read_bytes(), first)


class MVSECAblationShellTests(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        run("bash", "-n", TRAIN)
        run("bash", "-n", EVAL)
        run("bash", "-n", VISUALIZE)

    def test_train_plan_has_no_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            result = run(
                "bash",
                TRAIN,
                "--action",
                "plan",
                "--suite",
                "core",
                "--seeds",
                "0",
                "--python-bin",
                sys.executable,
                "--output-root",
                output,
            )
            self.assertIn("runs=2", result.stdout)
            self.assertIn("CMax=0.05", result.stdout)
            self.assertFalse(output.exists())

    def test_prepare_writes_only_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            result = run(
                "bash",
                TRAIN,
                "--action",
                "prepare",
                "--suite",
                "frame",
                "--seeds",
                "2",
                "--python-bin",
                sys.executable,
                "--output-root",
                output,
            )
            self.assertIn("Prepared 2 immutable configs", result.stdout)
            configs = sorted((output / "configs").glob("*.yaml"))
            self.assertEqual(len(configs), 2)
            self.assertFalse((output / "pretrain").exists())

    def test_train_skip_complete_reuses_only_verified_full_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            manifest_root = data_root / "manifests"
            manifest_root.mkdir(parents=True)
            rows = []
            for camera in ("left", "right"):
                events = data_root / f"{camera}.h5"
                events.write_bytes(f"events-{camera}".encode())
                rows.append(
                    {
                        "sequence_id": f"mvsec__outdoor_day2__{camera}",
                        "source_recording_id": "mvsec__outdoor_day2",
                        "dataset": "mvsec",
                        "split": "train",
                        "camera": camera,
                        "height": 260,
                        "width": 346,
                        "source_height": 260,
                        "source_width": 346,
                        "coordinate_frame": "distorted",
                        "spatial_downsample": 1,
                        "path": str(events),
                    }
                )
            manifest = manifest_root / "train.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "artifacts"
            run(
                "bash",
                TRAIN,
                "--action",
                "prepare",
                "--suite",
                "frame",
                "--seeds",
                "0",
                "--data-root",
                data_root,
                "--train-manifest",
                manifest,
                "--python-bin",
                sys.executable,
                "--output-root",
                output,
            )
            run_ids = (
                "mvsec_jepa_frame_support_sigreg_off__seed0",
                "mvsec_jepa__seed0",
            )
            for run_id in run_ids:
                run_output = output / "pretrain" / run_id
                run_output.mkdir(parents=True)
                resolved = run_output / "resolved_config.yaml"
                checkpoint = run_output / "checkpoint-epoch0100.pt"
                metrics = run_output / "train.jsonl"
                resolved.write_text("resolved: true\n", encoding="utf-8")
                checkpoint.write_bytes(f"checkpoint-{run_id}".encode())
                metrics.write_text('{"epoch": 99}\n', encoding="utf-8")
                log = output / "logs" / "pretrain" / f"{run_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(f"completed {run_id}\n", encoding="utf-8")
                run(
                    sys.executable,
                    HELPER,
                    "completion",
                    "--action",
                    "record",
                    "--path",
                    run_output / ".ablation-complete.json",
                    "--kind",
                    "pretrain",
                    "--identity",
                    f"run_id={run_id}",
                    "--identity",
                    "seed=0",
                    "--identity",
                    "final_epoch=100",
                    "--identity",
                    "milestone_epochs=10,25,50,75,100",
                    "--identity",
                    "nproc_per_node=1",
                    "--identity",
                    "precision=fp16",
                    "--identity",
                    "exposure_policy=equal_supervised_frames_variable_updates",
                    "--artifact",
                    f"generated_config={output / 'configs' / (run_id + '.yaml')}",
                    "--artifact",
                    f"resolved_config={resolved}",
                    "--artifact",
                    f"checkpoint={checkpoint}",
                    "--artifact",
                    f"metrics={metrics}",
                    "--artifact",
                    f"log={log}",
                    "--artifact",
                    f"train_manifest={manifest}",
                )
            command = (
                "bash",
                TRAIN,
                "--action",
                "run",
                "--suite",
                "frame",
                "--seeds",
                "0",
                "--data-root",
                data_root,
                "--train-manifest",
                manifest,
                "--python-bin",
                sys.executable,
                "--output-root",
                output,
                "--skip-complete",
            )
            reused = run(*command)
            self.assertEqual(reused.stdout.count("Skipping verified completed run"), 2)
            self.assertNotIn("Starting ", reused.stdout)
            corrupted = output / "pretrain" / run_ids[-1] / "checkpoint-epoch0100.pt"
            corrupted.write_bytes(b"corrupted-checkpoint")
            rejected = run(*command, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("strict completion verification", rejected.stderr)
            self.assertNotIn("Starting ", rejected.stdout)

    def test_train_plan_carries_latent_controls_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            result = run(
                "bash",
                TRAIN,
                "--action",
                "plan",
                "--suite",
                "latent_cmax",
                "--seeds",
                "0",
                "--python-bin",
                sys.executable,
                "--output-root",
                output,
            )
            self.assertIn("runs=4", result.stdout)
            self.assertIn("RA=0.01 gamma=1 LS=0.01", result.stdout)
            self.assertIn("--rate-alignment-normalization", result.stdout)
            self.assertIn("equal_supervised_frames_variable_updates", result.stdout)
            self.assertFalse(output.exists())

    def test_context_plan_equalizes_frames_and_discloses_variable_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            result = run(
                "bash",
                TRAIN,
                "--action",
                "plan",
                "--suite",
                "context",
                "--seeds",
                "0",
                "--python-bin",
                sys.executable,
                "--output-root",
                output,
            )
            self.assertIn("T=4 clips=12500", result.stdout)
            self.assertIn("T=8 clips=6250", result.stdout)
            self.assertIn("T=16 clips=3125", result.stdout)
            self.assertIn("equal_supervised_frames_variable_updates", result.stdout)
            self.assertFalse(output.exists())

    def test_large_run_matrices_require_an_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            train = run(
                "bash",
                TRAIN,
                "--action",
                "run",
                "--suite",
                "all",
                "--seeds",
                "0",
                "--python-bin",
                sys.executable,
                "--output-root",
                root,
                check=False,
            )
            self.assertNotEqual(train.returncode, 0)
            self.assertIn("--allow-large-matrix", train.stderr)
            self.assertFalse(root.exists())
            evaluation = run(
                "bash",
                EVAL,
                "--action",
                "run",
                "--suite",
                "all",
                "--seeds",
                "0",
                "--tasks",
                "primary",
                "--python-bin",
                sys.executable,
                "--artifact-root",
                root,
                check=False,
            )
            self.assertNotEqual(evaluation.returncode, 0)
            self.assertIn("--allow-large-matrix", evaluation.stderr)
            self.assertFalse(root.exists())

    def test_eval_plan_builds_primary_jobs_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = run(
                "bash",
                EVAL,
                "--action",
                "plan",
                "--suite",
                "core",
                "--seeds",
                "0",
                "--tasks",
                "primary",
                "--protocol-suite",
                "primary",
                "--python-bin",
                sys.executable,
                "--artifact-root",
                root,
            )
            self.assertIn("jobs=5", result.stdout)
            self.assertIn("--head-init random", result.stdout)
            self.assertIn("cmax-direct", result.stdout)
            self.assertIn("--protocol-stage dev", result.stdout)
            self.assertIn("--eval-split train", result.stdout)
            self.assertIn("--history-steps 10", result.stdout)
            self.assertIn("downstream history: 10 fixed steps", result.stdout)
            self.assertIn("checkpoint-epoch0100.pt", result.stdout)
            self.assertFalse(root.exists())

    def test_eval_skip_complete_checks_report_and_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            manifests = data_root / "manifests"
            manifests.mkdir(parents=True)

            def write_manifest(
                name: str, recording: str, split: str, cameras: tuple[str, ...]
            ) -> Path:
                rows = []
                for camera in cameras:
                    events = data_root / f"{recording}-{camera}.h5"
                    events.write_bytes(f"events-{recording}-{camera}".encode())
                    rows.append(
                        {
                            "sequence_id": f"mvsec__{recording}__{camera}",
                            "source_recording_id": f"mvsec__{recording}",
                            "dataset": "mvsec",
                            "split": split,
                            "camera": camera,
                            "height": 260,
                            "width": 346,
                            "source_height": 260,
                            "source_width": 346,
                            "coordinate_frame": "distorted",
                            "spatial_downsample": 1,
                            "path": str(events),
                        }
                    )
                path = manifests / name
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                return path

            train_manifest = write_manifest(
                "train.jsonl", "outdoor_day2", "train", ("left", "right")
            )
            test_manifest = write_manifest(
                "test.jsonl", "outdoor_day1", "test", ("left",)
            )
            artifact_root = root / "artifacts"
            run_id = "mvsec_jepa_cmax_w0p05__seed0"
            checkpoint = (
                artifact_root
                / "pretrain"
                / run_id
                / "checkpoint-epoch0100.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixed-encoder-checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output = (
                artifact_root
                / "eval"
                / "final"
                / run_id
                / "epoch0100"
                / "cmax-direct"
                / "causal"
                / "native"
            )
            output.mkdir(parents=True)
            report = output / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "command": "cmax-eval",
                        "checkpoint": {
                            "path": str(checkpoint.resolve()),
                            "checkpoint_sha256": checkpoint_sha,
                        },
                        "protocol": {
                            "stage": "final",
                            "alignment": {"mode": "causal"},
                            "flow_rate": {"cli_value": "native"},
                            "event_history": {"history_steps": 10},
                        },
                        "head": {"initialization": "checkpoint_cmax"},
                        "evaluation": {
                            "manifest": str(test_manifest.resolve()),
                            "manifest_artifact": {
                                "sha256": hashlib.sha256(
                                    test_manifest.read_bytes()
                                ).hexdigest()
                            },
                        },
                        "runtime": {"precision": "fp32", "seed": 0},
                    }
                ),
                encoding="utf-8",
            )
            log = (
                artifact_root
                / "eval"
                / "logs"
                / "final"
                / "epoch0100"
                / f"{run_id}__cmax-direct__causal__native.log"
            )
            log.parent.mkdir(parents=True)
            log.write_text("completed cmax evaluation\n", encoding="utf-8")
            marker_args = (
                "--path",
                output / ".ablation-complete.json",
                "--kind",
                "evaluation",
                "--identity",
                f"run_id={run_id}",
                "--identity",
                "task=cmax-direct",
                "--identity",
                "stage=final",
                "--identity",
                "pretrain_epoch=100",
                "--identity",
                "alignment=causal",
                "--identity",
                "dt=native",
                "--identity",
                "probe_seed=0",
                "--identity",
                "history_steps=10",
                "--identity",
                "dev_fraction=0.2",
                "--identity",
                "dev_guard_ms=auto",
                "--identity",
                "batch_size=4",
                "--identity",
                "workers=4",
                "--identity",
                "device=cuda",
                "--identity",
                "precision=fp32",
                "--identity",
                "max_eval_samples=0",
                "--identity",
                "save_visualizations=0",
                "--identity",
                "visualization_max_events=200000",
                "--identity",
                "eval_split=test",
                "--identity",
                "head_initialization=checkpoint_cmax",
                "--artifact",
                f"checkpoint={checkpoint}",
                "--artifact",
                f"train_manifest={train_manifest}",
                "--artifact",
                f"eval_manifest={test_manifest}",
                "--artifact",
                f"report={report}",
                "--artifact",
                f"log={log}",
                "--report-field",
                "command=cmax-eval",
                "--report-field",
                f"checkpoint.path={checkpoint.resolve()}",
                "--report-field",
                "checkpoint.checkpoint_sha256=@sha256:checkpoint",
                "--report-field",
                "protocol.stage=final",
                "--report-field",
                "protocol.alignment.mode=causal",
                "--report-field",
                "protocol.flow_rate.cli_value=native",
                "--report-field",
                "protocol.event_history.history_steps=10",
                "--report-field",
                "head.initialization=checkpoint_cmax",
                "--report-field",
                f"evaluation.manifest={test_manifest.resolve()}",
                "--report-field",
                "evaluation.manifest_artifact.sha256=@sha256:eval_manifest",
                "--report-field",
                "runtime.precision=fp32",
                "--report-field",
                "runtime.seed=0",
            )
            run(
                sys.executable,
                HELPER,
                "completion",
                "--action",
                "record",
                *marker_args,
            )
            command = (
                "bash",
                EVAL,
                "--action",
                "run",
                "--stage",
                "final",
                "--selected-run-id",
                run_id,
                "--tasks",
                "cmax-direct",
                "--data-root",
                data_root,
                "--python-bin",
                sys.executable,
                "--artifact-root",
                artifact_root,
                "--skip-complete",
            )
            reused = run(*command)
            self.assertIn("Skipping verified completed evaluation", reused.stdout)
            self.assertNotIn("Starting ", reused.stdout)
            checkpoint.write_bytes(b"changed-encoder-checkpoint")
            rejected = run(*command, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("strict completion verification", rejected.stderr)
            self.assertNotIn("Starting ", rejected.stdout)

    def test_eval_full_protocol_grid_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = run(
                "bash",
                EVAL,
                "--action",
                "plan",
                "--suite",
                "core",
                "--seeds",
                "0",
                "--tasks",
                "all",
                "--protocol-suite",
                "all",
                "--python-bin",
                sys.executable,
                "--artifact-root",
                root,
            )
            self.assertIn("jobs=20", result.stdout)
            self.assertIn("alignment=f3_centered dt=dt1", result.stdout)
            self.assertIn("flow-cmax-init", result.stdout)
            self.assertFalse(root.exists())

    def test_eval_rejects_duplicate_task_before_any_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = run(
                "bash",
                EVAL,
                "--action",
                "plan",
                "--suite",
                "core",
                "--seeds",
                "0",
                "--tasks",
                "depth,depth",
                "--python-bin",
                sys.executable,
                "--artifact-root",
                root,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate task", result.stderr)
            self.assertFalse(root.exists())

    def test_final_eval_requires_one_preselected_run(self) -> None:
        result = run(
            "bash",
            EVAL,
            "--action",
            "plan",
            "--stage",
            "final",
            "--python-bin",
            sys.executable,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--selected-run-id is required", result.stderr)

    def test_final_visualization_rejects_a_suite_sweep(self) -> None:
        result = run(
            "bash",
            VISUALIZE,
            "--action",
            "plan",
            "--stage",
            "final",
            "--selected-run-id",
            "mvsec_jepa__seed0",
            "--suite",
            "core",
            "--python-bin",
            sys.executable,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not --suite/--seeds", result.stderr)

    def test_probe_seed_is_independent_and_in_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = run(
                "bash",
                EVAL,
                "--action",
                "plan",
                "--suite",
                "core",
                "--seeds",
                "0",
                "--probe-seeds",
                "7,11",
                "--tasks",
                "flow-random",
                "--python-bin",
                sys.executable,
                "--artifact-root",
                root,
            )
            self.assertIn("jobs=4", result.stdout)
            self.assertIn("probe_seed7", result.stdout)
            self.assertIn("probe_seed11", result.stdout)
            self.assertIn("--seed 11", result.stdout)

    def test_visualization_plan_uses_explicit_matched_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = run(
                "bash",
                VISUALIZE,
                "--action",
                "plan",
                "--mode",
                "compare",
                "--suite",
                "latent",
                "--seeds",
                "0,1",
                "--probe-seeds",
                "7",
                "--python-bin",
                sys.executable,
                "--artifact-root",
                root,
            )
            self.assertIn("matched report comparison: 8 reports", result.stdout)
            self.assertIn("--aggregate-seeds", result.stdout)
            self.assertIn("probe_seed7/report.json", result.stdout)
            self.assertIn("jepa_ra_w0p01_g1_ls_w0p01__seed1", result.stdout)
            self.assertRegex(
                result.stdout,
                r"/compare/dev/suite_latent/set_[0-9a-f]{16}/epoch0100/",
            )
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
