"""Lightweight visualization tests; requires NumPy but not PyTorch/HDF5."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "event_window_jepa"
    / "downstream"
    / "mvsec_visualize.py"
)
SPEC = importlib.util.spec_from_file_location("mvsec_visualize_standalone", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mvsec_visualize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mvsec_visualize)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flow_arrays() -> dict[str, np.ndarray]:
    height, width = 12, 16
    target = np.zeros((2, height, width), dtype=np.float32)
    target[0] = 1.0
    prediction = target.copy()
    prediction[1] = 0.5
    valid = np.ones((height, width), dtype=np.bool_)
    event_image = np.zeros((2, height, width), dtype=np.float32)
    event_image[0, 3:9, 5] = 3.0
    event_image[1, 3:9, 9] = 4.0
    count = 40
    return {
        "event_image": event_image,
        "target": target,
        "prediction": prediction,
        "valid": valid,
        "event_x": np.arange(count, dtype=np.int64) % width,
        "event_y": np.arange(count, dtype=np.int64) % height,
        "event_t_us": np.arange(1, count + 1, dtype=np.int64) * 1_000,
        "event_polarity": np.arange(count, dtype=np.int8) % 2,
    }


def _flow_report(aepe: float, protocol_name: str = "flow-v1") -> dict[str, object]:
    return {
        "command": "probe",
        "head": {
            "class": "RecurrentTokenFlowHead",
            "embed_dim": 384,
            "hidden_dim": 256,
            "head_depth": 2,
            "flow_scale": 0.01,
            "max_displacement_pixels_per_base_window": 32.0,
            "initialization": "random",
            "architecture_source": "canonical_random_mvsec_flow_probe_v1",
        },
        "protocol": {
            "name": protocol_name,
            "stage": "final",
            "target_timebase_contract": "native-v1",
            "coordinate_frame": "distorted",
            "model_canvas_height_width": [272, 352],
            "native_sensor_center_padding_yx": [6, 3],
            "alignment": {"mode": "causal"},
            "event_history": {"window_ms": 50.0, "history_steps": 10},
            "temporal_dev_split": None,
            "representation_pretraining_visibility_contract": {
                "protocol_class": "inductive_cross_recording_final_evaluation",
                "geometry_labels_visible_to_pretraining": False,
            },
            "flow_rate": {"protocol": "native"},
            "validity_mask": {"car_hood_start_row": 193},
            "minimum_valid_pixels_per_frame": 100,
            "evflownet_test_interval": {
                "applied": True,
                "start_inclusive_us": 222_400_000,
                "stop_exclusive_us": 240_400_000,
            },
        },
        "training": {
            "split": "train",
            "role": "train",
            "manifest_artifact": {"sha256": "d" * 64},
            "selection": {
                "targets": 3,
                "target_index_timestamp_sha256": "e" * 64,
                "target_artifacts": [
                    {
                        "format": "mvsec_gt_flow_npz_v1",
                        "bytes": 456,
                        "sha256": "f" * 64,
                    }
                ],
            },
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "loss": "masked_mean_endpoint_charbonnier_epsilon_1e-3",
            "flow_scaling": {"mode": "native"},
            "history": [
                {"epoch": 1, "mean_endpoint_loss": 2.0},
                {"epoch": 2, "mean_endpoint_loss": 1.0},
            ]
        },
        "evaluation": {
            "split": "test",
            "manifest_artifact": {"sha256": "a" * 64},
            "selection": {
                "target_index_timestamp_sha256": "b" * 64,
                "targets": 2,
                "target_artifacts": [
                    {
                        "format": "mvsec_gt_flow_npz_v1",
                        "bytes": 123,
                        "sha256": "c" * 64,
                    }
                ],
            },
            "metrics": {
                "sample_average": {
                    "AEPE": aepe,
                    "3PE_percent": 10.0 * aepe,
                }
            },
        },
        "runtime": {"precision": "bf16", "batch_size": 4},
    }


def _depth_report(abs_rel: float, *, precision: str = "bf16") -> dict[str, object]:
    split = {
        "role": "dev_eval",
        "recording": "outdoor_day2",
        "manifest_sha256": "1" * 64,
        "sequence_ids": ["mvsec__outdoor_day2__left"],
        "target_count": 2,
        "target_index_timestamp_sha256": "2" * 64,
        "alignment": "causal",
        "causal": True,
        "uses_future_events": False,
        "target_artifacts": [],
    }
    return {
        "protocol": {
            "name": "frozen_recurrent_raw_metric_depth_v1",
            "task": "mvsec_frozen_recurrent_absolute_depth_probe",
            "backbone": {"history_steps": 10},
            "head": {
                "order": "LayerNorm-Conv2d-GELU-Conv2d(1ch log-depth)-bilinear",
                "embed_dim": 384,
                "hidden_dim": 128,
                "patch_grid": [17, 22],
            },
            "geometry": {"coordinate_frame": "distorted"},
            "depth": {"validity_m": {"strict_minimum": 0.1, "strict_maximum": 80.0}},
            "training_policy": {
                "recording": "outdoor_day2",
                "fixed_epochs": 30,
                "batch_size": 4,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "precision": precision,
                "evaluation_used_for_model_selection": True,
                "model_selection": "late day2 dev metrics",
            },
            "target_selection": {"minimum_events_in_final_window": 1},
            "train_targets": {**split, "role": "dev_train"},
            "evaluation_targets": [split],
        },
        "training_history": [
            {"epoch": 1, "valid_log_depth_smooth_l1": 0.2},
            {"epoch": 2, "valid_log_depth_smooth_l1": 0.1},
        ],
        "metrics": {
            "dev": {
                "pixel_average": {
                    "AbsRel": abs_rel,
                    "RMSE": 1.0,
                    "SILog": 0.1,
                }
            }
        },
    }


class MVSECVisualizationTests(unittest.TestCase):
    def test_snapshot_is_deterministic_pickle_free_and_renderable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            arrays = _flow_arrays()
            metadata = {
                "sequence_id": "outdoor_day1__left",
                "target_index": 7,
                "label_timestamp_us": 50_000,
                "event_window_start_us": 0,
                "event_window_end_us": 50_000,
                "event_count_total": len(arrays["event_x"]),
                "event_count_stored": len(arrays["event_x"]),
            }
            event_keys = {
                name: arrays[name]
                for name in (
                    "event_x",
                    "event_y",
                    "event_t_us",
                    "event_polarity",
                )
            }
            first = directory / "first.npz"
            second = directory / "second.npz"
            identities = []
            for path in (first, second):
                identities.append(
                    mvsec_visualize.write_snapshot(
                        path,
                        kind="flow",
                        event_image=arrays["event_image"],
                        target=arrays["target"],
                        prediction=arrays["prediction"],
                        valid=arrays["valid"],
                        metadata=metadata,
                        events=event_keys,
                    )
                )
            self.assertEqual(_sha256(first), _sha256(second))
            loaded_metadata, loaded_arrays = mvsec_visualize.load_snapshot(first)
            self.assertEqual(loaded_metadata["schema"], mvsec_visualize.SNAPSHOT_SCHEMA)
            self.assertTrue(np.array_equal(loaded_arrays["target"], arrays["target"]))

            report = mvsec_visualize.render_snapshot(
                first,
                directory / "rendered",
                expected_bytes=identities[0]["bytes"],
                expected_sha256=identities[0]["sha256"],
            )
            self.assertAlmostEqual(report["metrics"]["AEPE"], 0.5)
            self.assertIn("cmax_visual_diagnostic", report["metrics"])
            for filename in (
                "events.png",
                "target_flow.png",
                "prediction_flow.png",
                "epe.png",
                "cmax_before.png",
                "cmax_after.png",
            ):
                payload = (directory / "rendered" / filename).read_bytes()
                self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                mvsec_visualize.load_snapshot(
                    first,
                    expected_bytes=identities[0]["bytes"],
                    expected_sha256="0" * 64,
                )

    def test_depth_rejects_nonpositive_valid_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            valid = np.ones((3, 4), dtype=np.bool_)
            target = np.ones((3, 4), dtype=np.float32)
            prediction = target.copy()
            prediction[0, 0] = 0.0
            with self.assertRaisesRegex(ValueError, "must be positive"):
                mvsec_visualize.write_snapshot(
                    directory / "invalid.npz",
                    kind="depth",
                    event_image=np.zeros((2, 3, 4), dtype=np.float32),
                    target=target,
                    prediction=prediction,
                    valid=valid,
                    metadata={"sequence_id": "synthetic"},
                )

    def test_snapshot_decompressed_size_limit_precedes_array_loading(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            height, width = 384, 384
            valid = np.ones((height, width), dtype=np.bool_)
            path = directory / "compressed.npz"
            mvsec_visualize.write_snapshot(
                path,
                kind="depth",
                event_image=np.zeros((2, height, width), dtype=np.float32),
                target=np.ones((height, width), dtype=np.float32),
                prediction=np.ones((height, width), dtype=np.float32),
                valid=valid,
                metadata={"sequence_id": "synthetic"},
            )
            self.assertLess(path.stat().st_size, 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "decompression limit"):
                mvsec_visualize.load_snapshot(path, maximum_mib=1)

    def test_report_comparison_checks_contract_and_aggregates_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths = []
            for index, aepe in enumerate((1.0, 3.0)):
                path = directory / f"seed{index}.json"
                path.write_text(json.dumps(_flow_report(aepe)), encoding="utf-8")
                paths.append(path)
            result = mvsec_visualize.compare_reports(
                [
                    ("jepa_cmax__seed0", paths[0]),
                    ("jepa_cmax__seed1", paths[1]),
                ],
                directory / "comparison",
                aggregate_seeds=True,
            )
            self.assertEqual(result["compatibility"]["status"], "matched")
            aggregate = result["seed_aggregation"]["metrics"]
            metric = "evaluation.metrics.sample_average.AEPE"
            self.assertEqual(aggregate[metric]["jepa_cmax"]["mean"], 2.0)
            self.assertEqual(aggregate[metric]["jepa_cmax"]["std_population"], 1.0)
            self.assertTrue((directory / "comparison" / "metrics.png").is_file())
            self.assertTrue(
                (
                    directory
                    / "comparison"
                    / "metric-00-evaluation.metrics.sample_average.AEPE.png"
                ).is_file()
            )
            self.assertTrue(
                (directory / "comparison" / "curve-00-mean_endpoint_loss.png").is_file()
            )

            incompatible = directory / "incompatible.json"
            incompatible.write_text(
                json.dumps(_flow_report(2.0, protocol_name="different")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                mvsec_visualize.compare_reports(
                    [("a", paths[0]), ("b", incompatible)],
                    directory / "rejected",
                )

            different_training_targets = _flow_report(2.0)
            training = different_training_targets["training"]
            self.assertIsInstance(training, dict)
            selection = training["selection"]
            self.assertIsInstance(selection, dict)
            selection["target_index_timestamp_sha256"] = "0" * 64
            different_training = directory / "different-training.json"
            different_training.write_text(
                json.dumps(different_training_targets),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                mvsec_visualize.compare_reports(
                    [("a", paths[0]), ("b", different_training)],
                    directory / "rejected-training-targets",
                )

    def test_event_extraction_is_bounded_and_uses_center_transform(self) -> None:
        window = SimpleNamespace(
            x=np.asarray([0, 1, 2, 3, 4], dtype=np.int16),
            y=np.asarray([0, 1, 2, 3, 4], dtype=np.int16),
            t_us=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
            polarity=np.asarray([0, 1, 0, 1, 0], dtype=np.int8),
        )

        class Store:
            def slice(self, sequence_id: str, t_end_us: int, duration_us: int) -> object:
                self.arguments = (sequence_id, t_end_us, duration_us)
                return window

        store = Store()
        dataset = SimpleNamespace(
            references=(SimpleNamespace(source_index=0),),
            sources=(SimpleNamespace(sequence_id="sequence"),),
            store=store,
            crop=SimpleNamespace(x0=-3, y0=-6, output_width=16, output_height=12),
        )
        events, metadata = mvsec_visualize.extract_snapshot_events(
            dataset,
            0,
            t_end_us=10,
            duration_us=10,
            maximum_events=3,
        )
        self.assertEqual(store.arguments, ("sequence", 10, 10))
        self.assertEqual(events["event_x"].tolist(), [3, 5, 7])
        self.assertEqual(events["event_y"].tolist(), [6, 8, 10])
        self.assertEqual(metadata["event_count_total"], 5)
        self.assertEqual(metadata["event_count_stored"], 3)

    def test_report_comparison_aggregates_probe_then_encoder_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            runs = []
            values = {
                "jepa": ((1.0, 3.0), (5.0, 7.0)),
                "jepa_cmax": ((2.0, 4.0), (6.0, 8.0)),
            }
            for condition, encoder_values in values.items():
                for encoder_seed, probe_values in enumerate(encoder_values):
                    for probe_seed, aepe in enumerate(probe_values):
                        path = directory / (
                            f"{condition}-encoder{encoder_seed}-probe{probe_seed}.json"
                        )
                        path.write_text(
                            json.dumps(_flow_report(aepe)), encoding="utf-8"
                        )
                        label = (
                            f"{condition}__encoder_seed{encoder_seed}"
                            f"__probe_seed{probe_seed}"
                        )
                        runs.append((label, path))
            result = mvsec_visualize.compare_reports(
                runs,
                directory / "hierarchical",
                aggregate_seeds=True,
            )
            metric = "evaluation.metrics.sample_average.AEPE"
            aggregate = result["seed_aggregation"]["metrics"][metric]
            self.assertEqual(aggregate["jepa"]["mean"], 4.0)
            self.assertEqual(aggregate["jepa"]["std_population"], 2.0)
            self.assertEqual(aggregate["jepa"]["encoder_count"], 2)
            self.assertEqual(aggregate["jepa"]["probe_run_count"], 4)
            self.assertEqual(aggregate["jepa_cmax"]["mean"], 5.0)
            self.assertEqual(
                aggregate["jepa"]["aggregation_order"],
                "probe_mean_then_encoder_mean",
            )

    def test_depth_comparison_recognizes_dev_metrics_and_precision_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = directory / "first-depth.json"
            second = directory / "second-depth.json"
            first.write_text(json.dumps(_depth_report(0.2)), encoding="utf-8")
            second.write_text(json.dumps(_depth_report(0.3)), encoding="utf-8")
            result = mvsec_visualize.compare_reports(
                [("jepa__seed0", first), ("jepa__seed1", second)],
                directory / "depth-comparison",
                aggregate_seeds=True,
            )
            self.assertIn("metrics.dev.pixel_average.SILog", result["metrics"])
            self.assertIn(
                "valid_log_depth_smooth_l1", result["curves"]
            )

            incompatible = directory / "fp32-depth.json"
            incompatible.write_text(
                json.dumps(_depth_report(0.3, precision="fp32")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                mvsec_visualize.compare_reports(
                    [("a", first), ("b", incompatible)],
                    directory / "rejected-depth-precision",
                )


if __name__ == "__main__":
    unittest.main()
