from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_window_jepa.evaluation.window_sweep import summarize_jsonl


def write_rows(path: Path, sample_ids: tuple[str, str, str]) -> None:
    rows = []
    for seed, sample_set_id in enumerate(sample_ids):
        for window, value, group in (
            (5, 0.3 + seed * 0.01, "unseen_extrapolation"),
            (40, 0.5 + seed * 0.01, "seen"),
        ):
            rows.append(
                {
                    "method": "window_jepa",
                    "metric": "mAP",
                    "seed": seed,
                    "window_ms": window,
                    "value": value,
                    "higher_is_better": True,
                    "sample_set_id": sample_set_id,
                    "window_group": group,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summarizer_reports_unseen_and_three_seed_statistics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_rows(path, ("same", "same", "same"))
    result = summarize_jsonl(path, reference_window_ms=40, minimum_seeds=3)
    assert len(result["per_seed"]) == 3
    aggregate = result["across_seeds"][0]
    assert aggregate["seeds"] == [0, 1, 2]
    assert "unseen_all" in aggregate["summaries"]


def test_summarizer_rejects_different_anchor_populations(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_rows(path, ("a", "b", "c"))
    with pytest.raises(ValueError, match="mismatched"):
        summarize_jsonl(path, reference_window_ms=40, minimum_seeds=3)

