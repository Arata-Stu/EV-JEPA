from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from event_window_jepa.data.event_store import H5EventStore  # noqa: E402
from event_window_jepa.preprocessing.common import (  # noqa: E402
    EventSourceMetadata,
    PreprocessOptions,
    hdf5_filter_ids,
    preprocess_sequence,
    write_manifest,
)


class _SyntheticSource:
    def __init__(
        self,
        source_path: Path,
        *,
        fail_after_chunks: int | None = None,
    ) -> None:
        self.arrays = {
            "x": np.array([0, 1, 2, 3, 4], dtype=np.uint16),
            "y": np.array([1, 1, 1, 1, 1], dtype=np.uint16),
            "t_us": np.array([10_000, 11_000, 11_000, 11_500, 12_000]),
            "polarity": np.array([0, 1, 0, 1, 1], dtype=np.uint8),
        }
        self.metadata = EventSourceMetadata(
            sequence_id="synthetic__recording__left",
            dataset="synthetic",
            source_path=source_path,
            camera="left",
            width=8,
            height=4,
            event_count=5,
            first_timestamp_us=10_000,
            last_timestamp_us=12_000,
            coordinate_frame="distorted",
        )
        self.fail_after_chunks = fail_after_chunks
        self.start_events: list[int] = []
        self.closed = False

    def iter_event_chunks(
        self, chunk_events: int, start_event: int = 0
    ) -> Iterator[Mapping[str, np.ndarray]]:
        self.start_events.append(start_event)
        emitted = 0
        for start in range(start_event, self.metadata.event_count, chunk_events):
            if self.fail_after_chunks is not None and emitted >= self.fail_after_chunks:
                raise RuntimeError("injected source interruption")
            stop = min(start + chunk_events, self.metadata.event_count)
            yield {name: values[start:stop] for name, values in self.arrays.items()}
            emitted += 1

    def close(self) -> None:
        self.closed = True


def _options() -> PreprocessOptions:
    return PreprocessOptions(
        read_chunk_events=2,
        hdf5_chunk_events=4,
        zstd_level=3,
        index_step_us=1_000,
    )


def test_zstd_round_trip_and_causal_boundary(tmp_path: Path) -> None:
    raw = tmp_path / "source.events"
    raw.write_bytes(b"stable source fingerprint")
    output = tmp_path / "sequence.h5"
    record = preprocess_sequence(
        _SyntheticSource(raw), output, split="train", options=_options()
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)

    with h5py.File(output, "r") as handle:
        for dataset in (
            handle["events/x"],
            handle["events/y"],
            handle["events/t_us"],
            handle["events/polarity"],
            handle["index/ms_to_event_idx"],
        ):
            assert {2, 3, 32015}.issubset(set(hdf5_filter_ids(dataset)))
        assert handle["index/ms_to_event_idx"][:].tolist() == [0, 1, 4, 5]

    store = H5EventStore(manifest)
    window = store.slice("synthetic__recording__left", t_end_us=2_000, duration_us=1_000)
    assert window.t_us.tolist() == [1_500, 2_000]
    assert window.x.tolist() == [3, 4]
    store.close()


def test_interrupted_sequence_resumes_from_committed_source_event(tmp_path: Path) -> None:
    raw = tmp_path / "source.events"
    raw.write_bytes(b"stable source fingerprint")
    output = tmp_path / "sequence.h5"

    with pytest.raises(RuntimeError, match="injected"):
        preprocess_sequence(
            _SyntheticSource(raw, fail_after_chunks=1),
            output,
            split="train",
            options=_options(),
        )
    assert not output.exists()
    assert (tmp_path / ".sequence.h5.partial").exists()

    resumed = _SyntheticSource(raw)
    record = preprocess_sequence(resumed, output, split="train", options=_options())
    assert resumed.start_events == [2]
    assert record["event_count"] == 5
    assert output.exists()
    assert not (tmp_path / ".sequence.h5.partial").exists()
