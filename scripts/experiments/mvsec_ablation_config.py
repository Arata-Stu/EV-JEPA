#!/usr/bin/env python3
"""Build deterministic MVSEC ablation matrices and derived pretrain configs.

This helper intentionally uses only the Python standard library.  It does not
parse arbitrary YAML: it patches a small, explicitly checked set of scalar
keys in the two repository-owned MVSEC templates.  A changed or ambiguous
template therefore fails closed instead of silently producing a wrong run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


_SCALAR_LINE = re.compile(r"^(?P<indent>[ ]+)(?P<key>[A-Za-z0-9_]+):(?P<tail>.*)$")
_SECTION_LINE = re.compile(r"^(?P<section>[A-Za-z0-9_]+):[ ]*(?:#.*)?$")
_RATE_NORMALIZATION = "per_clip_mean_supported_patch_rate"
_EXPOSURE_POLICY = "equal_supervised_frames_variable_updates"


@dataclass(frozen=True)
class Condition:
    name: str
    cmax_weight: Decimal
    temporal_sigreg_weight: Decimal
    frame_sigreg_weight: Decimal = Decimal("0.02")
    sequence_length: int = 8
    cmax_reference_mode: str = "both"
    cmax_temporal_scales: tuple[int, ...] = (1, 2, 4)
    rate_alignment_weight: Decimal = Decimal("0")
    rate_alignment_gamma: Decimal = Decimal("1")
    rate_alignment_eps: Decimal = Decimal("0.000001")
    rate_alignment_normalization: str = _RATE_NORMALIZATION
    latent_straightening_weight: Decimal = Decimal("0")
    latent_straightening_eps: Decimal = Decimal("0.000001")


BASELINE = Condition("jepa", Decimal("0"), Decimal("0"))
CORE_CMAX = Condition("jepa_cmax_w0p05", Decimal("0.05"), Decimal("0"))
TEMPORAL_002 = Condition("jepa_tsig_w0p02", Decimal("0"), Decimal("0.02"))
RATE_001 = Condition(
    "jepa_ra_w0p001_g1",
    Decimal("0"),
    Decimal("0"),
    rate_alignment_weight=Decimal("0.001"),
)
RATE_01 = Condition(
    "jepa_ra_w0p01_g1",
    Decimal("0"),
    Decimal("0"),
    rate_alignment_weight=Decimal("0.01"),
)
RATE_05 = Condition(
    "jepa_ra_w0p05_g1",
    Decimal("0"),
    Decimal("0"),
    rate_alignment_weight=Decimal("0.05"),
)
STRAIGHTENING_001 = Condition(
    "jepa_ls_w0p001",
    Decimal("0"),
    Decimal("0"),
    latent_straightening_weight=Decimal("0.001"),
)
STRAIGHTENING_01 = Condition(
    "jepa_ls_w0p01",
    Decimal("0"),
    Decimal("0"),
    latent_straightening_weight=Decimal("0.01"),
)
STRAIGHTENING_05 = Condition(
    "jepa_ls_w0p05",
    Decimal("0"),
    Decimal("0"),
    latent_straightening_weight=Decimal("0.05"),
)
RATE_AND_STRAIGHTENING = Condition(
    "jepa_ra_w0p01_g1_ls_w0p01",
    Decimal("0"),
    Decimal("0"),
    rate_alignment_weight=Decimal("0.01"),
    latent_straightening_weight=Decimal("0.01"),
)
CMAX_RATE_AND_STRAIGHTENING = Condition(
    "jepa_cmax_w0p05_ra_w0p01_g1_ls_w0p01",
    Decimal("0.05"),
    Decimal("0"),
    rate_alignment_weight=Decimal("0.01"),
    latent_straightening_weight=Decimal("0.01"),
)

SUITES: dict[str, tuple[Condition, ...]] = {
    # Primary one-axis comparison.
    "core": (BASELINE, CORE_CMAX),
    # Temporal SIGReg is changed while CMax and recurrent context stay off/fixed.
    "temporal_sigreg": (
        BASELINE,
        Condition("jepa_tsig_w0p01", Decimal("0"), Decimal("0.01")),
        TEMPORAL_002,
        Condition("jepa_tsig_w0p05", Decimal("0"), Decimal("0.05")),
    ),
    # CMax weight is changed while Temporal SIGReg and context stay fixed.
    "cmax": (
        BASELINE,
        Condition("jepa_cmax_w0p01", Decimal("0.01"), Decimal("0")),
        CORE_CMAX,
        Condition("jepa_cmax_w0p10", Decimal("0.10"), Decimal("0")),
    ),
    # Loss-bearing recurrent context is the only changed axis (burn-in remains 2).
    "context": (
        Condition("jepa_ctx4", Decimal("0"), Decimal("0"), sequence_length=4),
        BASELINE,
        Condition("jepa_ctx16", Decimal("0"), Decimal("0"), sequence_length=16),
    ),
    # This is an explicitly named interaction experiment, separate from main effects.
    "interaction": (
        BASELINE,
        TEMPORAL_002,
        CORE_CMAX,
        Condition(
            "jepa_cmax_w0p05_tsig_w0p02", Decimal("0.05"), Decimal("0.02")
        ),
    ),
    # CMax reference direction is changed with every other value fixed.
    "reference": (
        Condition(
            "jepa_cmax_w0p05_ref_past",
            Decimal("0.05"),
            Decimal("0"),
            cmax_reference_mode="past",
        ),
        Condition(
            "jepa_cmax_w0p05_ref_future",
            Decimal("0.05"),
            Decimal("0"),
            cmax_reference_mode="future",
        ),
        CORE_CMAX,
    ),
    # CMax temporal partitions are changed with every other value fixed.
    "scales": (
        Condition(
            "jepa_cmax_w0p05_scales1",
            Decimal("0.05"),
            Decimal("0"),
            cmax_temporal_scales=(1,),
        ),
        Condition(
            "jepa_cmax_w0p05_scales1_2",
            Decimal("0.05"),
            Decimal("0"),
            cmax_temporal_scales=(1, 2),
        ),
        CORE_CMAX,
    ),
    # Collapse-control removal is explicit and never mixed with CMax or Temporal SIGReg.
    "frame": (
        Condition(
            "jepa_frame_support_sigreg_off",
            Decimal("0"),
            Decimal("0"),
            frame_sigreg_weight=Decimal("0"),
        ),
        BASELINE,
    ),
    # Neural Events-inspired Rate Alignment weight, with gamma fixed to one.
    "rate_alignment": (BASELINE, RATE_001, RATE_01, RATE_05),
    # Gamma sensitivity conditional on the selected RA weight.
    "rate_gamma": (
        Condition(
            "jepa_ra_w0p01_g0p5",
            Decimal("0"),
            Decimal("0"),
            rate_alignment_weight=Decimal("0.01"),
            rate_alignment_gamma=Decimal("0.5"),
        ),
        RATE_01,
        Condition(
            "jepa_ra_w0p01_g2",
            Decimal("0"),
            Decimal("0"),
            rate_alignment_weight=Decimal("0.01"),
            rate_alignment_gamma=Decimal("2"),
        ),
    ),
    # Neural Events-inspired Latent Straightening weight.
    "straightening": (
        BASELINE,
        STRAIGHTENING_001,
        STRAIGHTENING_01,
        STRAIGHTENING_05,
    ),
    # Full RA/LS main-effect and interaction 2x2.
    "latent": (
        BASELINE,
        RATE_01,
        STRAIGHTENING_01,
        RATE_AND_STRAIGHTENING,
    ),
    # Selected RA+LS regularizer crossed with the primary CMax setting.
    "latent_cmax": (
        BASELINE,
        RATE_AND_STRAIGHTENING,
        CORE_CMAX,
        CMAX_RATE_AND_STRAIGHTENING,
    ),
}


def _all_conditions() -> tuple[Condition, ...]:
    ordered: list[Condition] = []
    seen: set[Condition] = set()
    for suite in (
        "core",
        "temporal_sigreg",
        "cmax",
        "context",
        "interaction",
        "reference",
        "scales",
        "frame",
        "rate_alignment",
        "rate_gamma",
        "straightening",
        "latent",
        "latent_cmax",
    ):
        for condition in SUITES[suite]:
            if condition not in seen:
                seen.add(condition)
                ordered.append(condition)
    return tuple(ordered)


SUITES["all"] = _all_conditions()


def _parse_seeds(raw: str) -> tuple[int, ...]:
    if not raw or raw.startswith(",") or raw.endswith(","):
        raise ValueError("--seeds must be a comma-separated list of integers")
    result: list[int] = []
    for item in raw.split(","):
        if not re.fullmatch(r"0|[1-9][0-9]*", item):
            raise ValueError(f"invalid non-negative seed: {item!r}")
        value = int(item)
        if value in result:
            raise ValueError(f"duplicate seed: {value}")
        result.append(value)
    return tuple(result)


def matrix_rows(suite: str, seeds: str, smoke: bool = False) -> list[tuple[str, ...]]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
    rows: list[tuple[str, ...]] = []
    for condition in SUITES[suite]:
        for seed in _parse_seeds(seeds):
            suffix = "__smoke" if smoke else ""
            run_id = f"mvsec_{condition.name}__seed{seed}{suffix}"
            rows.append(
                (
                    run_id,
                    condition.name,
                    str(seed),
                    _decimal_text(condition.cmax_weight),
                    _decimal_text(condition.temporal_sigreg_weight),
                    _decimal_text(condition.frame_sigreg_weight),
                    str(condition.sequence_length),
                    condition.cmax_reference_mode,
                    ",".join(str(scale) for scale in condition.cmax_temporal_scales),
                    _decimal_text(condition.rate_alignment_weight),
                    _decimal_text(condition.rate_alignment_gamma),
                    _decimal_text(condition.rate_alignment_eps),
                    condition.rate_alignment_normalization,
                    _decimal_text(condition.latent_straightening_weight),
                    _decimal_text(condition.latent_straightening_eps),
                )
            )
    if len({row[0] for row in rows}) != len(rows):
        raise AssertionError("matrix generated duplicate run identifiers")
    return rows


def _canonical_decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError(f"decimal must be finite: {value}")
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite() or value < 0:
        raise ValueError(f"weight must be finite and non-negative: {value}")
    return _canonical_decimal_text(value)


def _section_bounds(lines: list[str], section: str) -> tuple[int, int]:
    starts = [
        index
        for index, line in enumerate(lines)
        if (match := _SECTION_LINE.fullmatch(line.rstrip("\n")))
        and match.group("section") == section
    ]
    if len(starts) != 1:
        raise ValueError(
            f"template must contain exactly one top-level {section!r} section; "
            f"found {len(starts)}"
        )
    start = starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _SECTION_LINE.fullmatch(lines[index].rstrip("\n")):
            end = index
            break
    return start, end


def _has_section(lines: list[str], section: str) -> bool:
    return any(
        (match := _SECTION_LINE.fullmatch(line.rstrip("\n")))
        and match.group("section") == section
        for line in lines
    )


def _replace_scalar(lines: list[str], section: str, key: str, value: str) -> None:
    start, end = _section_bounds(lines, section)
    matches: list[int] = []
    for index in range(start + 1, end):
        match = _SCALAR_LINE.match(lines[index].rstrip("\n"))
        if match and match.group("key") == key:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(
            f"template must contain exactly one {section}.{key}; found {len(matches)}"
        )
    index = matches[0]
    match = _SCALAR_LINE.match(lines[index].rstrip("\n"))
    assert match is not None
    tail = match.group("tail")
    comment = ""
    comment_match = re.search(r"(?P<spacing>[ ]+)#(?P<comment>.*)$", tail)
    if comment_match:
        comment = (
            comment_match.group("spacing")
            + "#"
            + comment_match.group("comment")
        )
    lines[index] = f"{match.group('indent')}{key}: {value}{comment}\n"


def _replace_or_insert_scalar(
    lines: list[str], section: str, key: str, value: str, *, after: str
) -> None:
    start, end = _section_bounds(lines, section)
    key_matches = []
    after_matches = []
    for index in range(start + 1, end):
        match = _SCALAR_LINE.match(lines[index].rstrip("\n"))
        if not match:
            continue
        if match.group("key") == key:
            key_matches.append(index)
        if match.group("key") == after:
            after_matches.append(index)
    if len(key_matches) > 1:
        raise ValueError(f"template contains duplicate {section}.{key}")
    if key_matches:
        _replace_scalar(lines, section, key, value)
        return
    if len(after_matches) != 1:
        raise ValueError(
            f"cannot insert {section}.{key}: expected one {section}.{after}"
        )
    indentation = _SCALAR_LINE.match(lines[after_matches[0]].rstrip("\n"))
    assert indentation is not None
    lines.insert(
        after_matches[0] + 1,
        f"{indentation.group('indent')}{key}: {value}\n",
    )


def _yaml_string(value: str) -> str:
    # JSON strings are valid YAML double-quoted scalars.
    return json.dumps(value, ensure_ascii=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _key_value_map(
    entries: Sequence[str], *, label: str, key_pattern: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or re.fullmatch(key_pattern, key) is None:
            raise ValueError(f"invalid {label}: {entry!r}")
        if key in result:
            raise ValueError(f"duplicate {label} key: {key!r}")
        result[key] = value
    return result


def _stable_file_report(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"completion artifact is not a file: {resolved}")
    before = resolved.stat()
    if before.st_size <= 0:
        raise ValueError(f"completion artifact is empty: {resolved}")
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    sha256 = _sha256_path(resolved)
    after = resolved.stat()
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_after != identity_before:
        raise ValueError(f"completion artifact changed while hashing: {resolved}")
    return {"path": str(resolved), "bytes": before.st_size, "sha256": sha256}


def _json_value_at(payload: object, dotted_path: str) -> object:
    current = payload
    for component in dotted_path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                raise ValueError(f"report field is missing: {dotted_path}")
            current = current[component]
        elif isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", component):
            index = int(component)
            if index >= len(current):
                raise ValueError(f"report field is missing: {dotted_path}")
            current = current[index]
        else:
            raise ValueError(f"report field is missing: {dotted_path}")
    return current


def _contract_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, complex):
        return _canonical_decimal_text(Decimal(str(value)))
    raise ValueError(f"report contract field is not scalar: {value!r}")


def _contract_expected(expected: str, actual_value: object) -> str:
    if isinstance(actual_value, (int, float)) and not isinstance(
        actual_value, (bool, complex)
    ):
        try:
            return _canonical_decimal_text(Decimal(expected))
        except InvalidOperation:
            return expected
    return expected


def completion_payload(args: argparse.Namespace) -> dict[str, object]:
    identities = _key_value_map(
        args.identity,
        label="completion identity",
        key_pattern=r"[a-z][a-z0-9_]*",
    )
    artifact_paths = _key_value_map(
        args.artifact,
        label="completion artifact",
        key_pattern=r"[a-z][a-z0-9_]*",
    )
    if not identities or not artifact_paths:
        raise ValueError("completion identity and artifacts cannot be empty")
    artifacts = {
        role: _stable_file_report(Path(value))
        for role, value in sorted(artifact_paths.items())
    }
    report_contract = _key_value_map(
        args.report_field,
        label="report contract",
        key_pattern=r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*",
    )
    resolved_contract: dict[str, str] = {}
    if report_contract:
        if args.report_role not in artifacts:
            raise ValueError(
                f"report artifact role is missing: {args.report_role!r}"
            )
        report_path = Path(str(artifacts[args.report_role]["path"]))
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"completion report is invalid JSON: {report_path}") from error
        for dotted_path, expected in sorted(report_contract.items()):
            if expected.startswith("@sha256:"):
                artifact_role = expected.removeprefix("@sha256:")
                if artifact_role not in artifacts:
                    raise ValueError(
                        f"report contract references unknown artifact: {artifact_role!r}"
                    )
                expected = str(artifacts[artifact_role]["sha256"])
            actual_value = _json_value_at(report, dotted_path)
            expected = _contract_expected(expected, actual_value)
            actual = _contract_scalar(actual_value)
            if actual != expected:
                raise ValueError(
                    f"report contract mismatch for {dotted_path}: "
                    f"expected {expected!r}, found {actual!r}"
                )
            resolved_contract[dotted_path] = expected
    return {
        "schema": "event-window-jepa-runner-completion-v1",
        "kind": args.kind,
        "identity": dict(sorted(identities.items())),
        "artifacts": artifacts,
        "report_role": args.report_role if report_contract else None,
        "report_contract": resolved_contract,
    }


def handle_completion(args: argparse.Namespace) -> str:
    marker = args.path.expanduser().resolve(strict=False)
    payload = completion_payload(args)
    artifact_paths = {
        str(artifact["path"])
        for artifact in payload["artifacts"].values()
        if isinstance(artifact, Mapping)
    }
    if str(marker) in artifact_paths:
        raise ValueError("completion marker cannot also be a tracked artifact")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if args.action == "record":
        status = _write_exclusive_or_verify(marker, encoded, check_only=False)
        return f"{status}\t{marker}"
    marker = marker.resolve(strict=True)
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"completion marker is invalid JSON: {marker}") from error
    if recorded != payload:
        raise ValueError(
            f"completion marker does not match the requested run identity: {marker}"
        )
    return f"verified\t{marker}"


def comparison_set_identifier(items: Sequence[str]) -> str:
    if not items:
        raise ValueError("comparison set cannot be empty")
    if len(items) != len(set(items)):
        raise ValueError("comparison set contains duplicate items")
    payload = json.dumps(list(items), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_manifest(args: argparse.Namespace) -> dict[str, object]:
    path = args.path.expanduser().resolve(strict=True)
    expected_cameras = set(args.expected_cameras.split(","))
    if not expected_cameras or "" in expected_cameras:
        raise ValueError("--expected-cameras must be a non-empty CSV")
    sequence_ids: set[str] = set()
    cameras: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object")
            sequence_id = str(row.get("sequence_id", ""))
            if not sequence_id or sequence_id in sequence_ids:
                raise ValueError(f"{path}:{line_number} has a missing/duplicate sequence_id")
            sequence_ids.add(sequence_id)
            expected_source = f"mvsec__{args.expected_recording}"
            source_recording_id = str(row.get("source_recording_id", ""))
            if source_recording_id != expected_source:
                raise ValueError(
                    f"{path}:{line_number} has source_recording_id="
                    f"{source_recording_id!r}, expected {expected_source!r}"
                )
            if str(row.get("dataset", "")) != "mvsec":
                raise ValueError(f"{path}:{line_number} is not an MVSEC row")
            if str(row.get("split", "")) != args.expected_split:
                raise ValueError(
                    f"{path}:{line_number} is not split={args.expected_split!r}"
                )
            camera = str(row.get("camera", ""))
            if camera not in expected_cameras:
                raise ValueError(
                    f"{path}:{line_number} has unexpected camera {camera!r}"
                )
            expected_sequence = f"{expected_source}__{camera}"
            if sequence_id != expected_sequence:
                raise ValueError(
                    f"{path}:{line_number} has sequence_id={sequence_id!r}, "
                    f"expected {expected_sequence!r}"
                )
            cameras.add(camera)
            geometry = (int(row.get("height", -1)), int(row.get("width", -1)))
            source_geometry = (
                int(row.get("source_height", -1)),
                int(row.get("source_width", -1)),
            )
            if geometry != (260, 346) or source_geometry != (260, 346):
                raise ValueError(f"{path}:{line_number} is not native 260x346")
            if str(row.get("coordinate_frame", "")) != "distorted":
                raise ValueError(f"{path}:{line_number} is not in distorted coordinates")
            if int(row.get("spatial_downsample", -1)) != 1:
                raise ValueError(f"{path}:{line_number} is spatially downsampled")
            event_value = row.get("path")
            if not isinstance(event_value, str) or not event_value:
                raise ValueError(f"{path}:{line_number} has no event artifact path")
            event_path = Path(event_value).expanduser()
            if not event_path.is_absolute():
                event_path = (path.parent / event_path).resolve(strict=False)
            if args.require_artifacts and not event_path.is_file():
                raise ValueError(f"event artifact is missing: {event_path}")
            rows += 1
    if rows == 0:
        raise ValueError(f"manifest has no rows: {path}")
    if cameras != expected_cameras:
        raise ValueError(
            f"manifest cameras {sorted(cameras)} do not equal expected "
            f"{sorted(expected_cameras)}"
        )
    return {
        "path": str(path),
        "rows": rows,
        "recording": args.expected_recording,
        "split": args.expected_split,
        "cameras": sorted(cameras),
        "artifacts_checked": args.require_artifacts,
    }


def snapshot_index_rows(path: Path, limit: int) -> list[tuple[str, ...]]:
    source = path.expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != (
        "event-window-jepa-mvsec-visualization-index-v1"
    ):
        raise ValueError(f"unsupported visualization index: {source}")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"visualization index has no samples list: {source}")
    if limit < 0:
        raise ValueError("snapshot limit cannot be negative")
    rows: list[tuple[str, ...]] = []
    seen: set[Path] = set()
    for ordinal, item in enumerate(samples):
        if limit and len(rows) >= limit:
            break
        if not isinstance(item, dict):
            raise ValueError(f"snapshot index item {ordinal} is not an object")
        snapshot = Path(str(item.get("path", ""))).expanduser().resolve(strict=True)
        try:
            snapshot.relative_to(source.parent)
        except ValueError as error:
            raise ValueError(f"indexed snapshot escapes its directory: {snapshot}") from error
        if snapshot.suffix.lower() != ".npz" or not snapshot.is_file():
            raise ValueError(f"indexed snapshot is not an NPZ file: {snapshot}")
        if snapshot in seen:
            raise ValueError(f"duplicate indexed snapshot: {snapshot}")
        seen.add(snapshot)
        kind = str(item.get("kind", ""))
        if kind not in {"flow", "depth"}:
            raise ValueError(f"indexed snapshot has invalid kind: {kind!r}")
        expected_bytes = item.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ValueError(f"indexed snapshot has invalid byte size: {snapshot}")
        expected_sha256 = item.get("sha256")
        if not isinstance(expected_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ) is None:
            raise ValueError(f"indexed snapshot has invalid SHA-256: {snapshot}")
        initial_stat = snapshot.stat()
        initial_identity = (
            initial_stat.st_dev,
            initial_stat.st_ino,
            initial_stat.st_size,
            initial_stat.st_mtime_ns,
            initial_stat.st_ctime_ns,
        )
        if initial_stat.st_size != expected_bytes:
            raise ValueError(f"indexed snapshot byte size changed: {snapshot}")
        if _sha256_path(snapshot) != expected_sha256:
            raise ValueError(f"indexed snapshot SHA-256 changed: {snapshot}")
        final_stat = snapshot.stat()
        final_identity = (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        )
        if final_identity != initial_identity:
            raise ValueError(f"indexed snapshot changed during verification: {snapshot}")
        rows.append(
            (
                str(ordinal),
                str(snapshot),
                kind,
                str(expected_bytes),
                expected_sha256,
            )
        )
    return rows


def render_config(args: argparse.Namespace) -> bytes:
    template = args.template.expanduser().resolve(strict=True)
    original = template.read_bytes()
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"template is not UTF-8: {template}") from error
    lines = text.splitlines(keepends=True)
    if not lines or any(not line.endswith("\n") for line in lines[:-1]):
        raise ValueError("template has unsupported line endings")

    cmax_weight = Decimal(args.cmax_weight)
    temporal_weight = Decimal(args.temporal_sigreg_weight)
    frame_weight = Decimal(args.frame_sigreg_weight)
    rate_weight = Decimal(args.rate_alignment_weight)
    rate_gamma = Decimal(args.rate_alignment_gamma)
    rate_eps = Decimal(args.rate_alignment_eps)
    straightening_weight = Decimal(args.latent_straightening_weight)
    straightening_eps = Decimal(args.latent_straightening_eps)
    for value in (
        cmax_weight,
        temporal_weight,
        frame_weight,
        rate_weight,
        rate_gamma,
        straightening_weight,
    ):
        _decimal_text(value)
    if not rate_eps.is_finite() or rate_eps <= 0:
        raise ValueError("Rate Alignment epsilon must be finite and positive")
    if not straightening_eps.is_finite() or straightening_eps <= 0:
        raise ValueError("Latent Straightening epsilon must be finite and positive")
    if args.rate_alignment_normalization != _RATE_NORMALIZATION:
        raise ValueError(
            "unsupported Rate Alignment normalization: "
            f"{args.rate_alignment_normalization!r}"
        )
    allow_unregularized = frame_weight == 0 and temporal_weight == 0
    cmax_enabled = cmax_weight > 0
    if cmax_enabled != _has_section(lines, "cmax"):
        expected = "CMax" if cmax_enabled else "JEPA-only"
        raise ValueError(f"{expected} condition was paired with the wrong template")
    if args.seed < 0 or args.sequence_length <= 0:
        raise ValueError("seed must be non-negative and sequence length positive")
    if args.cmax_reference_mode not in {"past", "future", "both"}:
        raise ValueError(f"unsupported CMax reference mode: {args.cmax_reference_mode}")
    try:
        cmax_scales = tuple(int(value) for value in args.cmax_temporal_scales.split(","))
    except ValueError as error:
        raise ValueError("CMax temporal scales must be comma-separated integers") from error
    if (
        not cmax_scales
        or any(scale <= 0 for scale in cmax_scales)
        or tuple(sorted(set(cmax_scales))) != cmax_scales
        or any(args.sequence_length % scale for scale in cmax_scales)
    ):
        raise ValueError(
            "CMax temporal scales must be positive, unique, increasing divisors "
            "of the recurrent sequence length"
        )
    if args.epochs <= 0 or args.warmup_epochs < 0:
        raise ValueError("epochs must be positive and warmup non-negative")
    if args.warmup_epochs >= args.epochs:
        raise ValueError("warmup epochs must be shorter than total epochs")
    if args.batch_size <= 0 or args.workers < 0 or args.samples_per_epoch <= 0:
        raise ValueError("batch/samples must be positive and workers non-negative")
    if args.precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"unsupported precision: {args.precision}")
    if not re.fullmatch(r"mvsec_[a-z0-9_]+__seed[0-9]+(?:__smoke)?", args.run_id):
        raise ValueError(f"unsafe run identifier: {args.run_id!r}")

    manifest = str(args.manifest.expanduser().resolve(strict=False))
    run_output = str(args.run_output.expanduser().resolve(strict=False))
    _replace_scalar(lines, "data", "manifest", _yaml_string(manifest))
    _replace_scalar(lines, "data", "samples_per_epoch", str(args.samples_per_epoch))
    _replace_scalar(lines, "data", "batch_size", str(args.batch_size))
    _replace_scalar(lines, "data", "workers", str(args.workers))
    _replace_scalar(
        lines,
        "future_prediction",
        "temporal_sigreg_weight",
        _decimal_text(temporal_weight),
    )
    _replace_scalar(
        lines,
        "future_prediction",
        "frame_sigreg_weight",
        _decimal_text(frame_weight),
    )
    _replace_or_insert_scalar(
        lines,
        "future_prediction",
        "rate_alignment_weight",
        _decimal_text(rate_weight),
        after="temporal_sigreg_weight",
    )
    _replace_or_insert_scalar(
        lines,
        "future_prediction",
        "rate_alignment_gamma",
        _decimal_text(rate_gamma),
        after="rate_alignment_weight",
    )
    _replace_or_insert_scalar(
        lines,
        "future_prediction",
        "rate_alignment_eps",
        _decimal_text(rate_eps),
        after="rate_alignment_gamma",
    )
    _replace_or_insert_scalar(
        lines,
        "future_prediction",
        "rate_alignment_normalization",
        _yaml_string(args.rate_alignment_normalization),
        after="rate_alignment_eps",
    )
    _replace_or_insert_scalar(
        lines,
        "future_prediction",
        "latent_straightening_weight",
        _decimal_text(straightening_weight),
        after="rate_alignment_normalization",
    )
    _replace_or_insert_scalar(
        lines,
        "future_prediction",
        "latent_straightening_eps",
        _decimal_text(straightening_eps),
        after="latent_straightening_weight",
    )
    _replace_or_insert_scalar(
        lines,
        "future_prediction",
        "allow_unregularized",
        "true" if allow_unregularized else "false",
        after="latent_straightening_eps",
    )
    _replace_scalar(lines, "recurrent", "sequence_length", str(args.sequence_length))
    _replace_scalar(lines, "recurrent", "tbptt_steps", str(args.sequence_length))
    if cmax_enabled:
        _replace_scalar(lines, "cmax", "enabled", "true")
        _replace_scalar(lines, "cmax", "weight", _decimal_text(cmax_weight))
        _replace_scalar(lines, "cmax", "reference_mode", args.cmax_reference_mode)
        _replace_scalar(
            lines,
            "cmax",
            "temporal_scales",
            "[" + ", ".join(str(scale) for scale in cmax_scales) + "]",
        )
    _replace_scalar(lines, "optimization", "epochs", str(args.epochs))
    _replace_scalar(lines, "optimization", "warmup_epochs", str(args.warmup_epochs))
    _replace_scalar(lines, "optimization", "precision", args.precision)
    _replace_scalar(lines, "runtime", "seed", str(args.seed))
    _replace_scalar(lines, "runtime", "output_dir", _yaml_string(run_output))
    _replace_scalar(lines, "runtime", "resume", "null")

    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "template": str(template),
        "template_sha256": _sha256(original),
        "seed": args.seed,
        "cmax_weight": _decimal_text(cmax_weight),
        "temporal_sigreg_weight": _decimal_text(temporal_weight),
        "frame_sigreg_weight": _decimal_text(frame_weight),
        "rate_alignment_weight": _decimal_text(rate_weight),
        "rate_alignment_gamma": _decimal_text(rate_gamma),
        "rate_alignment_eps": _decimal_text(rate_eps),
        "rate_alignment_normalization": args.rate_alignment_normalization,
        "latent_straightening_weight": _decimal_text(straightening_weight),
        "latent_straightening_eps": _decimal_text(straightening_eps),
        "allow_unregularized": allow_unregularized,
        "sequence_length": args.sequence_length,
        "cmax_reference_mode": args.cmax_reference_mode,
        "cmax_temporal_scales": list(cmax_scales),
        "manifest": manifest,
        "run_output": run_output,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "exposure_policy": _EXPOSURE_POLICY,
        "batch_size_per_rank": args.batch_size,
        "workers_per_rank": args.workers,
        "precision": args.precision,
    }
    header = (
        "# Generated by scripts/experiments/mvsec_ablation_config.py.\n"
        "# Do not edit: regenerate from the recorded template and matrix.\n"
        f"# ablation_metadata: {json.dumps(metadata, sort_keys=True)}\n"
    )
    return (header + "".join(lines)).encode("utf-8")


def _write_exclusive_or_verify(path: Path, content: bytes, check_only: bool) -> str:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"config destination is not a file: {destination}")
        if destination.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite a different generated config: {destination}"
            )
        return "unchanged"
    if check_only:
        return "new"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"config appeared concurrently with different content: {destination}"
                )
            return "unchanged"
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    matrix = commands.add_parser("matrix", help="print a named matrix as TSV")
    matrix.add_argument("--suite", choices=tuple(SUITES), default="core")
    matrix.add_argument("--seeds", default="0,1,2")
    matrix.add_argument("--smoke", action="store_true")

    lookup = commands.add_parser("lookup", help="resolve one known generated run ID")
    lookup.add_argument("--run-id", required=True)

    seeds = commands.add_parser("seeds", help="validate and print a seed CSV")
    seeds.add_argument("--values", required=True)

    path = commands.add_parser("path", help="print one normalized absolute path")
    path.add_argument("value")

    number = commands.add_parser("number", help="validate one finite decimal")
    number.add_argument("value")
    number.add_argument("--allow-zero", action="store_true")

    fraction = commands.add_parser("fraction", help="validate a decimal in (0, 1)")
    fraction.add_argument("value")

    manifest = commands.add_parser(
        "manifest", help="validate an MVSEC recording/split without HDF5 imports"
    )
    manifest.add_argument("--path", type=Path, required=True)
    manifest.add_argument("--expected-recording", required=True)
    manifest.add_argument("--expected-split", required=True)
    manifest.add_argument("--expected-cameras", required=True)
    manifest.add_argument("--require-artifacts", action="store_true")

    snapshots = commands.add_parser(
        "snapshot-index", help="print trusted snapshot-index entries as TSV"
    )
    snapshots.add_argument("--path", type=Path, required=True)
    snapshots.add_argument("--limit", type=int, default=0)

    completion = commands.add_parser(
        "completion", help="record or verify a strict runner completion marker"
    )
    completion.add_argument("--action", choices=("record", "verify"), required=True)
    completion.add_argument("--path", type=Path, required=True)
    completion.add_argument("--kind", choices=("pretrain", "evaluation"), required=True)
    completion.add_argument("--identity", action="append", default=[])
    completion.add_argument("--artifact", action="append", default=[])
    completion.add_argument("--report-role", default="report")
    completion.add_argument("--report-field", action="append", default=[])

    set_id = commands.add_parser(
        "set-id", help="hash an explicit ordered comparison set"
    )
    set_id.add_argument("--item", action="append", default=[])

    render = commands.add_parser("render", help="derive one immutable YAML config")
    render.add_argument("--template", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--run-id", required=True)
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--run-output", type=Path, required=True)
    render.add_argument("--seed", type=int, required=True)
    render.add_argument("--cmax-weight", required=True)
    render.add_argument("--temporal-sigreg-weight", required=True)
    render.add_argument("--frame-sigreg-weight", required=True)
    render.add_argument("--rate-alignment-weight", required=True)
    render.add_argument("--rate-alignment-gamma", required=True)
    render.add_argument("--rate-alignment-eps", required=True)
    render.add_argument("--rate-alignment-normalization", required=True)
    render.add_argument("--latent-straightening-weight", required=True)
    render.add_argument("--latent-straightening-eps", required=True)
    render.add_argument("--sequence-length", type=int, required=True)
    render.add_argument("--cmax-reference-mode", required=True)
    render.add_argument("--cmax-temporal-scales", required=True)
    render.add_argument("--epochs", type=int, required=True)
    render.add_argument("--warmup-epochs", type=int, required=True)
    render.add_argument("--samples-per-epoch", type=int, required=True)
    render.add_argument("--batch-size", type=int, required=True)
    render.add_argument("--workers", type=int, required=True)
    render.add_argument("--precision", required=True)
    render.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "matrix":
            for row in matrix_rows(args.suite, args.seeds, args.smoke):
                print("\t".join(row))
            return 0
        if args.command == "lookup":
            match = re.fullmatch(
                r"mvsec_[a-z0-9_]+__seed(?P<seed>[0-9]+)(?P<smoke>__smoke)?",
                args.run_id,
            )
            if match is None:
                raise ValueError(f"invalid generated run ID: {args.run_id!r}")
            rows = matrix_rows(
                "all",
                match.group("seed"),
                smoke=match.group("smoke") is not None,
            )
            selected = [row for row in rows if row[0] == args.run_id]
            if len(selected) != 1:
                raise ValueError(f"unknown generated run ID: {args.run_id!r}")
            print("\t".join(selected[0]))
            return 0
        if args.command == "seeds":
            for seed in _parse_seeds(args.values):
                print(seed)
            return 0
        if args.command == "path":
            print(Path(args.value).expanduser().resolve(strict=False))
            return 0
        if args.command == "number":
            value = Decimal(args.value)
            if not value.is_finite() or value < 0 or (value == 0 and not args.allow_zero):
                kind = "non-negative" if args.allow_zero else "positive"
                raise ValueError(f"expected a finite {kind} decimal: {args.value!r}")
            print(_decimal_text(value))
            return 0
        if args.command == "fraction":
            value = Decimal(args.value)
            if not value.is_finite() or not Decimal("0") < value < Decimal("1"):
                raise ValueError(f"expected a finite decimal in (0, 1): {args.value!r}")
            print(_decimal_text(value))
            return 0
        if args.command == "manifest":
            print(json.dumps(validate_manifest(args), sort_keys=True))
            return 0
        if args.command == "snapshot-index":
            for row in snapshot_index_rows(args.path, args.limit):
                print("\t".join(row))
            return 0
        if args.command == "completion":
            print(handle_completion(args))
            return 0
        if args.command == "set-id":
            print(comparison_set_identifier(args.item))
            return 0
        content = render_config(args)
        status = _write_exclusive_or_verify(args.output, content, args.check_only)
        if not args.check_only:
            print(f"{status}\t{args.output.expanduser().resolve(strict=False)}")
        return 0
    except (InvalidOperation, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
