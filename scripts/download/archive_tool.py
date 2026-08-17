#!/usr/bin/env python3
"""Dependency-free verification and resumable extraction helpers.

This module intentionally uses only the Python standard library so the download
scripts can run before the project environment is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator


METADATA_VERSION = 1
COPY_BUFFER_BYTES = 8 * 1024 * 1024
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_lines(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(COPY_BUFFER_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _prefix(path: Path, length: int = 8192) -> bytes:
    with path.open("rb") as handle:
        return handle.read(length)


def _reject_error_document(prefix: bytes) -> None:
    stripped = prefix.lstrip().lower()
    suspicious = (
        b"<!doctype html",
        b"<html",
        b"<?xml",
        b"<error",
        b"<accessdenied",
    )
    if any(stripped.startswith(marker) for marker in suspicious):
        raise ValueError(
            "downloaded content is an HTML/XML error or authentication page"
        )


def _archive_type(path: Path) -> str:
    prefix = _prefix(path, 16)
    if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "zip"
    if prefix.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if tarfile.is_tarfile(path):
        return "tar"
    raise ValueError("file is not a supported ZIP or TAR archive")


def _verify_archive(path: Path) -> str:
    archive_type = _archive_type(path)
    if archive_type == "7z":
        raise ValueError(
            "7z archive detected; extract it manually into an explicit split directory, "
            "then validate that directory with the dataset script's --extracted-root"
        )
    if archive_type == "zip":
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ValueError(f"ZIP CRC check failed: {corrupt_member}")
            if not archive.infolist():
                raise ValueError("ZIP archive is empty")
        return archive_type

    member_count = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            member_count += 1
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read TAR member: {member.name}")
                while extracted.read(COPY_BUFFER_BYTES):
                    pass
    if member_count == 0:
        raise ValueError("TAR archive is empty")
    return archive_type


def _verify_hdf5(path: Path) -> str:
    size = path.stat().st_size
    offsets = [0]
    offset = 512
    while offset < min(size, 1024 * 1024):
        offsets.append(offset)
        offset *= 2
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            if handle.read(len(HDF5_MAGIC)) == HDF5_MAGIC:
                return "hdf5"
    raise ValueError("HDF5 signature was not found")


def _verify_text(path: Path) -> str:
    prefix = _prefix(path)
    _reject_error_document(prefix)
    try:
        prefix.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("expected UTF-8 text") from error
    return "text"


def _validate_expected(
    *, size: int, digest: str, expected_bytes: int | None, expected_sha256: str | None
) -> None:
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"size mismatch: expected {expected_bytes}, got {size}")
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError("SHA-256 mismatch")


def verify_file(
    path: Path,
    kind: str,
    metadata_path: Path,
    expected_bytes: int | None,
    expected_sha256: str | None,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    stat_result = path.stat()
    if stat_result.st_size <= 0:
        raise ValueError("downloaded file is empty")
    prefix = _prefix(path)
    _reject_error_document(prefix)

    if kind == "archive":
        detected_kind = _verify_archive(path)
    elif kind == "hdf5":
        detected_kind = _verify_hdf5(path)
    elif kind == "text":
        detected_kind = _verify_text(path)
    else:
        raise ValueError(f"unsupported verification kind: {kind}")

    digest = _sha256(path)
    _validate_expected(
        size=stat_result.st_size,
        digest=digest,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    payload: dict[str, object] = {
        "metadata_version": METADATA_VERSION,
        "status": "verified",
        "kind": kind,
        "detected_kind": detected_kind,
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "sha256": digest,
        "verified_unix_ns": time.time_ns(),
    }
    _atomic_json(metadata_path, payload)
    return payload


def check_metadata(
    path: Path,
    kind: str,
    metadata_path: Path,
    expected_bytes: int | None,
    expected_sha256: str | None,
) -> dict[str, object]:
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("file or verification metadata is missing")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    stat_result = path.stat()
    required = {
        "metadata_version": METADATA_VERSION,
        "status": "verified",
        "kind": kind,
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"verification metadata mismatch: {key}")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("verification metadata has an invalid SHA-256")
    _validate_expected(
        size=stat_result.st_size,
        digest=digest,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    return payload


def read_http_identity(headers_path: Path) -> dict[str, object]:
    text = headers_path.read_text(encoding="latin-1")
    blocks = re.split(r"\r?\n\r?\n", text)
    selected: dict[str, str] = {}
    status = ""
    for block in blocks:
        lines = [line for line in re.split(r"\r?\n", block) if line]
        if not lines or not lines[0].startswith("HTTP/"):
            continue
        current: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            current[key.strip().lower()] = value.strip()
        if current:
            status = lines[0]
            selected = current
    if not selected:
        raise ValueError("no final HTTP response headers were found")

    content_length: int | None = None
    raw_length = selected.get("content-length")
    if raw_length is not None and raw_length.isdigit():
        content_length = int(raw_length)
    content_range = selected.get("content-range", "")
    range_match = re.search(r"/(\d+)$", content_range)
    if range_match:
        content_length = int(range_match.group(1))

    payload: dict[str, object] = {
        "identity_version": 1,
        "http_status": status,
    }
    if content_length is not None:
        payload["content_length"] = content_length
    safe_fields = {
        "accept-ranges": "accept_ranges",
        "etag": "etag",
        "last-modified": "last_modified",
        "x-amz-checksum-crc64nvme": "checksum_crc64nvme",
        "x-amz-checksum-type": "checksum_type",
    }
    for header, key in safe_fields.items():
        if header in selected:
            payload[key] = selected[header]
    if "content_length" not in payload and not any(
        key in payload for key in ("etag", "last_modified", "checksum_crc64nvme")
    ):
        raise ValueError("HTTP response did not provide a stable object identity")
    return payload


def write_http_identity(headers_path: Path, metadata_path: Path) -> dict[str, object]:
    payload = read_http_identity(headers_path)
    _atomic_json(metadata_path, payload)
    return payload


def compare_http_identity(
    previous_path: Path, current_path: Path, *, allow_weak: bool = False
) -> None:
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    compared = 0
    etag = previous.get("etag")
    has_strong_etag = isinstance(etag, str) and etag and not etag.startswith("W/")
    has_other_validator = "checksum_crc64nvme" in previous
    if not allow_weak and not has_strong_etag and not has_other_validator:
        raise ValueError(
            "remote identity has no strong ETag or provider checksum; "
            "refusing unsafe resume"
        )
    for key in ("content_length", "etag", "last_modified", "checksum_crc64nvme"):
        if key in previous:
            if key not in current:
                raise ValueError(f"remote identity no longer provides {key}")
            compared += 1
            if previous[key] != current[key]:
                raise ValueError(f"remote object changed: {key} mismatch")
    if compared == 0:
        raise ValueError("previous remote identity has no comparable fields")


def _safe_relative_path(name: str) -> Path:
    if "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    return Path(*pure.parts)


def _atomic_copy(source: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            shutil.copyfileobj(source, output, length=COPY_BUFFER_BYTES)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _guard_destination(output_root: Path, relative: Path) -> Path:
    destination = output_root
    for part in relative.parts:
        destination = destination / part
        if destination.is_symlink():
            raise ValueError(f"refusing to extract through a symbolic link: {destination}")
    return destination


def _load_extract_state(
    state_path: Path, fingerprint: dict[str, object]
) -> tuple[set[tuple[str, int]], bool]:
    if not state_path.exists():
        return set(), False
    completed: set[tuple[str, int]] = set()
    text = state_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"empty extraction state: {state_path}")
    rows: list[dict[str, object]] = []
    recovered_tail = False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"corrupt extraction state: {state_path}") from None
            recovered_tail = True
            break
        if not isinstance(row, dict):
            raise ValueError(f"invalid extraction state row: {state_path}")
        rows.append(row)
    if not rows:
        raise ValueError(f"extraction state has no valid header: {state_path}")
    if recovered_tail or not text.endswith("\n"):
        _atomic_json_lines(state_path, rows)

    header = rows[0]
    if header != {"fingerprint": fingerprint, "state_version": 1}:
        raise ValueError(
            f"archive changed since extraction began; inspect {state_path} before retrying"
        )
    is_complete = False
    for row in rows[1:]:
        if row.get("status") == "complete":
            is_complete = True
            continue
        name = row.get("name")
        size = row.get("size")
        if isinstance(name, str) and isinstance(size, int):
            completed.add((name, size))
    return completed, is_complete


def _append_state(state_path: Path, payload: dict[str, object]) -> None:
    with state_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _state_path(archive: Path, output_root: Path, state_root: Path) -> Path:
    identity_source = f"{archive.resolve()}\0{output_root.resolve()}"
    identity = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", archive.name)
    return state_root / f"{safe_name}.{identity}.jsonl"


def _prepare_state(
    archive: Path, output_root: Path, state_path: Path
) -> tuple[set[tuple[str, int]], bool]:
    stat_result = archive.stat()
    fingerprint: dict[str, object] = {
        "path": str(archive.resolve()),
        "output_root": str(output_root.resolve()),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        _atomic_json_line(
            state_path, {"fingerprint": fingerprint, "state_version": 1}
        )
    return _load_extract_state(state_path, fingerprint)


def _zip_members(archive: zipfile.ZipFile) -> Iterator[tuple[str, int, bool, object]]:
    seen: set[str] = set()
    for member in archive.infolist():
        name = member.filename.rstrip("/") if member.is_dir() else member.filename
        if not name:
            continue
        normalized = "/".join(_safe_relative_path(name).parts).casefold()
        if normalized in seen:
            raise ValueError(f"duplicate ZIP member: {name}")
        seen.add(normalized)
        mode = (member.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f"symbolic links are not accepted: {name}")
        yield name, member.file_size, member.is_dir(), member


def _tar_members(archive: tarfile.TarFile) -> Iterator[tuple[str, int, bool, object]]:
    seen: set[str] = set()
    for member in archive:
        name = member.name.rstrip("/") if member.isdir() else member.name
        if not name:
            continue
        normalized = "/".join(_safe_relative_path(name).parts).casefold()
        if normalized in seen:
            raise ValueError(f"duplicate TAR member: {name}")
        seen.add(normalized)
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"links and special TAR members are not accepted: {name}")
        yield name, member.size, member.isdir(), member


def extract_archive(archive_path: Path, output_root: Path, state_root: Path) -> None:
    archive_path = archive_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve()
    state_path = _state_path(archive_path, output_root, state_root)
    completed, is_complete = _prepare_state(archive_path, output_root, state_path)
    if is_complete:
        inventory_complete = True
        archive_type = _archive_type(archive_path)
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path) as archive:
                members = list(_zip_members(archive))
        elif archive_type == "tar":
            with tarfile.open(archive_path, "r:*") as archive:
                members = list(_tar_members(archive))
        else:
            raise ValueError(f"unsupported archive type for extraction: {archive_type}")
        for name, size, is_directory, _member in members:
            destination = _guard_destination(
                output_root, _safe_relative_path(name)
            )
            if is_directory:
                inventory_complete &= destination.is_dir()
            else:
                inventory_complete &= (
                    destination.is_file() and destination.stat().st_size == size
                )
        if inventory_complete:
            print(
                json.dumps(
                    {"status": "already_extracted", "archive": archive_path.name}
                )
            )
            return

    archive_type = _archive_type(archive_path)
    extracted_count = 0
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for name, size, is_directory, member in _zip_members(archive):
                relative = _safe_relative_path(name)
                destination = _guard_destination(output_root, relative)
                if is_directory:
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if (name, size) in completed and destination.is_file():
                    if destination.stat().st_size == size:
                        continue
                with archive.open(member, "r") as source:
                    _atomic_copy(source, destination)
                _append_state(state_path, {"name": name, "size": size})
                extracted_count += 1
    elif archive_type == "tar":
        with tarfile.open(archive_path, "r:*") as archive:
            for name, size, is_directory, member in _tar_members(archive):
                relative = _safe_relative_path(name)
                destination = _guard_destination(output_root, relative)
                if is_directory:
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if (name, size) in completed and destination.is_file():
                    if destination.stat().st_size == size:
                        continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read TAR member: {name}")
                with source:
                    _atomic_copy(source, destination)
                _append_state(state_path, {"name": name, "size": size})
                extracted_count += 1
    else:
        raise ValueError(f"unsupported archive type for extraction: {archive_type}")

    _append_state(state_path, {"status": "complete"})
    print(
        json.dumps(
            {
                "status": "extracted",
                "archive": archive_path.name,
                "new_files": extracted_count,
                "output": str(output_root),
            },
            sort_keys=True,
        )
    )


def _read_dat_header(
    path: Path,
) -> tuple[int, int, int, int | None, int | None]:
    width: int | None = None
    height: int | None = None
    with path.open("rb") as handle:
        while True:
            position = handle.tell()
            marker = handle.read(1)
            if marker != b"%":
                handle.seek(position)
                break
            line = marker + handle.readline()
            decoded = line.decode("latin-1", errors="replace")
            width_match = re.search(r"\bWidth\s+(\d+)", decoded, flags=re.IGNORECASE)
            height_match = re.search(r"\bHeight\s+(\d+)", decoded, flags=re.IGNORECASE)
            if width_match:
                width = int(width_match.group(1))
            if height_match:
                height = int(height_match.group(1))
        type_and_size = handle.read(2)
        if len(type_and_size) != 2:
            raise ValueError(f"truncated DAT header: {path}")
        event_type, event_size = struct.unpack("BB", type_and_size)
        payload_offset = handle.tell()
    return payload_offset, event_type, event_size, width, height


def validate_prophesee(
    root: Path, expected_width: int, expected_height: int, expected_split: str
) -> None:
    dat_files = sorted(root.rglob("*_td.dat"))
    if not dat_files:
        raise ValueError(f"no *_td.dat files were found below {root}")
    split_by_stem: dict[str, str] = {}
    labelled = 0
    headers_with_resolution = 0
    for path in dat_files:
        payload_offset, event_type, event_size, width, height = _read_dat_header(path)
        if event_type != 0 or event_size != 8:
            raise ValueError(
                f"unexpected DAT encoding in {path}: type={event_type}, size={event_size}"
            )
        payload_bytes = path.stat().st_size - payload_offset
        if payload_bytes < 0 or payload_bytes % event_size != 0:
            raise ValueError(f"DAT payload is truncated or misaligned: {path}")
        if (width is None) != (height is None):
            raise ValueError(f"DAT header has an incomplete resolution: {path}")
        if width is not None and width != expected_width:
            raise ValueError(f"DAT width mismatch in {path}: {width}")
        if height is not None and height != expected_height:
            raise ValueError(f"DAT height mismatch in {path}: {height}")
        if width is not None:
            headers_with_resolution += 1
        prefix = path.name.removesuffix("_td.dat")
        label_path = path.with_name(prefix + "_bbox.npy")
        if label_path.is_file():
            labelled += 1
        else:
            raise ValueError(f"bbox annotation is missing for {path}")

        split = "unknown"
        for parent in path.parents:
            lowered = parent.name.lower()
            if lowered in {"train", "test"}:
                split = lowered
                break
            if lowered in {"val", "validation"}:
                split = "val"
                break
        previous = split_by_stem.get(prefix)
        if previous is not None and previous != split:
            raise ValueError(
                f"recording stem appears in multiple splits: {prefix} ({previous}, {split})"
            )
        split_by_stem[prefix] = split

    observed_splits = set(split_by_stem.values())
    if "unknown" in observed_splits:
        raise ValueError(
            "DAT files must be placed below an explicit train/val/test directory"
        )
    if observed_splits != {expected_split}:
        raise ValueError(
            f"expected only split {expected_split}, found {sorted(observed_splits)}"
        )

    print(
        json.dumps(
            {
                "status": "validated",
                "dat_files": len(dat_files),
                "bbox_files": labelled,
                "expected_resolution": [expected_width, expected_height],
                "headers_with_resolution": headers_with_resolution,
                "splits": sorted(observed_splits),
            },
            sort_keys=True,
        )
    )


def _simple_yaml_value(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    return value


def _parse_m3ed_dataset_list(path: Path) -> dict[str, bool]:
    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            if current is not None:
                rows.append(current)
            current = {}
            stripped = stripped[1:].strip()
            if not stripped:
                continue
        if current is None or ":" not in stripped:
            raise ValueError(f"unsupported M3ED YAML syntax at line {line_number}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key in {"file", "filetype", "is_test_file"}:
            current[key] = _simple_yaml_value(value)
    if current is not None:
        rows.append(current)

    assignments: dict[str, bool] = {}
    for row in rows:
        if str(row.get("filetype", "")).lower() != "data":
            continue
        name = row.get("file")
        is_test = row.get("is_test_file")
        if not isinstance(name, str) or not name:
            raise ValueError("M3ED data row has no file name")
        if not isinstance(is_test, bool):
            raise ValueError(f"M3ED data row has invalid is_test_file: {name}")
        if name in assignments and assignments[name] != is_test:
            raise ValueError(f"conflicting M3ED split assignment: {name}")
        assignments[name] = is_test
    if not assignments:
        raise ValueError("no M3ED data entries were found")
    return assignments


def _plain_sequence_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError("sequence list is empty")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9._+-]+", name):
            raise ValueError(f"unsafe sequence name: {name!r}")
    if len(names) != len(set(names)):
        raise ValueError("sequence list contains duplicates")
    return names


def validate_m3ed_plan(dataset_list: Path, sequence_list: Path, split: str) -> None:
    assignments = _parse_m3ed_dataset_list(dataset_list)
    names = _plain_sequence_names(sequence_list)
    expected_test = split == "test"
    missing = sorted(set(names) - set(assignments))
    if missing:
        raise ValueError(f"sequences are absent from official M3ED list: {missing}")
    wrong_split = sorted(name for name in names if assignments[name] != expected_test)
    if wrong_split:
        raise ValueError(
            f"sequences cross the official M3ED train/test boundary: {wrong_split}"
        )
    print(
        json.dumps(
            {"status": "validated", "split": split, "sequences": names},
            sort_keys=True,
        )
    )


def _optional_positive_int(value: str | None) -> int | None:
    if value in {None, "", "-"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _optional_sha256(value: str | None) -> str | None:
    if value in {None, "", "-"}:
        return None
    lowered = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", lowered):
        raise argparse.ArgumentTypeError("expected SHA-256 must contain 64 hex digits")
    return lowered


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--path", required=True, type=Path)
    verify.add_argument("--kind", required=True, choices=("archive", "hdf5", "text"))
    verify.add_argument("--metadata", required=True, type=Path)
    verify.add_argument("--expected-bytes", type=_optional_positive_int)
    verify.add_argument("--expected-sha256", type=_optional_sha256)

    check = subparsers.add_parser("check")
    check.add_argument("--path", required=True, type=Path)
    check.add_argument("--kind", required=True, choices=("archive", "hdf5", "text"))
    check.add_argument("--metadata", required=True, type=Path)
    check.add_argument("--expected-bytes", type=_optional_positive_int)
    check.add_argument("--expected-sha256", type=_optional_sha256)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--state-root", required=True, type=Path)

    validate = subparsers.add_parser("validate-prophesee")
    validate.add_argument("--root", required=True, type=Path)
    validate.add_argument("--width", required=True, type=int)
    validate.add_argument("--height", required=True, type=int)
    validate.add_argument("--split", required=True, choices=("train", "val", "test"))

    m3ed = subparsers.add_parser("validate-m3ed-plan")
    m3ed.add_argument("--dataset-list", required=True, type=Path)
    m3ed.add_argument("--sequence-list", required=True, type=Path)
    m3ed.add_argument("--split", required=True, choices=("train", "val", "test"))

    identity = subparsers.add_parser("http-identity")
    identity.add_argument("--headers", required=True, type=Path)
    identity.add_argument("--metadata", required=True, type=Path)

    compare_identity = subparsers.add_parser("compare-http-identity")
    compare_identity.add_argument("--previous", required=True, type=Path)
    compare_identity.add_argument("--current", required=True, type=Path)
    compare_identity.add_argument(
        "--allow-weak",
        action="store_true",
        help="allow weak HTTP metadata when a publisher SHA-256 is verified later",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command in {"verify", "check"}:
        function = verify_file if args.command == "verify" else check_metadata
        payload = function(
            args.path,
            args.kind,
            args.metadata,
            args.expected_bytes,
            args.expected_sha256,
        )
        print(
            json.dumps(
                {
                    "status": "verified",
                    "kind": payload["kind"],
                    "size_bytes": payload["size_bytes"],
                    "sha256": payload["sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "extract":
        extract_archive(args.archive, args.output, args.state_root)
        return
    if args.command == "validate-prophesee":
        validate_prophesee(args.root, args.width, args.height, args.split)
        return
    if args.command == "validate-m3ed-plan":
        validate_m3ed_plan(args.dataset_list, args.sequence_list, args.split)
        return
    if args.command == "http-identity":
        payload = write_http_identity(args.headers, args.metadata)
        content_length = payload.get("content_length")
        print("" if content_length is None else content_length)
        return
    if args.command == "compare-http-identity":
        compare_http_identity(
            args.previous, args.current, allow_weak=args.allow_weak
        )
        print(json.dumps({"status": "same_remote_object"}))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        raise SystemExit(f"error: {error}") from error
