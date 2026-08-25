from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    PROJECT_ROOT / "scripts/preprocess/preprocess_dsec.sh",
    PROJECT_ROOT / "scripts/preprocess/preprocess_gen1.sh",
)


def test_native_resolution_preprocess_wrappers_have_valid_bash_syntax() -> None:
    for wrapper in WRAPPERS:
        subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_native_resolution_preprocess_wrappers_expose_help() -> None:
    for wrapper in WRAPPERS:
        completed = subprocess.run(
            ["bash", str(wrapper), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--spatial-downsample" not in completed.stdout


def test_native_resolution_preprocess_wrappers_fix_coordinate_conversion() -> None:
    for wrapper in WRAPPERS:
        source = wrapper.read_text(encoding="utf-8")
        assert "--spatial-downsample 1" in source
        assert "--spatial-downsample-method coordinate" in source
        assert "area_accumulate" not in source
