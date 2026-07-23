"""Fast regression tests for benchmark manifest validation."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "tools"))

from evaluate_restoration import load_manifest, validate_files


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "inputs"
        reference_dir = root / "references"
        input_dir.mkdir()
        reference_dir.mkdir()
        (input_dir / "sample.jpg").write_bytes(b"test")
        (reference_dir / "sample.jpg").write_bytes(b"test")
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "input_path", "reference_path", "category", "notes"])
            writer.writeheader()
            writer.writerow({
                "id": "sample_001",
                "input_path": "inputs/sample.jpg",
                "reference_path": "references/sample.jpg",
                "category": "jpeg_compression",
                "notes": "fixture",
            })

        samples = load_manifest(manifest)
        assert len(samples) == 1
        assert validate_files(samples) == (1, 1)

        (input_dir / "sample.jpg").unlink()
        try:
            validate_files(samples)
        except ValueError as error:
            assert "Missing input" in str(error)
        else:
            raise AssertionError("Missing benchmark input must fail validation.")

    print("SUCCESS: benchmark manifest workflow validated.")


if __name__ == "__main__":
    main()
