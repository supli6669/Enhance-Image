"""Validate the fixed face-restoration benchmark before model evaluation.

This deliberately separates benchmark curation from training. It has a
``--dry-run`` mode so the evaluation set can be checked without loading a model
or processing private images.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {"id", "input_path", "reference_path", "category", "notes"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    input_path: Path
    reference_path: Path | None
    category: str
    notes: str


def _resolve_path(manifest_path: Path, raw_path: str) -> Path | None:
    raw_path = raw_path.strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def load_manifest(manifest_path: Path) -> list[BenchmarkSample]:
    """Load and validate a benchmark manifest without opening its images."""
    if not manifest_path.is_file():
        raise ValueError(f"Benchmark manifest was not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        samples: list[BenchmarkSample] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("id") or "").strip()
            category = (row.get("category") or "").strip()
            input_path = _resolve_path(manifest_path, row.get("input_path") or "")
            reference_path = _resolve_path(manifest_path, row.get("reference_path") or "")
            if not sample_id or not category or input_path is None:
                raise ValueError(f"Manifest row {row_number} needs id, input_path and category.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate benchmark id: {sample_id}")
            if input_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported input image type for {sample_id}: {input_path.suffix}")
            if reference_path and reference_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported reference image type for {sample_id}: {reference_path.suffix}")
            seen_ids.add(sample_id)
            samples.append(BenchmarkSample(sample_id, input_path, reference_path, category, row.get("notes") or ""))

    if not samples:
        raise ValueError("Benchmark manifest has no samples.")
    return samples


def validate_files(samples: Iterable[BenchmarkSample]) -> tuple[int, int]:
    """Return input/reference counts, failing closed for missing benchmark data."""
    input_count = 0
    reference_count = 0
    for sample in samples:
        if not sample.input_path.is_file():
            raise ValueError(f"Missing input for {sample.sample_id}: {sample.input_path}")
        input_count += 1
        if sample.reference_path:
            if not sample.reference_path.is_file():
                raise ValueError(f"Missing reference for {sample.sample_id}: {sample.reference_path}")
            reference_count += 1
    return input_count, reference_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the fixed restoration benchmark manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest for held-out benchmark pairs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; required until the evaluator is configured.")
    args = parser.parse_args()

    if not args.dry_run:
        parser.error("Only --dry-run is enabled until a reviewed benchmark is available.")

    samples = load_manifest(args.manifest)
    input_count, reference_count = validate_files(samples)
    categories = sorted({sample.category for sample in samples})
    print(f"Benchmark valid: {input_count} inputs, {reference_count} references")
    print(f"Categories: {', '.join(categories)}")


if __name__ == "__main__":
    main()
