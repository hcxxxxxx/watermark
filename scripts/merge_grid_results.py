#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded grid_results.csv files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("inputs", nargs="+", help="Input grid_results.csv files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    for input_path in args.inputs:
        path = Path(input_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif fieldnames != list(reader.fieldnames):
                raise ValueError(f"Field mismatch in {path}")
            rows.extend(dict(row) for row in reader)

    if not rows or fieldnames is None:
        raise ValueError("No rows found in input files.")

    rows.sort(key=lambda row: int(float(row["candidate"])))
    best = max(rows, key=lambda row: float(row["objective"]))

    with (output_dir / "grid_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "best": _coerce_row(best),
        "rows": [_coerce_row(row) for row in rows],
        "num_rows": len(rows),
        "inputs": [str(Path(path)) for path in args.inputs],
    }
    with (output_dir / "grid_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with (output_dir / "best_candidate.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["best"], handle, indent=2, sort_keys=True)

    print(f"merged {len(rows)} rows into {output_dir}")
    print(json.dumps(payload["best"], indent=2, sort_keys=True))


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            coerced[key] = None
            continue
        if key in {
            "group",
            "band",
            "mask_mode",
            "detector_mode",
            "norm",
            "mel_scale",
            "center",
        }:
            coerced[key] = value
            continue
        try:
            numeric = float(value)
        except ValueError:
            coerced[key] = value
            continue
        if key in {
            "candidate",
            "block_frames",
            "block_stride",
            "min_block_frames",
            "bits_per_block",
            "pair_bins",
            "pair_candidates",
            "align_max_shift",
            "smooth_frames",
            "payload_bits",
            "code_bits",
            "ecc_repeat",
            "num_rows",
        }:
            coerced[key] = int(numeric)
        else:
            coerced[key] = numeric
    return coerced


if __name__ == "__main__":
    main()
