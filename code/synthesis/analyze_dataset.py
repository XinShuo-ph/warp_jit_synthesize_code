#!/usr/bin/env python3
"""Analyze and generate basic statistics for a dataset directory (JSON/JSONL)."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            yield json.load(f)
        return


def analyze_dataset(data_dir: Path) -> dict[str, Any]:
    ignore_names = {"generation_stats.json", "dataset_stats.json"}
    files = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.name not in ignore_names and p.suffix in {".json", ".jsonl"}
    ]

    if not files:
        raise FileNotFoundError(f"no dataset files under: {data_dir}")

    stats: dict[str, Any] = {
        "data_dir": str(data_dir),
        "file_count": len(files),
        "total_bytes": 0,
        "record_count": 0,
        "targets": Counter(),
        "kernel_types": Counter(),
        "kernels": set(),
        "python_lines": [],
        "code_lines": [],
    }

    for fp in files:
        stats["total_bytes"] += os.path.getsize(fp)
        try:
            for rec in _iter_records(fp):
                stats["record_count"] += 1
                if rec.get("target"):
                    stats["targets"][rec["target"]] += 1
                if rec.get("kernel_type"):
                    stats["kernel_types"][rec["kernel_type"]] += 1
                if rec.get("kernel_name"):
                    stats["kernels"].add(rec["kernel_name"])

                if rec.get("python_source"):
                    stats["python_lines"].append(len(rec["python_source"].splitlines()))
                if rec.get("generated_code"):
                    stats["code_lines"].append(len(rec["generated_code"].splitlines()))
        except Exception as e:
            # Keep analysis resilient; count files but skip broken ones
            print(f"Error processing {fp}: {e}")

    def _avg(xs: list[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    stats["total_mb"] = stats["total_bytes"] / (1024 * 1024)
    stats["unique_kernels"] = len(stats["kernels"])
    stats["avg_python_lines"] = _avg(stats["python_lines"])
    stats["avg_code_lines"] = _avg(stats["code_lines"])

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a Warp JIT dataset directory (JSON/JSONL).")
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data/samples"), help="Dataset directory")
    parser.add_argument("--out-json", type=Path, default=None, help="Write stats JSON to this path")
    args = parser.parse_args()

    stats = analyze_dataset(args.data_dir)

    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Data dir:       {stats['data_dir']}")
    print(f"Files:          {stats['file_count']}")
    print(f"Records:        {stats['record_count']}")
    print(f"Total size:     {stats['total_mb']:.2f} MB")
    print(f"Unique kernels: {stats['unique_kernels']}")
    print(f"Avg py lines:   {stats['avg_python_lines']:.1f}")
    print(f"Avg code lines: {stats['avg_code_lines']:.1f}")
    print("\nTargets:")
    for k, v in sorted(stats["targets"].items()):
        print(f"  {k:6s}: {v}")
    print("\nKernel types:")
    for k, v in sorted(stats["kernel_types"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:12s}: {v}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        serializable = dict(stats)
        serializable["targets"] = dict(stats["targets"])
        serializable["kernel_types"] = dict(stats["kernel_types"])
        serializable["kernels"] = sorted(stats["kernels"])
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nWrote: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
