#!/usr/bin/env python3
"""
Validate random samples from the dataset.

Supports:
- Per-sample JSON files (e.g. `*.json`)
- Sharded JSONL files (e.g. `*.jsonl`)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


def _iter_records(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    """
    Yield (source_id, record) from either:
    - a JSON file containing a single object
    - a JSONL file containing multiple objects
    """
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield f"{path.name}:{i+1}", json.loads(line)
        return

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            yield path.name, json.load(f)
        return


def validate_record(rec: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single dataset record (schema-light sanity checks)."""
    required = ["kernel_name", "kernel_type", "python_source", "generated_code", "target"]
    for k in required:
        if not rec.get(k):
            return False, f"missing/empty field: {k}"

    py_src = rec["python_source"]
    if "@wp.kernel" not in py_src or "def " not in py_src:
        return False, "python_source doesn't look like a Warp kernel"

    tgt = rec["target"]
    if tgt not in {"cpu", "cuda"}:
        return False, "target must be 'cpu' or 'cuda'"

    code = rec["generated_code"]
    if "void " not in code:
        return False, "generated_code doesn't look like C/C++"

    if tgt == "cpu" and "_cpu_kernel_forward" not in code:
        return False, "cpu generated_code missing _cpu_kernel_forward"
    if tgt == "cuda" and "_cuda_kernel_forward" not in code:
        # If Warp changes naming, keep this warning soft by allowing other CUDA markers.
        if "__global__" not in code and "__device__" not in code:
            return False, "cuda generated_code missing expected CUDA markers"

    if len(py_src) < 40:
        return False, "python_source too short"
    if len(code) < 120:
        return False, "generated_code too short"

    return True, "OK"


def collect_candidate_files(data_dir: Path) -> list[Path]:
    ignore_names = {"generation_stats.json", "dataset_stats.json"}
    files: list[Path] = []
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name in ignore_names:
            continue
        if p.suffix in {".json", ".jsonl"}:
            files.append(p)
    return files


def validate_random_samples(data_dir: Path, sample_size: int, seed: int) -> bool:
    files = collect_candidate_files(data_dir)
    if not files:
        print("No samples found!")
        return False

    rng = random.Random(seed)
    files = rng.sample(files, min(sample_size, len(files)))

    passed = 0
    failed = 0

    print(f"Validating up to {sample_size} file(s) from: {data_dir}")
    print("=" * 70)

    for fp in files:
        try:
            any_record = False
            ok = False
            msg = "no records"
            for _, rec in _iter_records(fp):
                any_record = True
                ok, msg = validate_record(rec)
                break  # only validate first record in each file for speed

            if not any_record:
                failed += 1
                print(f"✗ {fp.name:40s} empty file")
                continue

            if ok:
                passed += 1
                print(f"✓ {fp.name:40s} {msg}")
            else:
                failed += 1
                print(f"✗ {fp.name:40s} {msg}")
        except json.JSONDecodeError as e:
            failed += 1
            print(f"✗ {fp.name:40s} JSON parse error: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ {fp.name:40s} error: {e}")

    print("=" * 70)
    print(f"Records checked: {passed + failed}  (passed={passed}, failed={failed})")
    return failed == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Warp JIT dataset directory (JSON/JSONL).")
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data/samples"), help="Dataset directory")
    parser.add_argument("--sample-size", type=int, default=25, help="How many files to spot-check")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling files")
    args = parser.parse_args()

    ok = validate_random_samples(args.data_dir, args.sample_size, args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
