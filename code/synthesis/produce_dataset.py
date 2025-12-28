#!/usr/bin/env python3
"""
Produce a size-targeted Warp JIT dataset (CPU or CUDA) as sharded JSONL.

This generator is designed to:
- Work in CPU-only environments
- Still emit CUDA code via Warp's codegen (no GPU execution required)
- Stop based on an approximate output byte budget
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from generator import KernelGenerator  # noqa: E402


KERNEL_TYPES: list[str] = [
    "arithmetic",
    "atomic",
    "combined",
    "conditional",
    "loop",
    "math",
    "multi_cond",
    "nested",
    "scalar_param",
    "vector",
]


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def extract_function_by_name(code: str, func_name: str) -> str:
    """
    Extract a single C/CUDA function by name by brace matching.
    Returns empty string if not found.
    """
    idx = code.find(func_name)
    if idx < 0:
        return ""

    # Start at the beginning of the line containing the function name
    start = code.rfind("\n", 0, idx)
    start = 0 if start < 0 else start + 1

    brace_open = code.find("{", idx)
    if brace_open < 0:
        return ""

    brace_count = 0
    end = -1
    for i in range(start, len(code)):
        c = code[i]
        if c == "{":
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0 and i >= brace_open:
                end = i + 1
                break

    if end < 0:
        return ""
    return code[start:end]


def build_module_source(kernel_sources: list[str]) -> str:
    return "import warp as wp\n\n" + "\n\n".join(kernel_sources) + "\n"


def load_temp_module(module_name: str, module_source: str, temp_dir: Path):
    temp_file = temp_dir / f"{module_name}.py"
    temp_file.write_text(module_source, encoding="utf-8")

    spec_loader = importlib.util.spec_from_file_location(module_name, temp_file)
    if spec_loader is None or spec_loader.loader is None:
        raise RuntimeError("failed to create module spec")
    module = importlib.util.module_from_spec(spec_loader)
    sys.modules[module_name] = module
    spec_loader.loader.exec_module(module)
    return module


def write_jsonl_record(f, rec: dict[str, Any]) -> int:
    line = json.dumps(rec, ensure_ascii=False)
    f.write(line + "\n")
    return len((line + "\n").encode("utf-8"))


def produce(
    *,
    target: str,
    output_dir: Path,
    target_bytes: int,
    seed: int,
    kernels_per_module: int,
    shard_records: int,
    max_failures: int,
) -> dict[str, Any]:
    import warp as wp
    import warp._src.context as ctx

    if target not in {"cpu", "cuda"}:
        raise ValueError("target must be cpu|cuda")

    wp.init()

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.gettempdir()) / "warp_dataset_modules"
    temp_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    gen = KernelGenerator(seed=seed)

    shard_idx = 0
    records_in_shard = 0
    bytes_written = 0

    shard_path = output_dir / f"shard_{shard_idx:05d}.jsonl"
    shard_f = shard_path.open("w", encoding="utf-8")

    attempted = 0
    successful = 0
    failed = 0
    by_type: dict[str, int] = {t: 0 for t in KERNEL_TYPES}

    start_time = time.time()
    module_id = 0

    def rotate_shard():
        nonlocal shard_idx, records_in_shard, shard_path, shard_f
        shard_f.close()
        shard_idx += 1
        records_in_shard = 0
        shard_path = output_dir / f"shard_{shard_idx:05d}.jsonl"
        shard_f = shard_path.open("w", encoding="utf-8")

    while bytes_written < target_bytes:
        if failed > max_failures:
            break

        # Build one module containing multiple kernels
        specs: list[tuple[str, str, str]] = []  # (kernel_type, kernel_name, python_source)
        kernel_sources: list[str] = []
        for i in range(kernels_per_module):
            ktype = KERNEL_TYPES[(attempted + i) % len(KERNEL_TYPES)]
            spec = gen.generate(ktype)
            py_src = gen.to_python_source(spec)
            specs.append((ktype, spec.name, py_src))
            kernel_sources.append(py_src)

        attempted += len(specs)

        module_source = build_module_source(kernel_sources)
        source_hash = short_hash(module_source)
        module_name = f"gen_{module_id}_{source_hash}"
        module_id += 1

        try:
            module = load_temp_module(module_name, module_source, temp_dir)
        except Exception:
            failed += len(specs)
            if module_name in sys.modules:
                del sys.modules[module_name]
            continue

        try:
            # All kernels share the same Warp module
            first_kernel = getattr(module, specs[0][1], None)
            if first_kernel is None:
                failed += len(specs)
                del sys.modules[module_name]
                continue

            kernel_module = first_kernel.module
            hasher = ctx.ModuleHasher(kernel_module)

            options = kernel_module.options.copy() if kernel_module.options else {}
            options.setdefault("block_dim", 256)
            options.setdefault("enable_backward", False)
            options.setdefault("mode", "release")

            builder = ctx.ModuleBuilder(kernel_module, options, hasher)
            generated = builder.codegen(target)

            for ktype, kname, py_src in specs:
                try:
                    kernel = getattr(module, kname, None)
                    if kernel is None:
                        failed += 1
                        continue

                    mangled = kernel.get_mangled_name()
                    forward_name = f"{mangled}_{target}_kernel_forward"
                    forward_code = extract_function_by_name(generated, forward_name)
                    if not forward_code:
                        failed += 1
                        continue

                    rec = {
                        "id": short_hash(py_src),
                        "kernel_name": kname,
                        "kernel_type": ktype,
                        "python_source": py_src,
                        "generated_code": forward_code,
                        "target": target,
                        "generated_at": datetime.now().isoformat(),
                        "metadata": {
                            "warp_version": getattr(wp, "__version__", "unknown"),
                            "module_name": module_name,
                            "module_hash": source_hash,
                            "options": options,
                        },
                    }

                    bytes_written += write_jsonl_record(shard_f, rec)
                    records_in_shard += 1
                    successful += 1
                    by_type[ktype] = by_type.get(ktype, 0) + 1

                    if records_in_shard >= shard_records:
                        rotate_shard()

                    if bytes_written >= target_bytes:
                        break
                except Exception:
                    failed += 1
                    continue
        finally:
            # Cleanup module reference
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Lightweight progress print
        elapsed = time.time() - start_time
        if elapsed > 0 and successful % 200 == 0 and successful > 0:
            rate = successful / elapsed
            mb = bytes_written / (1024 * 1024)
            print(f"Progress: {successful} records, {mb:.1f} MB, {rate:.2f} rec/s")

    shard_f.close()

    elapsed = time.time() - start_time
    stats = {
        "target": target,
        "output_dir": str(output_dir),
        "target_mb": target_bytes / (1024 * 1024),
        "bytes_written": bytes_written,
        "mb_written": bytes_written / (1024 * 1024),
        "attempted": attempted,
        "successful": successful,
        "failed": failed,
        "by_type": by_type,
        "seed": seed,
        "kernels_per_module": kernels_per_module,
        "shard_records": shard_records,
        "time_sec": elapsed,
        "records_per_sec": (successful / elapsed) if elapsed > 0 else 0.0,
    }

    (output_dir / "generation_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Produce a size-targeted Warp JIT dataset (CPU or CUDA).")
    parser.add_argument("--target", choices=["cpu", "cuda"], required=True, help="Which backend code to emit")
    parser.add_argument("--output", type=Path, required=True, help="Output directory (JSONL shards)")
    parser.add_argument("--target-mb", type=float, required=True, help="Stop when written bytes >= this many MB")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kernels-per-module", type=int, default=10, help="Kernels generated per temp module")
    parser.add_argument("--shard-records", type=int, default=200, help="Records per JSONL shard file")
    parser.add_argument("--max-failures", type=int, default=5000, help="Abort after this many failures")
    args = parser.parse_args()

    stats = produce(
        target=args.target,
        output_dir=args.output,
        target_bytes=int(args.target_mb * 1024 * 1024),
        seed=args.seed,
        kernels_per_module=args.kernels_per_module,
        shard_records=args.shard_records,
        max_failures=args.max_failures,
    )

    print("\n=== Generation Complete ===")
    print(json.dumps(stats, indent=2))
    return 0 if stats["mb_written"] >= args.target_mb else 2


if __name__ == "__main__":
    raise SystemExit(main())

