#!/usr/bin/env python3
"""
Generate Warp JIT code datasets to a target byte size.

Outputs (per record) are JSON lines containing:
- python_source
- generated code (CPU: C++ | CUDA: CUDA C/C++)
- forward/backward kernels when available
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _extract_function_block(code: str, func_name: str) -> str | None:
    """
    Extract a full C/C++ function body for a known function name.
    Uses brace counting to handle nested scopes.
    """
    import re

    # Warp codegen typically emits `void <name>(...) { ... }`
    pat = re.compile(rf"\bvoid\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{")
    m = pat.search(code)
    if not m:
        # Some builds may prefix with macros (rare); be a bit looser.
        pat2 = re.compile(rf"{re.escape(func_name)}\s*\([^)]*\)\s*\{{")
        m = pat2.search(code)
        if not m:
            return None

    start = m.start()
    brace_count = 0
    end = start
    in_body = False
    for i, ch in enumerate(code[start:], start):
        if ch == "{":
            brace_count += 1
            in_body = True
        elif ch == "}":
            brace_count -= 1
            if in_body and brace_count == 0:
                end = i + 1
                break
    if end <= start:
        return None
    return code[start:end]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
            written += 1
    return written


def _import_kernels_from_sources(sources: list[str], module_name: str, work_dir: Path):
    """
    Write a temporary python module containing the given kernel sources and import it.
    Returns the imported module.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    module_file = work_dir / f"{module_name}.py"
    module_src = "import warp as wp\n\nwp.init()\n\n" + "\n\n".join(sources) + "\n"
    module_file.write_text(module_src, encoding="utf-8")
    return _load_module_from_path(module_name, module_file)


def _codegen_for_module(wp_module, device: str, enable_backward: bool) -> str:
    import warp._src.context as ctx

    hasher = ctx.ModuleHasher(wp_module)
    options = wp_module.options.copy() if wp_module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", bool(enable_backward))
    options.setdefault("mode", "release")

    builder = ctx.ModuleBuilder(wp_module, options, hasher)
    return builder.codegen(device)


def _cpu_kernel_type_order():
    return [
        "arithmetic",
        "conditional",
        "loop",
        "math",
        "vector",
        "atomic",
        "nested",
        "multi_cond",
        "combined",
        "scalar_param",
    ]


def _cuda_kernel_type_order():
    return ["arithmetic", "math", "vector", "matrix", "control_flow", "atomic"]


def generate_to_size(
    *,
    mode: str,
    out_dir: Path,
    target_bytes: int,
    batch_kernels: int,
    kernels_per_module: int,
    seed: int,
    enable_backward: bool,
    max_seconds: int | None,
) -> dict[str, Any]:
    import warp as wp

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.jsonl"
    stats_path = out_dir / "stats.json"
    temp_dir = out_dir / "_tmp_modules"
    temp_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if mode == "cpu" else "cuda"

    # Load generators dynamically to avoid `code` stdlib module name collisions.
    cpu_gen_mod = _load_module_from_path(
        "_cpu_generator", Path("/workspace/code/synthesis/generator.py")
    )
    cuda_gen_mod = _load_module_from_path(
        "_cuda_generator", Path("/workspace/cuda/code/synthesis/generator.py")
    )

    wp.init()

    start_time = time.time()
    record_count = 0
    failures = 0
    bytes_start = dataset_path.stat().st_size if dataset_path.exists() else 0
    bytes_now = bytes_start

    by_type: dict[str, int] = {}

    # Deterministic category cycling for coverage.
    if mode == "cpu":
        type_cycle = _cpu_kernel_type_order()
        cpu_gen = cpu_gen_mod.KernelGenerator(seed=seed)
    else:
        type_cycle = _cuda_kernel_type_order()
        cpu_gen = None

    cycle_idx = 0
    next_seed = seed

    while bytes_now < target_bytes:
        if max_seconds is not None and (time.time() - start_time) > max_seconds:
            break

        # Generate specs for this chunk
        specs: list[dict[str, Any]] = []
        sources: list[str] = []

        for _ in range(batch_kernels):
            ktype = type_cycle[cycle_idx % len(type_cycle)]
            cycle_idx += 1

            if mode == "cpu":
                spec = cpu_gen.generate(ktype)
                src = cpu_gen.to_python_source(spec)
                specs.append(
                    {
                        "kernel_type": ktype,
                        "kernel_name": spec.name,
                        "python_source": src,
                        "seed": next_seed,
                    }
                )
                sources.append(src)
            else:
                spec = cuda_gen_mod.generate_kernel(ktype, seed=next_seed)
                specs.append(
                    {
                        "kernel_type": spec.category,
                        "kernel_name": spec.name,
                        "python_source": spec.source,
                        "seed": next_seed,
                        "description": spec.description,
                        "metadata": spec.metadata,
                    }
                )
                sources.append(spec.source)

            next_seed += 1

        # Compile/codegen per-module for efficiency
        chunk_records: list[dict[str, Any]] = []
        for module_start in range(0, len(sources), kernels_per_module):
            mod_sources = sources[module_start : module_start + kernels_per_module]
            mod_specs = specs[module_start : module_start + kernels_per_module]
            module_name = f"synth_{mode}_{_sha12(''.join(mod_sources))}"

            try:
                mod = _import_kernels_from_sources(mod_sources, module_name, temp_dir)
            except Exception:
                failures += len(mod_specs)
                continue

            # Fetch kernel objects
            kernels = []
            for s in mod_specs:
                k = getattr(mod, s["kernel_name"], None)
                if k is not None:
                    kernels.append((k, s))

            if not kernels:
                failures += len(mod_specs)
                continue

            try:
                code = _codegen_for_module(kernels[0][0].module, device=device, enable_backward=enable_backward)
            except Exception:
                failures += len(mod_specs)
                continue

            for kernel, s in kernels:
                try:
                    mangled = kernel.get_mangled_name()
                    fwd_name = f"{mangled}_{device}_kernel_forward"
                    bwd_name = f"{mangled}_{device}_kernel_backward"

                    forward = _extract_function_block(code, fwd_name)
                    backward = _extract_function_block(code, bwd_name) if enable_backward else None

                    if forward is None:
                        failures += 1
                        continue

                    python_src = s["python_source"]
                    rec = {
                        "id": _sha12(f"{mode}:{device}:{python_src}"),
                        "kernel_name": s["kernel_name"],
                        "kernel_type": s["kernel_type"],
                        "device": device,
                        "python_source": python_src,
                        "code_forward": forward,
                        "code_backward": backward,
                        "metadata": {
                            "seed": s["seed"],
                            **({} if mode == "cpu" else {"description": s.get("description"), **(s.get("metadata") or {})}),
                        },
                    }
                    chunk_records.append(rec)
                    by_type[s["kernel_type"]] = by_type.get(s["kernel_type"], 0) + 1
                except Exception:
                    failures += 1
                    continue

        if not chunk_records:
            # Prevent tight infinite loops on systemic failure.
            failures += batch_kernels
            time.sleep(0.1)
            continue

        record_count += _write_jsonl(dataset_path, chunk_records)
        bytes_now = dataset_path.stat().st_size

        elapsed = time.time() - start_time
        rate = (bytes_now - bytes_start) / max(elapsed, 1e-6)
        print(
            f"[{mode}] {bytes_now}/{target_bytes} bytes "
            f"({bytes_now/1024/1024:.1f}MB/{target_bytes/1024/1024:.1f}MB), "
            f"records={record_count}, failures={failures}, "
            f"write_rate={rate/1024/1024:.2f}MB/s"
        )

    elapsed = time.time() - start_time
    stats = {
        "mode": mode,
        "device": device,
        "target_bytes": target_bytes,
        "bytes": bytes_now,
        "records": record_count,
        "failures": failures,
        "by_type": dict(sorted(by_type.items(), key=lambda kv: kv[0])),
        "seed_start": seed,
        "seed_end": next_seed,
        "batch_kernels": batch_kernels,
        "kernels_per_module": kernels_per_module,
        "enable_backward": enable_backward,
        "elapsed_sec": elapsed,
        "bytes_per_sec": (bytes_now - bytes_start) / max(elapsed, 1e-6),
        "dataset_path": str(dataset_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--target-mb", type=float, default=200.0)
    ap.add_argument("--out", type=str, required=True, help="Output directory (will contain dataset.jsonl and stats.json)")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--batch-kernels", type=int, default=200, help="Number of kernels attempted per outer loop")
    ap.add_argument("--kernels-per-module", type=int, default=10, help="Kernels per temporary Python module")
    ap.add_argument("--enable-backward", action="store_true", help="Also try to extract backward kernels")
    ap.add_argument("--max-seconds", type=int, default=None, help="Optional time limit for generation")
    args = ap.parse_args()

    stats = generate_to_size(
        mode=args.mode,
        out_dir=Path(args.out),
        target_bytes=int(args.target_mb * 1024 * 1024),
        batch_kernels=args.batch_kernels,
        kernels_per_module=args.kernels_per_module,
        seed=args.seed,
        enable_backward=args.enable_backward,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

