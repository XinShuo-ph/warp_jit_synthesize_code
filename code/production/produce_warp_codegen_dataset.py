"""
Size-target dataset producer for Warp-generated CPU/CUDA code.

Writes JSONL records where each record contains:
- A batch of N kernel Python sources (Warp @wp.kernel)
- The full generated C++/CUDA source from Warp codegen for that module
- Per-kernel metadata (category, description, forward symbol name)

This intentionally does NOT require a working CUDA driver. For CUDA it uses Warp's
code generator (`ModuleBuilder.codegen("cuda")`) which works in CPU-only mode.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHESIS_DIR = REPO_ROOT / "code" / "synthesis"
sys.path.insert(0, str(SYNTHESIS_DIR))

from generator import GENERATORS, KernelSpec, generate_kernel  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_short(s: str, n: int = 12) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def build_module_source(specs: list[KernelSpec]) -> str:
    # Warp requires kernels to come from a real file import (inspect source).
    return "import warp as wp\n\n" + "\n\n".join(spec.source.strip() for spec in specs) + "\n"


def import_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_c_like_function(code: str, function_name: str) -> str | None:
    """
    Extract a single C/CUDA function body by brace matching.

    We look for the first '{' that follows the function signature and then
    walk braces until we return to the starting depth.
    """
    import re

    # Works for both CPU and CUDA because the signature contains "... void <name>(...){"
    pattern = re.compile(rf"\\bvoid\\s+{re.escape(function_name)}\\s*\\([^)]*\\)\\s*\\{{", re.MULTILINE)
    m = pattern.search(code)
    if not m:
        return None

    start = m.start()
    brace_count = 0
    in_body = False
    end = start

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


def codegen_module(
    module,
    device: str,
    *,
    block_dim: int = 256,
    enable_backward: bool = False,
    mode: str = "release",
) -> str:
    import warp._src.context as ctx
    import warp as wp

    # All kernels in a Python module share one Warp module.
    kernel_obj = None
    for v in module.__dict__.values():
        if isinstance(v, wp.Kernel):
            kernel_obj = v
            break

    kernel_module = getattr(kernel_obj, "module", None)
    if kernel_module is None:
        raise RuntimeError("Could not locate Warp module from imported module contents")

    hasher = ctx.ModuleHasher(kernel_module)
    options = kernel_module.options.copy() if kernel_module.options else {}
    options.setdefault("block_dim", block_dim)
    options.setdefault("enable_backward", enable_backward)
    options.setdefault("mode", mode)

    builder = ctx.ModuleBuilder(kernel_module, options, hasher)
    return builder.codegen(device)


def choose_specs(kernels_per_module: int, seed: int) -> list[KernelSpec]:
    cats = list(GENERATORS.keys())
    specs: list[KernelSpec] = []
    for i in range(kernels_per_module):
        cat = random.choice(cats)
        specs.append(generate_kernel(cat, seed=seed + i))
    return specs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bytes_target_from_mib(mib: int) -> int:
    return mib * 1024 * 1024


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Produce size-target Warp codegen dataset (JSONL).")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True, help="Codegen target device")
    parser.add_argument("--target-mb", type=int, default=200, help="Target dataset size in MiB (default: 200)")
    parser.add_argument("--kernels-per-module", type=int, default=50, help="Kernels per record/module (default: 50)")
    parser.add_argument("--seed", type=int, default=1, help="Base seed (default: 1)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--append", action="store_true", help="Append to existing dataset file (default)")
    parser.add_argument("--no-append", dest="append", action="store_false", help="Overwrite existing dataset file")
    parser.set_defaults(append=True)
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on records/modules generated")

    args = parser.parse_args()

    device: str = args.device
    target_bytes = bytes_target_from_mib(args.target_mb)
    kernels_per_module: int = args.kernels_per_module
    base_seed: int = args.seed

    if args.output_dir is None:
        out_dir = REPO_ROOT / "data" / ("production_cpu" if device == "cpu" else "production_cuda")
    else:
        out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    dataset_path = out_dir / (f"{device}_code_dataset.jsonl")
    manifest_path = REPO_ROOT / "data" / "manifests" / f"{device}_manifest.json"

    # Initialize Warp once
    import warp as wp

    wp.init()

    # Resume / overwrite behavior
    if dataset_path.exists() and not args.append:
        dataset_path.unlink()

    written_bytes = dataset_path.stat().st_size if dataset_path.exists() else 0
    records_written = 0
    kernels_written = 0
    category_counts: dict[str, int] = {k: 0 for k in GENERATORS.keys()}

    # If resuming, we don't scan the JSONL (too big). We just keep going with higher seeds.
    # Seed offset uses current file size as a rough monotonic proxy.
    seed_cursor = base_seed + (written_bytes // 1024)  # coarse but deterministic-ish

    start_time = time.time()
    last_manifest_write = 0.0

    mode = "a" if dataset_path.exists() else "w"
    with dataset_path.open(mode, encoding="utf-8") as f:
        while written_bytes < target_bytes:
            if args.max_records is not None and records_written >= args.max_records:
                break

            specs = choose_specs(kernels_per_module, seed=seed_cursor)
            seed_cursor += kernels_per_module + 1

            module_source = build_module_source(specs)
            source_hash = sha256_short(module_source, 12)
            module_name = f"warp_codegen_{device}_{source_hash}"
            module_path = Path(tempfile.gettempdir()) / f"{module_name}.py"
            module_path.write_text(module_source, encoding="utf-8")

            try:
                module = import_module_from_path(module_name, module_path)

                # Codegen full module source once
                generated_code = codegen_module(
                    module,
                    device=device,
                    block_dim=256,
                    enable_backward=False,
                    mode="release",
                )

                # Extract per-kernel forward functions for convenience
                kernels_meta: list[dict[str, Any]] = []
                forward_by_kernel: dict[str, str] = {}

                for spec in specs:
                    kernel = getattr(module, spec.name, None)
                    if kernel is None:
                        continue
                    mangled = kernel.get_mangled_name()
                    forward_symbol = f"{mangled}_{device}_kernel_forward"
                    forward_code = extract_c_like_function(generated_code, forward_symbol)
                    if forward_code is not None:
                        forward_by_kernel[spec.name] = forward_code

                    kernels_meta.append(
                        {
                            "kernel_name": spec.name,
                            "mangled_name": mangled,
                            "forward_symbol": forward_symbol,
                            "category": spec.category,
                            "description": spec.description,
                            "metadata": dict(spec.metadata),
                        }
                    )
                    category_counts[spec.category] = category_counts.get(spec.category, 0) + 1

                record = {
                    "schema_version": 1,
                    "device": device,
                    "generated_at": utc_now_iso(),
                    "warp_version": wp.__version__,
                    "module_name": module_name,
                    "module_source_hash": source_hash,
                    "kernels_per_module": kernels_per_module,
                    "python_sources": [s.source for s in specs],
                    "kernels": kernels_meta,
                    "generated_code": generated_code,
                    "forward_functions": forward_by_kernel,
                }

                line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()

                records_written += 1
                kernels_written += len(kernels_meta)
                written_bytes = dataset_path.stat().st_size

                elapsed = time.time() - start_time
                rate_mib = (written_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{device}] {written_bytes/1024/1024:.1f} MiB / {args.target_mb} MiB  "
                    f"records={records_written} kernels={kernels_written}  rate={rate_mib:.2f} MiB/s"
                )

                # Periodic manifest write (every ~15s)
                now = time.time()
                if now - last_manifest_write > 15:
                    manifest = {
                        "schema_version": 1,
                        "device": device,
                        "warp_version": wp.__version__,
                        "dataset_path": str(dataset_path),
                        "bytes": written_bytes,
                        "mib": written_bytes / 1024 / 1024,
                        "target_mib": args.target_mb,
                        "records_written_this_run": records_written,
                        "kernels_written_this_run": kernels_written,
                        "kernels_per_module": kernels_per_module,
                        "category_counts_this_run": category_counts,
                        "seed_cursor": seed_cursor,
                        "started_at_utc": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                        "updated_at_utc": utc_now_iso(),
                    }
                    write_manifest(manifest_path, manifest)
                    last_manifest_write = now

            finally:
                # Cleanup temp module and sys.modules entry
                try:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                except Exception:
                    pass
                try:
                    module_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # Final manifest
    total_elapsed = time.time() - start_time
    final_manifest = {
        "schema_version": 1,
        "device": device,
        "warp_version": wp.__version__,
        "dataset_path": str(dataset_path),
        "bytes": written_bytes,
        "mib": written_bytes / 1024 / 1024,
        "target_mib": args.target_mb,
        "records_written_this_run": records_written,
        "kernels_written_this_run": kernels_written,
        "kernels_per_module": kernels_per_module,
        "category_counts_this_run": category_counts,
        "generation_time_sec_this_run": total_elapsed,
        "seed_cursor": seed_cursor,
        "started_at_utc": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
        "completed_at_utc": utc_now_iso(),
    }
    write_manifest(manifest_path, final_manifest)
    print(f"[{device}] Done. Wrote {written_bytes/1024/1024:.1f} MiB to {dataset_path}")
    print(f"[{device}] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

