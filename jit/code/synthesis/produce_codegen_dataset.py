"""
Produce a code dataset from Warp JIT/codegen.

This script programmatically generates Warp @wp.kernel functions (Python source),
then extracts the generated source code that Warp emits for:
- CPU: device="cpu"  (C++ code)
- CUDA: device="cuda" (CUDA C++ code)

Output format: JSONL (one JSON object per line).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import warp as wp

# Local imports (repo-relative)
_ROOT = Path(__file__).resolve().parents[1]  # jit/code
sys.path.insert(0, str(_ROOT / "extraction"))
sys.path.insert(0, str(_ROOT / "synthesis"))

from ir_extractor import extract_ir  # noqa: E402
from generator import generate_kernels, GENERATORS, KernelSpec  # noqa: E402


def _kernel_source_hash(source: str) -> str:
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:10]


def _compile_kernel_from_source(source: str, kernel_name: str) -> Any:
    """
    Create a kernel object from a source string by writing/importing a temp module.

    We do not require launching the kernel for codegen extraction; code is generated
    via warp._src.context.ModuleBuilder in the extractor.
    """
    module_source = f"""import warp as wp

{source}
"""

    source_hash = _kernel_source_hash(source)
    temp_dir = Path(tempfile.gettempdir()) / "warp_codegen_dataset"
    temp_dir.mkdir(parents=True, exist_ok=True)

    module_name = f"synth_{kernel_name}_{source_hash}"
    temp_file = temp_dir / f"{module_name}.py"
    temp_file.write_text(module_source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for temp module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Kernel {kernel_name} not found in module")

    return kernel


def _record_for_device(
    spec: KernelSpec,
    *,
    device: str,
    include_backward: bool,
    include_python_source: bool,
    code_scope: str,
) -> dict[str, Any] | None:
    try:
        kernel = _compile_kernel_from_source(spec.source, spec.name)
        _ = kernel.module  # ensure module exists
        ir = extract_ir(kernel, device=device, include_backward=include_backward)

        if ir["forward_code"] is None:
            return None

        if code_scope == "forward":
            code = ir["forward_code"]
        elif code_scope == "full":
            code = ir["code"]
        elif code_scope == "both":
            code = {"forward": ir["forward_code"], "full": ir["code"]}
        else:
            raise ValueError("code_scope must be one of: forward, full, both")

        record: dict[str, Any] = {
            "id": f"{spec.name}_{_kernel_source_hash(spec.source)}",
            "device": device,
            "category": spec.category,
            "description": spec.description,
            "arg_types": spec.arg_types,
            "metadata": dict(spec.metadata),
            "generated_code": code,
            "warp_metadata": ir["metadata"],
        }
        if include_python_source:
            record["python_source"] = spec.source

        return record
    except Exception as e:
        # Keep going; failures are expected for some randomly generated kernels.
        sys.stderr.write(f"[warn] failed {device} codegen for {spec.name}: {e}\n")
        return None


def _write_jsonl_line(fp, obj: dict[str, Any]) -> int:
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    fp.write(line)
    return len(line)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def produce_dataset(
    *,
    device: str,
    target_bytes: int,
    out_path: Path,
    seed: int,
    batch_size: int,
    include_backward: bool,
    include_python_source: bool,
    code_scope: str,
    categories: list[str] | None,
    max_records: int,
) -> dict[str, Any]:
    _ensure_parent(out_path)

    written_bytes = 0
    records = 0
    category_counts: dict[str, int] = {}

    start = time.time()
    with out_path.open("wb") as fp:
        while written_bytes < target_bytes and records < max_records:
            specs = generate_kernels(batch_size, categories=categories, seed=seed + records)
            for spec in specs:
                if written_bytes >= target_bytes or records >= max_records:
                    break

                rec = _record_for_device(
                    spec,
                    device=device,
                    include_backward=include_backward,
                    include_python_source=include_python_source,
                    code_scope=code_scope,
                )
                if rec is None:
                    continue

                written_bytes += _write_jsonl_line(fp, rec)
                records += 1
                category_counts[spec.category] = category_counts.get(spec.category, 0) + 1

    elapsed = time.time() - start
    return {
        "device": device,
        "out_path": str(out_path),
        "bytes": written_bytes,
        "mb": written_bytes / (1024 * 1024),
        "records": records,
        "elapsed_sec": elapsed,
        "records_per_sec": (records / elapsed) if elapsed > 0 else None,
        "category_counts": dict(sorted(category_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "code_scope": code_scope,
        "include_backward": include_backward,
        "include_python_source": include_python_source,
        "seed": seed,
        "batch_size": batch_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")

    parser.add_argument("--target-mb", type=int, default=200, help="Target size in MB for single-device runs")
    parser.add_argument("--target-mb-cpu", type=int, default=200, help="Target CPU size in MB (device=both)")
    parser.add_argument("--target-mb-cuda", type=int, default=200, help="Target CUDA size in MB (device=both)")

    parser.add_argument("--out", type=str, default="", help="Output path for single-device runs (JSONL)")
    parser.add_argument("--out-cpu", type=str, default="jit/data/generated/cpu_code.jsonl")
    parser.add_argument("--out-cuda", type=str, default="jit/data/generated/cuda_code.jsonl")
    parser.add_argument("--stats-out", type=str, default="jit/data/generated/dataset_stats.json")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=1_000_000)

    parser.add_argument("--categories", nargs="+", choices=list(GENERATORS.keys()))

    parser.add_argument("--include-backward", action="store_true", help="Also extract backward kernel functions when present")
    parser.add_argument("--no-python-source", action="store_true", help="Exclude python_source from records")
    parser.add_argument("--code-scope", choices=["forward", "full", "both"], default="full")

    args = parser.parse_args()

    wp.set_module_options({"enable_backward": args.include_backward})
    wp.init()

    stats: dict[str, Any] = {
        "generated_at_unix": int(time.time()),
        "warp_version": getattr(wp, "__version__", None),
        "generator_categories": list(GENERATORS.keys()),
        "runs": [],
    }

    if args.device in {"cpu", "cuda"}:
        out = Path(args.out) if args.out else Path(f"jit/data/generated/{args.device}_code.jsonl")
        run = produce_dataset(
            device=args.device,
            target_bytes=args.target_mb * 1024 * 1024,
            out_path=out,
            seed=args.seed,
            batch_size=args.batch_size,
            include_backward=args.include_backward,
            include_python_source=not args.no_python_source,
            code_scope=args.code_scope,
            categories=args.categories,
            max_records=args.max_records,
        )
        stats["runs"].append(run)
    else:
        cpu_run = produce_dataset(
            device="cpu",
            target_bytes=args.target_mb_cpu * 1024 * 1024,
            out_path=Path(args.out_cpu),
            seed=args.seed,
            batch_size=args.batch_size,
            include_backward=args.include_backward,
            include_python_source=not args.no_python_source,
            code_scope=args.code_scope,
            categories=args.categories,
            max_records=args.max_records,
        )
        stats["runs"].append(cpu_run)

        cuda_run = produce_dataset(
            device="cuda",
            target_bytes=args.target_mb_cuda * 1024 * 1024,
            out_path=Path(args.out_cuda),
            seed=args.seed + 10_000_000,
            batch_size=args.batch_size,
            include_backward=args.include_backward,
            include_python_source=not args.no_python_source,
            code_scope=args.code_scope,
            categories=args.categories,
            max_records=args.max_records,
        )
        stats["runs"].append(cuda_run)

    stats_out = Path(args.stats_out)
    _ensure_parent(stats_out)
    stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Print a minimal human summary
    for run in stats["runs"]:
        sys.stdout.write(f'{run["device"]}: {run["mb"]:.1f}MB, {run["records"]} records -> {run["out_path"]}\n')
    sys.stdout.write(f"stats -> {stats_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

