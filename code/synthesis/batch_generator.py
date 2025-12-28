"""Batch generator: compile multiple kernels per module.

This complements `code/synthesis/pipeline.py` by reducing import overhead when
generating large numbers of samples, while producing the same JSON schema and
supporting `--device cpu|cuda`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


# Ensure repo root is on sys.path so `code.*` imports work when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from code.common.device import resolve_warp_device
from code.extraction.ir_extractor import extract_kernel_functions, get_generated_source_path
from code.synthesis.generator import KernelGenerator, KernelSpec, generate_kernel_module


def hash_source(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def _infer_kernel_type(kernel_name: str) -> str:
    prefix = kernel_name.split("_", 1)[0]
    return {
        "arith": "arithmetic",
        "cond": "conditional",
        "loop": "loop",
        "math": "math",
        "vec": "vector",
        "atomic": "atomic",
        "nested": "nested",
        "multicond": "multi_cond",
        "combined": "combined",
        "scalar": "scalar_param",
    }.get(prefix, "unknown")


def _import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compile_module_and_extract(
    specs: list[KernelSpec],
    *,
    device: str,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    """Compile a single multi-kernel module and return extracted pairs."""
    import warp as wp

    resolved = resolve_warp_device(device)

    module_name = f"batch_{hash_source(''.join(s.name for s in specs))}"
    module_source = generate_kernel_module(specs, module_name=module_name)

    temp_file = temp_dir / f"{module_name}.py"
    temp_file.write_text(module_source)

    module = _import_module_from_path(module_name, temp_file)

    # Compile once; module_id shared for all kernels in this module.
    first_kernel = getattr(module, specs[0].name)
    wp_module = first_kernel.module
    wp_module.load(resolved.name)

    module_id = wp_module.get_module_identifier()
    source_file = get_generated_source_path(module_id, resolved.name)
    full_source = source_file.read_text()

    gen = KernelGenerator()

    results: list[dict[str, Any]] = []
    for spec in specs:
        kernel = getattr(module, spec.name, None)
        if kernel is None:
            continue

        funcs = extract_kernel_functions(full_source, kernel.key, device=resolved.name)
        forward_ir = funcs.get("forward", "")
        backward_ir = funcs.get("backward", "")
        if not forward_ir:
            continue

        py_source = gen.to_python_source(spec)
        results.append(
            {
                "id": hash_source(py_source),
                "kernel_name": spec.name,
                "kernel_type": _infer_kernel_type(spec.name),
                "device": resolved.name,
                "python_source": py_source,
                "cpp_ir_forward": forward_ir,
                "cpp_ir_backward": backward_ir,
                "generated_at": datetime.now().isoformat(),
                "metadata": {
                    "num_params": len(spec.params),
                    "num_lines": len(spec.body_lines),
                    "module_id": module_id,
                    "device": resolved.name,
                    "source_file": str(source_file),
                },
            }
        )

    # Cleanup
    sys.modules.pop(module_name, None)
    try:
        temp_file.unlink(missing_ok=True)
    except Exception:
        pass

    return results


def run_batch_generator(
    *,
    output_dir: str,
    count: int,
    seed: int,
    device: str,
    kernels_per_module: int,
) -> int:
    """Generate `count` pairs into `output_dir`."""
    import warp as wp

    wp.init()
    resolved = resolve_warp_device(device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.gettempdir()) / "warp_batch_synthesis"
    temp_dir.mkdir(parents=True, exist_ok=True)

    gen = KernelGenerator(seed=seed)
    produced = 0

    while produced < count:
        batch_n = min(kernels_per_module, count - produced)
        specs = [gen.generate(None) for _ in range(batch_n)]
        pairs = compile_module_and_extract(specs, device=resolved.name, temp_dir=temp_dir)

        for pair in pairs:
            fp = out / f"{pair['id']}_{pair['kernel_name']}.json"
            fp.write_text(json.dumps(pair, indent=2))
            produced += 1
            if produced >= count:
                break

    return produced


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch synthesis (multiple kernels per module)")
    parser.add_argument("--output", type=str, default="data/batch_samples", help="Output directory")
    parser.add_argument("--count", type=int, default=100, help="Number of pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compilation/extraction device")
    parser.add_argument("--kernels-per-module", type=int, default=10, help="Kernels per temporary module")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    n = run_batch_generator(
        output_dir=str(output_path),
        count=args.count,
        seed=args.seed,
        device=args.device,
        kernels_per_module=args.kernels_per_module,
    )

    print(f"Generated {n} pairs in {output_path}")

