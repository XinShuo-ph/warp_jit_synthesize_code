"""
CUDA Codegen-Only Pipeline: Generate CUDA (.cu) source without requiring a GPU/driver.

This pipeline only calls Warp's code generator (ModuleBuilder.codegen("cuda")) and extracts
the generated forward kernel code. It does NOT launch kernels and therefore can run on
CPU-only machines (as long as this Warp build supports CUDA codegen).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import warp as wp

try:
    from .generator import generate_kernels, GENERATORS, KernelSpec  # type: ignore
    from .pipeline import compile_kernel_from_source
except Exception:
    # Script mode
    from generator import generate_kernels, GENERATORS, KernelSpec  # type: ignore
    from pipeline import compile_kernel_from_source  # type: ignore


def synthesize_cuda_codegen_pair(spec: KernelSpec) -> dict[str, Any] | None:
    try:
        # Import kernel from a real module file (Warp disallows exec-defined kernels).
        kernel = compile_kernel_from_source(spec.source, spec.name)
        _ = kernel.module

        try:
            from ..extraction.ir_extractor import extract_ir  # type: ignore
        except Exception:
            # Script mode: ensure extraction directory is importable
            extraction_dir = Path(__file__).parent.parent / "extraction"
            if str(extraction_dir) not in sys.path:
                sys.path.insert(0, str(extraction_dir))
            from ir_extractor import extract_ir  # type: ignore

        ir = extract_ir(kernel, device="cuda", include_backward=False, require_device=False)
        if not ir["forward_code"]:
            return None

        return {
            "python_source": spec.source,
            "code_forward": ir["forward_code"],
            "code_full": ir["cpp_code"],
            "cu_forward": ir["forward_code"],
            "metadata": {
                "kernel_name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "device": "cuda",
                "code_ext": ir["metadata"]["code_ext"],
                "codegen_only": True,
                **spec.metadata,
            },
        }
    except Exception as e:
        print(f"  Failed CUDA codegen-only for {spec.name}: {e}")
        return None


def run_cuda_codegen_pipeline(
    n: int,
    output_dir: str | Path,
    categories: list[str] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    wp.set_module_options({"enable_backward": False})
    wp.init()

    specs = generate_kernels(n, categories, seed)
    pairs: list[dict[str, Any]] = []

    for i, spec in enumerate(specs):
        if (i + 1) % 10 == 0:
            print(f"  Synthesizing {i + 1}/{n}...")
        pair = synthesize_cuda_codegen_pair(spec)
        if pair is not None:
            pairs.append(pair)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, pair in enumerate(pairs):
        (out / f"cuda_codegen_{i:04d}.json").write_text(json.dumps(pair, indent=2))

    return pairs


def _default_output_dir() -> str:
    return str(Path(tempfile.gettempdir()) / "jit_cuda_codegen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default=_default_output_dir(), help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-c", "--categories", nargs="+", choices=list(GENERATORS.keys()), help="Categories to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("Warp CUDA Codegen-Only Pipeline (no GPU required)")
    print("=" * 60)
    print(f"Generating: {args.n}")
    print(f"Categories: {args.categories or 'all'}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print()

    pairs = run_cuda_codegen_pipeline(args.n, args.output, args.categories, args.seed)
    print(f"Generated {len(pairs)}/{args.n} CUDA codegen-only pairs")
    sys.exit(0 if len(pairs) == args.n else 1)

