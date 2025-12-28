"""
Synthesis Pipeline: End-to-end Python→IR pair generation.

Pipeline: generate kernel → write to file → import → compile → extract IR → save pair
"""
import os
import sys
import json
import tempfile
import importlib.util
import hashlib
import random
from pathlib import Path
from typing import Any, Optional
from dataclasses import asdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

import warp as wp
try:
    from .generator import KernelGenerator, KernelSpec
except ImportError:
    from generator import KernelGenerator, KernelSpec

wp.set_module_options({"enable_backward": False})

KERNEL_CATEGORIES = [
    "arithmetic", "conditional", "loop", "math", "vector", 
    "atomic", "nested", "multi_cond", "combined", "scalar_param", "random_math"
]

def kernel_source_hash(source: str) -> str:
    """Generate a short hash of kernel source."""
    return hashlib.md5(source.encode()).hexdigest()[:8]


def compile_kernel_from_source(source: str, kernel_name: str) -> Any:
    """
    Compile a kernel from source code.
    
    Writes source to a temp file, imports as module, returns kernel object.
    """
    # Create temp file with kernel source (don't set module options to avoid issues)
    module_source = f'''import warp as wp

{source}
'''
    
    # Write to temp file
    source_hash = kernel_source_hash(source)
    temp_dir = Path(tempfile.gettempdir()) / "warp_synthesis"
    temp_dir.mkdir(exist_ok=True)
    
    module_name = f"synth_{kernel_name}_{source_hash}"
    temp_file = temp_dir / f"{module_name}.py"
    
    with open(temp_file, 'w') as f:
        f.write(module_source)
    
    # Import module - must add to sys.modules BEFORE exec_module
    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise RuntimeError(f"Failed to load module: {e}")
    
    # Get kernel from module
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise RuntimeError(f"Kernel {kernel_name} not found in module")
    
    return kernel


def extract_ir_from_kernel(kernel, device: str = "cpu") -> dict[str, str]:
    """Extract IR (generated C++ code) from a compiled kernel."""
    import warp._src.context as ctx
    
    module = kernel.module
    hasher = ctx.ModuleHasher(module)
    
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", False)
    options.setdefault("mode", "release")
    
    builder = ctx.ModuleBuilder(module, options, hasher)
    try:
        cpp_code = builder.codegen(device)
    except Exception as e:
        print(f"Codegen failed: {e}")
        return {"full_cpp": "", "forward_code": None}
    
    # Extract just the forward kernel function
    mangled_name = kernel.get_mangled_name()
    forward_func_name = f"{mangled_name}_{device}_kernel_forward"
    
    # Find function in code
    import re
    pattern = rf'void {re.escape(forward_func_name)}\s*\([^)]*\)\s*\{{'
    match = re.search(pattern, cpp_code)
    
    if match:
        start = match.start()
        brace_count = 0
        in_function = False
        end = start
        
        for i, char in enumerate(cpp_code[start:], start):
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    end = i + 1
                    break
        
        forward_code = cpp_code[start:end]
    else:
        forward_code = None
    
    return {
        "full_cpp": cpp_code,
        "forward_code": forward_code,
    }


def synthesize_pair(source: str, name: str, category: str, device: str = "cpu") -> dict[str, Any] | None:
    """
    Synthesize a Python→IR pair from a kernel source.
    
    Returns None if compilation fails.
    """
    try:
        # Compile kernel
        kernel = compile_kernel_from_source(source, name)
        
        # Force compilation by triggering module load
        # (Module will be built on first reference to kernel internals)
        _ = kernel.module
        
        # Extract IR
        ir = extract_ir_from_kernel(kernel, device)
        
        if ir["forward_code"] is None:
            return None
        
        return {
            "python_source": source,
            "cpp_forward": ir["forward_code"],
            "metadata": {
                "kernel_name": name,
                "category": category,
                "device": device,
            }
        }
    
    except Exception as e:
        print(f"  Failed to synthesize {name}: {e}")
        return None


def synthesize_batch(
    n: int,
    categories: list[str] | None = None,
    seed: int | None = None,
    device: str = "cpu"
) -> list[dict[str, Any]]:
    """
    Synthesize a batch of Python→IR pairs.
    """
    gen = KernelGenerator(seed=seed)
    pairs = []
    
    if categories is None:
        categories = KERNEL_CATEGORIES
    
    for i in range(n):
        if (i + 1) % 10 == 0:
            print(f"  Synthesizing {i + 1}/{n}...")
        
        cat = random.choice(categories)
        try:
            spec = gen.generate(cat)
            source = gen.to_python_source(spec)
            
            pair = synthesize_pair(source, spec.name, cat, device)
            if pair is not None:
                pairs.append(pair)
        except Exception as e:
            print(f"Error generating kernel: {e}")
            continue
    
    return pairs


def save_pairs(pairs: list[dict], output_dir: str | Path, prefix: str = "pair"):
    """Save pairs to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, pair in enumerate(pairs):
        filename = f"{prefix}_{i:04d}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(pair, f, indent=2)
    
    print(f"Saved {len(pairs)} pairs to {output_dir}")


def run_pipeline(
    n: int = 100,
    output_dir: str = "/workspace/jit/data/samples",
    categories: list[str] | None = None,
    seed: int = 42
):
    """
    Run the full synthesis pipeline.
    """
    print("=" * 60)
    print("Warp Kernel Synthesis Pipeline")
    print("=" * 60)
    print(f"Generating {n} kernel pairs...")
    print(f"Categories: {categories or 'all'}")
    print(f"Seed: {seed}")
    print()
    
    wp.init()
    
    pairs = synthesize_batch(n, categories, seed)
    
    print(f"\nSuccessfully synthesized: {len(pairs)}/{n} pairs")
    
    if pairs:
        save_pairs(pairs, output_dir, prefix="synth")
    
    # Print statistics
    if pairs:
        categories_count = {}
        for pair in pairs:
            cat = pair["metadata"]["category"]
            categories_count[cat] = categories_count.get(cat, 0) + 1
        
        print("\nCategory distribution:")
        for cat, count in sorted(categories_count.items()):
            print(f"  {cat}: {count}")
    
    return pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="/workspace/jit/data/samples", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-c", "--categories", nargs="+", choices=KERNEL_CATEGORIES, 
                        help="Categories to generate")
    
    args = parser.parse_args()
    
    run_pipeline(args.n, args.output, args.categories, args.seed)
