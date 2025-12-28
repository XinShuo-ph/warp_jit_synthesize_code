"""
CUDA Synthesis Pipeline: End-to-end Python→CUDA IR pair generation.

Pipeline: generate kernel → write to file → import → compile → extract CUDA IR → save pair
"""
import os
import sys
import json
import tempfile
import importlib.util
import hashlib
from pathlib import Path
from typing import Any
from dataclasses import asdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

import warp as wp
from generator import generate_kernel, generate_kernels, KernelSpec, GENERATORS
from cuda_ir_extractor import extract_cuda_ir

wp.set_module_options({"enable_backward": False})


def kernel_source_hash(source: str) -> str:
    """Generate a short hash of kernel source."""
    return hashlib.md5(source.encode()).hexdigest()[:8]


def compile_kernel_from_source(source: str, kernel_name: str) -> Any:
    """
    Compile a kernel from source code.
    
    Writes source to a temp file, imports as module, returns kernel object.
    """
    # Create temp file with kernel source
    module_source = f'''import warp as wp

{source}
'''
    
    # Write to temp file
    source_hash = kernel_source_hash(source)
    temp_dir = Path(tempfile.gettempdir()) / "warp_cuda_synthesis"
    temp_dir.mkdir(exist_ok=True)
    
    module_name = f"cuda_synth_{kernel_name}_{source_hash}"
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
        del sys.modules[module_name]
        raise RuntimeError(f"Failed to load module: {e}")
    
    # Get kernel from module
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        del sys.modules[module_name]
        raise RuntimeError(f"Kernel {kernel_name} not found in module")
    
    return kernel


def synthesize_cuda_pair(spec: KernelSpec, device: str = "cuda") -> dict[str, Any] | None:
    """
    Synthesize a Python→CUDA IR pair from a kernel specification.
    
    Returns None if compilation fails.
    """
    try:
        # Compile kernel
        kernel = compile_kernel_from_source(spec.source, spec.name)
        
        # Force compilation by triggering module load
        _ = kernel.module
        
        # Extract CUDA IR
        ir = extract_cuda_ir(kernel, device, include_backward=False)
        
        if ir["forward_code"] is None:
            return None
        
        return {
            "python_source": spec.source,
            "cuda_forward": ir["forward_code"],
            "metadata": {
                "kernel_name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "device": device,
                **spec.metadata
            }
        }
    
    except Exception as e:
        print(f"  Failed to synthesize {spec.name}: {e}")
        return None


def synthesize_cuda_batch(
    n: int,
    categories: list[str] | None = None,
    seed: int | None = None,
    device: str = "cuda"
) -> list[dict[str, Any]]:
    """
    Synthesize a batch of Python→CUDA IR pairs.
    
    Args:
        n: Number of pairs to generate
        categories: Optional list of categories to sample from
        seed: Random seed for reproducibility
        device: Target device for IR generation ("cuda" or "cpu")
    
    Returns:
        List of successfully synthesized pairs
    """
    specs = generate_kernels(n, categories, seed)
    pairs = []
    
    for i, spec in enumerate(specs):
        if (i + 1) % 10 == 0:
            print(f"  Synthesizing {i + 1}/{n}...")
        
        pair = synthesize_cuda_pair(spec, device)
        if pair is not None:
            pairs.append(pair)
    
    return pairs


def save_cuda_pairs(pairs: list[dict], output_dir: str | Path, prefix: str = "cuda_pair"):
    """Save CUDA pairs to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, pair in enumerate(pairs):
        filename = f"{prefix}_{i:04d}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(pair, f, indent=2)
    
    print(f"Saved {len(pairs)} CUDA pairs to {output_dir}")


def run_cuda_pipeline(
    n: int = 100,
    output_dir: str = "/workspace/cuda/data/cuda_samples",
    categories: list[str] | None = None,
    seed: int = 42,
    device: str = "cuda"
):
    """
    Run the full CUDA synthesis pipeline.
    
    Args:
        n: Number of pairs to generate
        output_dir: Directory to save pairs
        categories: Optional list of categories
        seed: Random seed
        device: Target device ("cuda" or "cpu" for comparison)
    """
    print("=" * 60)
    print("Warp CUDA Kernel Synthesis Pipeline")
    print("=" * 60)
    print(f"Generating {n} kernel pairs...")
    print(f"Device: {device}")
    print(f"Categories: {categories or 'all'}")
    print(f"Seed: {seed}")
    print()
    
    wp.init()
    
    pairs = synthesize_cuda_batch(n, categories, seed, device)
    
    print(f"\nSuccessfully synthesized: {len(pairs)}/{n} pairs")
    
    if pairs:
        save_cuda_pairs(pairs, output_dir, prefix=f"{device}_synth")
    
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
    
    parser = argparse.ArgumentParser(description="CUDA Kernel Synthesis Pipeline")
    parser.add_argument("-n", type=int, default=100, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="/workspace/cuda/data/cuda_samples", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu"], help="Target device")
    parser.add_argument("-c", "--categories", nargs="+", choices=list(GENERATORS.keys()), 
                        help="Categories to generate")
    
    args = parser.parse_args()
    
    run_cuda_pipeline(args.n, args.output, args.categories, args.seed, args.device)
