"""
Batch Generator: Efficient large-scale Python→IR pair generation.

Optimizations:
1. Multiple kernels per module file (reduces import overhead)
2. Chunked processing to manage memory
3. Progress tracking with resumability
4. Skip backward codegen for faster compilation
"""
import os
import sys
import json
import tempfile
import importlib.util
import hashlib
import random
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

from generator import generate_kernel, GENERATORS, KernelSpec


KERNELS_PER_MODULE = 10  # Number of kernels to put in each module file


def batch_source_hash(sources: list[str]) -> str:
    """Generate a hash for a batch of kernel sources."""
    combined = "\n".join(sources)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def compile_kernel_batch(
    specs: list[KernelSpec], 
    batch_id: int,
    temp_dir: Path,
    device: str = "cpu",
    include_backward: bool = False
) -> list[dict[str, Any]]:
    """
    Compile a batch of kernels from a single module.
    
    Returns list of synthesized pairs.
    """
    import warp as wp
    import warp._src.context as ctx
    
    # Build module source with all kernels
    kernel_sources = [spec.source for spec in specs]
    module_source = "import warp as wp\n\n" + "\n".join(kernel_sources)
    
    source_hash = batch_source_hash(kernel_sources)
    module_name = f"batch_{batch_id}_{source_hash}"
    temp_file = temp_dir / f"{module_name}.py"
    
    with open(temp_file, 'w') as f:
        f.write(module_source)
    
    # Import module
    spec_loader = importlib.util.spec_from_file_location(module_name, temp_file)
    module = importlib.util.module_from_spec(spec_loader)
    sys.modules[module_name] = module
    
    try:
        spec_loader.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        return []
    
    pairs = []
    
    for kernel_spec in specs:
        try:
            kernel = getattr(module, kernel_spec.name, None)
            if kernel is None:
                continue
            
            # Extract IR
            kernel_module = kernel.module
            hasher = ctx.ModuleHasher(kernel_module)
            
            options = kernel_module.options.copy() if kernel_module.options else {}
            options.setdefault("block_dim", 256)
            options.setdefault("enable_backward", include_backward)
            options.setdefault("mode", "release")
            
            builder = ctx.ModuleBuilder(kernel_module, options, hasher)
            ir_code = builder.codegen(device)
            
            # Extract forward function
            mangled_name = kernel.get_mangled_name()
            forward_func_name = f"{mangled_name}_{device}_kernel_forward"
            
            import re
            pattern = rf'void {re.escape(forward_func_name)}\s*\([^)]*\)\s*\{{'
            match = re.search(pattern, ir_code)
            
            if match:
                start = match.start()
                brace_count = 0
                in_function = False
                end = start
                
                for i, char in enumerate(ir_code[start:], start):
                    if char == '{':
                        brace_count += 1
                        in_function = True
                    elif char == '}':
                        brace_count -= 1
                        if in_function and brace_count == 0:
                            end = i + 1
                            break
                
                forward_code = ir_code[start:end]
            else:
                continue
            
            # Extract backward function if requested
            backward_code = None
            if include_backward:
                backward_func_name = f"{mangled_name}_{device}_kernel_backward"
                pattern = rf'void {re.escape(backward_func_name)}\s*\([^)]*\)\s*\{{'
                match = re.search(pattern, ir_code)
                
                if match:
                    start = match.start()
                    brace_count = 0
                    in_function = False
                    end = start
                    
                    for i, char in enumerate(ir_code[start:], start):
                        if char == '{':
                            brace_count += 1
                            in_function = True
                        elif char == '}':
                            brace_count -= 1
                            if in_function and brace_count == 0:
                                end = i + 1
                                break
                    
                    backward_code = ir_code[start:end]
            
            pair = {
                "python_source": kernel_spec.source,
                "ir_forward": forward_code,
                "metadata": {
                    "kernel_name": kernel_spec.name,
                    "category": kernel_spec.category,
                    "description": kernel_spec.description,
                    "device": device,
                    "has_backward": backward_code is not None,
                    **kernel_spec.metadata
                }
            }
            
            if backward_code is not None:
                pair["ir_backward"] = backward_code
            
            pairs.append(pair)
            
        except Exception:
            continue
    
    # Cleanup
    del sys.modules[module_name]
    
    return pairs


def generate_batch(
    n: int,
    output_dir: str | Path,
    seed: int = 42,
    chunk_size: int = 100,
    start_index: int = 0,
    device: str = "cpu",
    include_backward: bool = False
) -> dict[str, Any]:
    """
    Generate n Python→IR pairs in batches.
    
    Args:
        n: Total number of pairs to generate
        output_dir: Directory to save pairs
        seed: Random seed
        chunk_size: Number of pairs per save operation
        start_index: Starting index for file naming (for resumability)
        device: Target device ("cpu" or "cuda")
        include_backward: If True, also generate backward/gradient kernels
    
    Returns:
        Statistics dict
    """
    import warp as wp
    wp.init()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.gettempdir()) / "warp_batch_synthesis"
    temp_dir.mkdir(exist_ok=True)
    
    random.seed(seed)
    
    total_generated = 0
    category_counts = {cat: 0 for cat in GENERATORS.keys()}
    backward_count = 0
    
    start_time = time.time()
    batch_id = 0
    file_index = start_index
    
    print(f"Generating {n} pairs in chunks of {chunk_size}...")
    print(f"Target device: {device}")
    print(f"Include backward: {include_backward}")
    
    while total_generated < n:
        # Generate a chunk of kernel specs
        chunk_n = min(chunk_size, n - total_generated)
        specs = []
        
        for i in range(chunk_n):
            cat = random.choice(list(GENERATORS.keys()))
            spec = generate_kernel(cat, seed=seed + total_generated + i)
            specs.append(spec)
        
        # Process in batches of KERNELS_PER_MODULE
        chunk_pairs = []
        for batch_start in range(0, len(specs), KERNELS_PER_MODULE):
            batch_specs = specs[batch_start:batch_start + KERNELS_PER_MODULE]
            pairs = compile_kernel_batch(batch_specs, batch_id, temp_dir, device, include_backward)
            chunk_pairs.extend(pairs)
            batch_id += 1
        
        # Save chunk
        for pair in chunk_pairs:
            filename = f"pair_{file_index:06d}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
            
            category_counts[pair["metadata"]["category"]] += 1
            if pair["metadata"].get("has_backward", False):
                backward_count += 1
            file_index += 1
        
        total_generated += len(chunk_pairs)
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed if elapsed > 0 else 0
        
        print(f"  Progress: {total_generated}/{n} ({rate:.1f} pairs/sec)")
    
    elapsed = time.time() - start_time
    
    stats = {
        "total_pairs": total_generated,
        "category_distribution": category_counts,
        "backward_count": backward_count,
        "generation_time_sec": elapsed,
        "pairs_per_second": total_generated / elapsed if elapsed > 0 else 0,
        "seed": seed,
        "device": device,
        "include_backward": include_backward,
    }
    
    # Save stats
    stats_file = output_dir / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def run_large_scale_generation(
    n: int = 10000,
    output_dir: str = "/workspace/cuda/data/large",
    seed: int = 42,
    device: str = "cpu",
    include_backward: bool = False
):
    """Run large-scale pair generation."""
    print("=" * 60)
    print("Large-Scale Warp Kernel Synthesis")
    print("=" * 60)
    print(f"Target: {n} pairs")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Include backward: {include_backward}")
    print(f"Seed: {seed}")
    print()
    
    stats = generate_batch(n, output_dir, seed, chunk_size=500, device=device, include_backward=include_backward)
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Time: {stats['generation_time_sec']:.1f}s")
    print(f"Rate: {stats['pairs_per_second']:.1f} pairs/sec")
    if include_backward:
        print(f"Backward kernels: {stats['backward_count']}")
    print("\nCategory distribution:")
    for cat, count in sorted(stats['category_distribution'].items()):
        print(f"  {cat}: {count}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10000, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="/workspace/cuda/data/large", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cpu", help="Target device")
    parser.add_argument("-b", "--backward", action="store_true", help="Include backward/gradient kernels")
    
    args = parser.parse_args()
    
    run_large_scale_generation(args.n, args.output, args.seed, args.device, args.backward)
