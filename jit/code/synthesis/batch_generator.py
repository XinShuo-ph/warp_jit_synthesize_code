"""
Batch Generator: Efficient large-scale Python→IR pair generation.

Optimizations:
1. Multiple kernels per module file (reduces import overhead)
2. Chunked processing to manage memory
3. Progress tracking with resumability

Supports both CPU and CUDA backends.
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
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

from generator import KernelGenerator, KernelSpec


KERNELS_PER_MODULE = 10  # Number of kernels to put in each module file

# All kernel types supported by the generator
KERNEL_TYPES = [
    "arithmetic", "conditional", "loop", "math", "vector",
    "atomic", "nested", "multi_cond", "combined", "scalar_param"
]


def is_cuda_available() -> bool:
    """Check if CUDA device is available."""
    import warp as wp
    try:
        devices = wp.get_devices()
        return any("cuda" in str(d) for d in devices)
    except Exception:
        return False


def batch_source_hash(sources: list[str]) -> str:
    """Generate a hash for a batch of kernel sources."""
    combined = "\n".join(sources)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def compile_kernel_batch(
    specs: list[tuple[KernelSpec, str, str]],  # (spec, source, kernel_type)
    batch_id: int,
    temp_dir: Path,
    device: str = "cpu"
) -> list[dict[str, Any]]:
    """
    Compile a batch of kernels from a single module.
    
    Args:
        specs: List of (KernelSpec, source_code, kernel_type) tuples
        batch_id: Batch identifier
        temp_dir: Temporary directory for module files
        device: Target device ("cpu" or "cuda")
    
    Returns list of synthesized pairs.
    """
    import warp as wp
    
    # Build module source with all kernels
    kernel_sources = [source for _, source, _ in specs]
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
    
    # Device-specific pattern suffix
    device_suffix = "cuda" if device == "cuda" else "cpu"
    
    for kernel_spec, source_code, kernel_type in specs:
        try:
            kernel = getattr(module, kernel_spec.name, None)
            if kernel is None:
                continue
            
            # Force module compilation for the target device
            kernel_module = kernel.module
            kernel_module.load(device)
            
            # Get cache path and read generated IR
            module_id = kernel_module.get_module_identifier()
            cache_dir = Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))
            
            if device == "cuda":
                ir_file = cache_dir / module_id / f"{module_id}.cu"
                if not ir_file.exists():
                    ir_file = cache_dir / module_id / f"{module_id}.cpp"
            else:
                ir_file = cache_dir / module_id / f"{module_id}.cpp"
            
            if not ir_file.exists():
                continue
            
            ir_code = ir_file.read_text()
            
            # Extract forward function
            kernel_key = kernel.key
            forward_func_pattern = f"{kernel_key}_[a-f0-9]+_{device_suffix}_kernel_forward"
            
            import re
            if device == "cuda":
                pattern = rf'(?:extern\s+"C"\s+)?__global__\s+void\s+{forward_func_pattern}\s*\([^)]*\)\s*\{{'
            else:
                pattern = rf'void\s+{forward_func_pattern}\s*\([^)]*\)\s*\{{'
            
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
            
            pairs.append({
                "python_source": source_code,
                "ir_forward": forward_code,
                "device": device,
                "kernel_type": kernel_type,
                "metadata": {
                    "kernel_name": kernel_spec.name,
                    "num_params": len(kernel_spec.params),
                    "num_lines": len(kernel_spec.body_lines),
                }
            })
            
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
    device: str = "cpu"
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
    
    Returns:
        Statistics dict
    """
    import warp as wp
    wp.init()
    
    # Check CUDA availability
    if device == "cuda" and not is_cuda_available():
        print("ERROR: CUDA device requested but not available.")
        print("Run on a machine with GPU hardware and CUDA driver.")
        return {"error": "CUDA not available"}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.gettempdir()) / "warp_batch_synthesis"
    temp_dir.mkdir(exist_ok=True)
    
    generator = KernelGenerator(seed=seed)
    
    total_generated = 0
    category_counts = {cat: 0 for cat in KERNEL_TYPES}
    
    start_time = time.time()
    batch_id = 0
    file_index = start_index
    
    print(f"Generating {n} {device.upper()} pairs in chunks of {chunk_size}...")
    
    while total_generated < n:
        # Generate a chunk of kernel specs
        chunk_n = min(chunk_size, n - total_generated)
        specs = []
        
        for i in range(chunk_n):
            kernel_type = random.choice(KERNEL_TYPES)
            spec = generator.generate(kernel_type)
            source = generator.to_python_source(spec)
            specs.append((spec, source, kernel_type))
        
        # Process in batches of KERNELS_PER_MODULE
        chunk_pairs = []
        for batch_start in range(0, len(specs), KERNELS_PER_MODULE):
            batch_specs = specs[batch_start:batch_start + KERNELS_PER_MODULE]
            pairs = compile_kernel_batch(batch_specs, batch_id, temp_dir, device=device)
            chunk_pairs.extend(pairs)
            batch_id += 1
        
        # Save chunk
        for pair in chunk_pairs:
            filename = f"pair_{file_index:06d}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
            
            category_counts[pair["kernel_type"]] += 1
            file_index += 1
        
        total_generated += len(chunk_pairs)
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed if elapsed > 0 else 0
        
        print(f"  Progress: {total_generated}/{n} ({rate:.1f} pairs/sec)")
    
    elapsed = time.time() - start_time
    
    stats = {
        "total_pairs": total_generated,
        "device": device,
        "category_distribution": category_counts,
        "generation_time_sec": elapsed,
        "pairs_per_second": total_generated / elapsed if elapsed > 0 else 0,
        "seed": seed,
    }
    
    # Save stats
    stats_file = output_dir / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def run_large_scale_generation(
    n: int = 10000,
    output_dir: str = "/workspace/jit/data/large",
    seed: int = 42,
    device: str = "cpu"
):
    """Run large-scale pair generation.
    
    Args:
        n: Number of pairs to generate
        output_dir: Output directory
        seed: Random seed
        device: Target device ("cpu" or "cuda")
    """
    print("=" * 60)
    print("Large-Scale Warp Kernel Synthesis")
    print("=" * 60)
    print(f"Target: {n} pairs")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print()
    
    stats = generate_batch(n, output_dir, seed, chunk_size=500, device=device)
    
    if "error" in stats:
        print(f"\nGeneration failed: {stats['error']}")
        return stats
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Device: {stats['device']}")
    print(f"Time: {stats['generation_time_sec']:.1f}s")
    print(f"Rate: {stats['pairs_per_second']:.1f} pairs/sec")
    print("\nCategory distribution:")
    for cat, count in sorted(stats['category_distribution'].items()):
        print(f"  {cat}: {count}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Large-scale Python→IR pair generation")
    parser.add_argument("-n", type=int, default=10000, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="/workspace/jit/data/large", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"],
                        help="Target device for IR generation")
    
    args = parser.parse_args()
    
    run_large_scale_generation(args.n, args.output, args.seed, args.device)
