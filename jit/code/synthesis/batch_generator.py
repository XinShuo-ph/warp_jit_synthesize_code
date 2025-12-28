"""
Batch Generator: Efficient large-scale Python→IR pair generation.

Supports both CPU and CUDA backends.
CUDA code generation works WITHOUT a GPU - only execution requires a GPU.
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
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

from generator import KernelGenerator


# Kernel types available
KERNEL_TYPES = [
    "arithmetic", "conditional", "loop", "math", "vector",
    "atomic", "nested", "multi_cond", "combined", "scalar_param"
]


def extract_ir_directly(kernel, device: str = "cpu") -> str | None:
    """
    Extract IR code directly from a kernel using warp's codegen.
    
    This is more reliable than reading from cache files.
    """
    import warp._src.context as ctx
    import re
    
    try:
        module = kernel.module
        hasher = ctx.ModuleHasher(module)
        
        options = module.options.copy() if module.options else {}
        options.setdefault("block_dim", 256)
        options.setdefault("enable_backward", False)
        options.setdefault("mode", "release")
        
        builder = ctx.ModuleBuilder(module, options, hasher)
        full_code = builder.codegen(device)
        
        # Extract forward function
        kernel_name = kernel.key
        mangled_name = kernel.get_mangled_name()
        
        if device == "cuda":
            pattern = rf'__global__ void {re.escape(mangled_name)}_cuda_kernel_forward\s*\([^)]*\)\s*\{{'
        else:
            pattern = rf'void {re.escape(mangled_name)}_cpu_kernel_forward\s*\([^)]*\)\s*\{{'
        
        match = re.search(pattern, full_code)
        if not match:
            return None
        
        start = match.start()
        brace_count = 0
        in_function = False
        end = start
        
        for i, char in enumerate(full_code[start:], start):
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    end = i + 1
                    break
        
        return full_code[start:end]
    
    except Exception as e:
        return None


def generate_batch(
    n: int,
    output_dir: str | Path,
    seed: int = 42,
    device: str = "cpu"
) -> dict[str, Any]:
    """
    Generate n Python→IR pairs.
    
    Args:
        n: Total number of pairs to generate
        output_dir: Directory to save pairs
        seed: Random seed
        device: Device to compile for ("cpu" or "cuda")
    
    Returns:
        Statistics dict
    """
    import warp as wp
    wp.init()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    gen = KernelGenerator(seed=seed)
    
    total_generated = 0
    failed = 0
    category_counts = {cat: 0 for cat in KERNEL_TYPES}
    
    start_time = time.time()
    
    print(f"Generating {n} pairs (device={device})...")
    
    for i in range(n):
        # Cycle through kernel types for variety
        kernel_type = KERNEL_TYPES[i % len(KERNEL_TYPES)]
        
        try:
            # Generate kernel spec
            spec = gen.generate(kernel_type)
            python_source = gen.to_python_source(spec)
            
            # Create temp module
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                module_code = f'''"""Auto-generated kernel module."""
import warp as wp

{python_source}
'''
                f.write(module_code)
                temp_path = f.name
            
            try:
                # Load module
                mod_spec = importlib.util.spec_from_file_location("temp_kernel", temp_path)
                module = importlib.util.module_from_spec(mod_spec)
                sys.modules["temp_kernel"] = module
                mod_spec.loader.exec_module(module)
                
                # Get the kernel
                kernel = getattr(module, spec.name)
                
                # Extract IR directly using codegen (doesn't require device load)
                ir_code = extract_ir_directly(kernel, device)
                
                if not ir_code:
                    raise ValueError("Failed to extract IR")
                
                # Validate it's the right type of code
                if device == "cuda":
                    if "_cuda_kernel_forward" not in ir_code:
                        raise ValueError("Not valid CUDA code")
                else:
                    if "_cpu_kernel_forward" not in ir_code:
                        raise ValueError("Not valid CPU code")
                
                # Create pair
                pair_id = hashlib.sha256(python_source.encode()).hexdigest()[:12]
                pair = {
                    "id": pair_id,
                    "kernel_name": spec.name,
                    "kernel_type": kernel_type,
                    "python_source": python_source,
                    "ir_code": ir_code,
                    "device": device,
                    "metadata": {
                        "num_params": len(spec.params),
                        "num_lines": len(spec.body_lines),
                    }
                }
                
                # Save pair
                filename = f"pair_{total_generated:06d}.json"
                filepath = output_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(pair, f, indent=2)
                
                category_counts[kernel_type] += 1
                total_generated += 1
                
            finally:
                os.unlink(temp_path)
                if "temp_kernel" in sys.modules:
                    del sys.modules["temp_kernel"]
            
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: Failed to generate {kernel_type}: {e}")
        
        # Progress update every 100 pairs
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_generated}/{n} ({rate:.1f} pairs/sec)")
    
    elapsed = time.time() - start_time
    
    stats = {
        "total_pairs": total_generated,
        "failed": failed,
        "category_distribution": category_counts,
        "generation_time_sec": elapsed,
        "pairs_per_second": total_generated / elapsed if elapsed > 0 else 0,
        "seed": seed,
        "device": device,
    }
    
    # Save stats
    stats_file = output_dir / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nGeneration complete: {total_generated} pairs in {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    
    return stats


def run_large_scale_generation(
    n: int = 10000,
    output_dir: str = "/workspace/jit/data/cpu",
    seed: int = 42,
    device: str = "cpu"
):
    """Run large-scale pair generation."""
    print("=" * 60)
    print("Large-Scale Warp Kernel Synthesis")
    print("=" * 60)
    print(f"Target: {n} pairs")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()
    
    if device == "cuda":
        print("Note: CUDA IR generation works WITHOUT a GPU!")
        print("      Only kernel execution requires a GPU.")
        print()
    
    stats = generate_batch(n, output_dir, seed, device)
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Failed: {stats['failed']}")
    print(f"Time: {stats['generation_time_sec']:.1f}s")
    print(f"Rate: {stats['pairs_per_second']:.1f} pairs/sec")
    print("\nCategory distribution:")
    for cat, count in sorted(stats['category_distribution'].items()):
        print(f"  {cat}: {count}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→IR training pairs")
    parser.add_argument("-n", type=int, default=10000, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="/workspace/jit/data/cpu", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"], help="Device backend")
    
    args = parser.parse_args()
    
    run_large_scale_generation(args.n, args.output, args.seed, args.device)
