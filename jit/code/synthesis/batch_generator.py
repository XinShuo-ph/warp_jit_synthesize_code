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

try:
    from .generator import KernelGenerator, KernelSpec
except ImportError:
    from generator import KernelGenerator, KernelSpec

try:
    from .pipeline import KERNEL_CATEGORIES
except ImportError:
    from pipeline import KERNEL_CATEGORIES

KERNELS_PER_MODULE = 10  # Number of kernels to put in each module file


def batch_source_hash(sources: list[str]) -> str:
    """Generate a hash for a batch of kernel sources."""
    combined = "\n".join(sources)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def compile_kernel_batch(
    specs: list[KernelSpec], 
    batch_id: int,
    temp_dir: Path
) -> list[dict[str, Any]]:
    """
    Compile a batch of kernels from a single module.
    
    Returns list of synthesized pairs.
    """
    import warp as wp
    import warp._src.context as ctx
    
    # Generate combined module source
    # We need to make sure kernel names are unique in the module, but they should be already
    
    header = """import warp as wp

"""
    
    # Combine sources
    sources = [spec.source for spec in specs]
    combined_source = header + "\n\n".join(sources)
    
    # Write to temp file
    batch_hash = batch_source_hash(sources)
    module_name = f"batch_{batch_id}_{batch_hash}"
    temp_file = temp_dir / f"{module_name}.py"
    
    with open(temp_file, 'w') as f:
        f.write(combined_source)
    
    # Import module
    spec_mod = importlib.util.spec_from_file_location(module_name, temp_file)
    module = importlib.util.module_from_spec(spec_mod)
    sys.modules[module_name] = module
    
    pairs = []
    
    try:
        spec_mod.loader.exec_module(module)
        
        # Process each kernel in the batch
        for i, spec in enumerate(specs):
            try:
                kernel_name = spec.name
                kernel = getattr(module, kernel_name, None)
                
                if kernel is None:
                    print(f"Kernel {kernel_name} not found in batch module")
                    continue
                
                # Force compilation
                _ = kernel.module
                
                # Extract IR
                hasher = ctx.ModuleHasher(kernel.module)
                options = kernel.module.options.copy() if kernel.module.options else {}
                options.setdefault("block_dim", 256)
                options.setdefault("enable_backward", False)
                options.setdefault("mode", "release")
                
                builder = ctx.ModuleBuilder(kernel.module, options, hasher)
                cpp_code = builder.codegen("cpu")
                
                # Extract forward function
                mangled_name = kernel.get_mangled_name()
                forward_func_name = f"{mangled_name}_cpu_kernel_forward"
                
                import re
                pattern = rf'void {re.escape(forward_func_name)}\s*\([^)]*\)\s*\{{'
                match = re.search(pattern, cpp_code)
                
                if match:
                    start = match.start()
                    brace_count = 0
                    in_function = False
                    end = start
                    
                    for j, char in enumerate(cpp_code[start:], start):
                        if char == '{':
                            brace_count += 1
                            in_function = True
                        elif char == '}':
                            brace_count -= 1
                            if in_function and brace_count == 0:
                                end = j + 1
                                break
                    
                    forward_code = cpp_code[start:end]
                    
                    # Create pair
                    pairs.append({
                        "python_source": spec.source,
                        "cpp_forward": forward_code,
                        "metadata": {
                            "kernel_name": spec.name,
                            # Try to find category if possible, else 'unknown'
                            # Since we don't store category in spec, we might need to pass it
                            "category": spec.category,
                            "device": "cpu",
                            "batch_id": batch_id
                        }
                    })
            except Exception as e:
                print(f"Error processing kernel {spec.name}: {e}")
                
    except Exception as e:
        print(f"Failed to load batch module {module_name}: {e}")
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    return pairs


def generate_batch(
    n: int,
    output_dir: str | Path,
    seed: int = 42,
    chunk_size: int = 1000,
    start_index: int = 0
) -> dict[str, Any]:
    """
    Generate n Python→IR pairs in batches.
    """
    import warp as wp
    wp.init()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.gettempdir()) / "warp_batch_synthesis"
    temp_dir.mkdir(exist_ok=True)
    
    random.seed(seed)
    
    total_generated = 0
    # Initialize counts based on available categories
    category_counts = {cat: 0 for cat in KERNEL_CATEGORIES}
    category_counts["unknown"] = 0
    
    start_time = time.time()
    batch_id = 0
    file_index = start_index
    
    print(f"Generating {n} pairs in chunks of {chunk_size}...")
    
    while total_generated < n:
        # Generate a chunk of kernel specs
        chunk_n = min(chunk_size, n - total_generated)
        specs = []
        spec_categories = [] # Keep track of categories
        
        for i in range(chunk_n):
            cat = random.choice(KERNEL_CATEGORIES)
            gen = KernelGenerator(seed=seed + total_generated + i)
            try:
                spec = gen.generate(cat)
                spec.source = gen.to_python_source(spec)
                spec.category = cat
                specs.append(spec)
                spec_categories.append(cat)
            except Exception as e:
                print(f"Error generating spec: {e}")
        
        # Process in batches of KERNELS_PER_MODULE
        chunk_pairs = []
        for batch_start in range(0, len(specs), KERNELS_PER_MODULE):
            batch_end = batch_start + KERNELS_PER_MODULE
            batch_specs = specs[batch_start:batch_end]
            # We need to pass categories to compile_kernel_batch or fix it after
            
            pairs = compile_kernel_batch(batch_specs, batch_id, temp_dir)
            
            # Fix categories
            for k, pair in enumerate(pairs):
                # This alignment assumes pairs correspond to specs in order
                # which might not be true if some failed inside compile_kernel_batch
                # But compile_kernel_batch iterates over specs.
                # If compile_kernel_batch skips some, we lose alignment.
                # Better to store category in spec.
                pass
            
            # Since we didn't add category to KernelSpec, let's just rely on metadata
            # Or add it to metadata in compile_kernel_batch if we could
            # But compile_kernel_batch takes specs.
            # I'll rely on the fact that I should have added category to KernelSpec.
            
            chunk_pairs.extend(pairs)
            batch_id += 1
        
        # Save chunk
        for i, pair in enumerate(chunk_pairs):
            filename = f"pair_{file_index:06d}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
            
            # Update category count if possible
            # pair["metadata"]["category"] is "unknown" currently
            # We can't easily recover it unless we stored it.
            # I'll update KernelSpec again to store category.
            category_counts[pair["metadata"]["category"]] += 1
            file_index += 1
        
        total_generated += len(chunk_pairs)
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed if elapsed > 0 else 0
        
        print(f"  Progress: {total_generated}/{n} ({rate:.1f} pairs/sec)")
    
    elapsed = time.time() - start_time
    
    stats = {
        "total_pairs": total_generated,
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
    seed: int = 42
):
    """Run large-scale pair generation."""
    print("=" * 60)
    print("Large-Scale Warp Kernel Synthesis")
    print("=" * 60)
    print(f"Target: {n} pairs")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print()
    
    try:
        stats = generate_batch(n, output_dir, seed)
        print("\nSUCCESS")
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10000, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="/workspace/jit/data/large", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_large_scale_generation(args.n, args.output, args.seed)
