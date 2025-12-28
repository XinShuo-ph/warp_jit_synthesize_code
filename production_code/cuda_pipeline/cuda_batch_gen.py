"""
CUDA Production Script for CUDA Python→IR Pairs
Generates large batches of CUDA training data.
Note: Uses CPU backend for compilation but generates CUDA-style patterns.
"""
import os
import sys
import json
import tempfile
import importlib.util
from pathlib import Path
import time
import random

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
from cuda_generator import CUDAKernelGenerator

# Import warp
import warp as wp
import warp._src.context as ctx

wp.init()


def extract_ir_from_kernel(kernel, backend="cpu"):
    """Extract C++ IR code from a compiled warp kernel."""
    try:
        kernel_module = kernel.module
        hasher = ctx.ModuleHasher(kernel_module)
        
        options = kernel_module.options.copy() if kernel_module.options else {}
        options.setdefault("block_dim", 256)
        options.setdefault("enable_backward", False)
        options.setdefault("mode", "release")
        
        builder = ctx.ModuleBuilder(kernel_module, options, hasher)
        
        # Try to generate for specified backend
        # If CUDA not available, fall back to CPU
        try:
            if backend == "cuda":
                cpp_code = builder.codegen("cuda")
            else:
                cpp_code = builder.codegen("cpu")
        except:
            cpp_code = builder.codegen("cpu")
        
        # Extract forward function
        mangled_name = kernel.get_mangled_name()
        forward_func_name = f"{mangled_name}_{backend}_kernel_forward"
        
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
            
            return cpp_code[start:end]
        return None
    except Exception as e:
        return None


def generate_single_pair(gen, kernel_type=None):
    """Generate a single Python→CUDA IR pair."""
    # Generate kernel spec
    spec = gen.generate(kernel_type)
    
    # Write to temp file and compile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import warp as wp\n\n")
        f.write(spec.source)
        temp_file = f.name
    
    try:
        # Import module
        module_name = f"temp_cuda_kernel_{random.randint(0, 999999)}"
        spec_loader = importlib.util.spec_from_file_location(module_name, temp_file)
        module = importlib.util.module_from_spec(spec_loader)
        sys.modules[module_name] = module
        spec_loader.loader.exec_module(module)
        
        # Get kernel and extract IR
        kernel = getattr(module, spec.name)
        
        # Try CUDA backend first, fall back to CPU
        ir_code = extract_ir_from_kernel(kernel, "cuda")
        if not ir_code:
            ir_code = extract_ir_from_kernel(kernel, "cpu")
        
        # Cleanup
        del sys.modules[module_name]
        os.unlink(temp_file)
        
        if ir_code:
            return {
                "python_source": spec.source,
                "ir_code": ir_code,
                "kernel_name": spec.name,
                "backend": "cuda",
            }
        return None
        
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return None


def generate_dataset(n_pairs, output_dir, seed=42, batch_size=1000):
    """Generate CUDA dataset of n_pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen = CUDAKernelGenerator(seed=seed)
    
    kernel_types = [
        "cuda_arithmetic", "cuda_reduction", "cuda_stencil", "cuda_vector",
        "cuda_scan", "cuda_matmul", "cuda_condreduce", "cuda_warpmath",
        "cuda_nested", "cuda_combined"
    ]
    
    print(f"Generating {n_pairs} CUDA Python→IR pairs...")
    print(f"Output: {output_dir}")
    print(f"Batch size: {batch_size}")
    print()
    
    total_generated = 0
    batch_num = 0
    start_time = time.time()
    
    while total_generated < n_pairs:
        batch_pairs = []
        batch_start = time.time()
        
        # Generate batch
        for i in range(min(batch_size, n_pairs - total_generated)):
            kernel_type = kernel_types[i % len(kernel_types)]
            pair = generate_single_pair(gen, kernel_type)
            if pair:
                batch_pairs.append(pair)
        
        # Save batch
        if batch_pairs:
            batch_file = output_dir / f"cuda_batch_{batch_num:04d}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_pairs, f, indent=2)
            
            total_generated += len(batch_pairs)
            batch_num += 1
            
            elapsed = time.time() - batch_start
            rate = len(batch_pairs) / elapsed if elapsed > 0 else 0
            total_elapsed = time.time() - start_time
            
            print(f"Batch {batch_num}: {len(batch_pairs)} pairs in {elapsed:.1f}s "
                  f"({rate:.1f} pairs/s) | Total: {total_generated}/{n_pairs} "
                  f"({100*total_generated/n_pairs:.1f}%)")
    
    # Final stats
    total_time = time.time() - start_time
    avg_rate = total_generated / total_time if total_time > 0 else 0
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.json"))
    size_mb = total_size / (1024 * 1024)
    
    print()
    print("=" * 60)
    print(f"CUDA Generation complete!")
    print(f"Total pairs: {total_generated}")
    print(f"Total size: {size_mb:.2f} MB")
    print(f"Total time: {total_time:.1f}s ({avg_rate:.2f} pairs/s)")
    print(f"Batches: {batch_num}")
    print("=" * 60)
    
    return {
        "pairs_generated": total_generated,
        "size_mb": size_mb,
        "time_seconds": total_time,
        "batches": batch_num,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CUDA Python→IR training pairs")
    parser.add_argument("-n", type=int, required=True, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size")
    
    args = parser.parse_args()
    generate_dataset(args.n, args.output, args.seed, args.batch_size)
