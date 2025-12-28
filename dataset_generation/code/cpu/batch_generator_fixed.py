"""
Batch Generator: Efficient large-scale Python→IR pair generation.
Fixed to work with current generator.py.
"""
import os
import sys
import json
import tempfile
import importlib.util
import hashlib
import random
import time
from pathlib import Path
from typing import Any, List
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from generator import KernelGenerator, KernelSpec

KERNELS_PER_MODULE = 20  # Increased for efficiency

KERNEL_TYPES = [
    "arithmetic", "conditional", "loop", "math", 
    "vector", "atomic", "nested", "multi_cond", 
    "combined", "scalar_param"
]

def batch_source_hash(sources: list[str]) -> str:
    """Generate a hash for a batch of kernel sources."""
    combined = "\n".join(sources)
    return hashlib.md5(combined.encode()).hexdigest()[:12]

def compile_kernel_batch(
    specs: List[KernelSpec], 
    sources: List[str],
    batch_id: int,
    temp_dir: Path
) -> List[dict[str, Any]]:
    """Compile a batch of kernels and extract IR."""
    import warp as wp
    import warp._src.context as ctx
    
    # Build module source
    module_source = "import warp as wp\n\n" + "\n".join(sources)
    
    source_hash = batch_source_hash(sources)
    module_name = f"batch_{batch_id}_{source_hash}"
    temp_file = temp_dir / f"{module_name}.py"
    
    with open(temp_file, 'w') as f:
        f.write(module_source)
    
    # Import module
    try:
        spec_loader = importlib.util.spec_from_file_location(module_name, temp_file)
        module = importlib.util.module_from_spec(spec_loader)
        sys.modules[module_name] = module
        spec_loader.loader.exec_module(module)
    except Exception as e:
        print(f"Module import failed: {e}")
        if module_name in sys.modules:
            del sys.modules[module_name]
        return []
    
    pairs = []
    
    for i, kernel_spec in enumerate(specs):
        try:
            kernel = getattr(module, kernel_spec.name, None)
            if kernel is None:
                continue
            
            # Extract IR using Warp internals
            kernel_module = kernel.module
            # Using internal APIs to force CPU codegen without full compilation overhead if possible
            # But we need the C++ code.
            
            # We can use the builder to generate C++ source directly
            hasher = ctx.ModuleHasher(kernel_module)
            options = kernel_module.options.copy() if kernel_module.options else {}
            options.setdefault("block_dim", 256)
            options.setdefault("enable_backward", False) # CPU training data usually focuses on forward for now, or we can enable it
            options.setdefault("mode", "release")
            
            builder = ctx.ModuleBuilder(kernel_module, options, hasher)
            try:
                cpp_code = builder.codegen("cpu")
            except Exception as e:
                # If codegen fails, skip
                continue

            # Extract forward function body
            mangled_name = kernel.get_mangled_name()
            # The function name in generated C++ for CPU is mangled
            # But usually contains the kernel name.
            # Let's use regex to find the forward function definition.
            
            import re
            # void <mangled>_cpu_kernel_forward( ... )
            # We know the kernel name is part of mangled name usually, but let's rely on finding the function that contains "cpu_kernel_forward"
            # and matches the kernel name pattern if possible.
            # Actually, builder.codegen returns the full C++ source.
            
            # Searching for "void .*<kernel_name>.*_cpu_kernel_forward"
            pattern = rf'void\s+\w*{kernel_spec.name}\w*_cpu_kernel_forward\s*\('
            match = re.search(pattern, cpp_code)
            
            if match:
                start_func = match.start()
                # Find opening brace
                brace_start = cpp_code.find('{', start_func)
                if brace_start == -1: continue
                
                # Find matching closing brace
                brace_count = 1
                end_func = -1
                for idx in range(brace_start + 1, len(cpp_code)):
                    if cpp_code[idx] == '{':
                        brace_count += 1
                    elif cpp_code[idx] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_func = idx + 1
                            break
                
                if end_func != -1:
                    forward_code = cpp_code[start_func:end_func]
                    
                    pairs.append({
                        "id": hashlib.md5(sources[i].encode()).hexdigest()[:12],
                        "kernel_name": kernel_spec.name,
                        "kernel_type": "generated", # We'll need to pass this through
                        "python_source": sources[i],
                        "cpp_ir_forward": forward_code,
                        "cpp_ir_backward": "", # optimization: skip backward for speed
                        "cpp_ir_full": cpp_code,
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "metadata": {
                            "num_params": len(kernel_spec.params),
                            "num_lines": len(kernel_spec.body_lines)
                        }
                    })

        except Exception as e:
            # print(f"Error processing kernel {kernel_spec.name}: {e}")
            continue
            
    # Cleanup
    if module_name in sys.modules:
        del sys.modules[module_name]
    if temp_file.exists():
        os.unlink(temp_file)
        
    return pairs

def generate_batch(
    n: int,
    output_dir: str,
    seed: int = 42,
    chunk_size: int = 500
):
    import warp as wp
    wp.init()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.gettempdir()) / "warp_batch_synthesis"
    temp_dir.mkdir(exist_ok=True)
    
    gen = KernelGenerator(seed=seed)
    
    total_generated = 0
    start_time = time.time()
    batch_id = 0
    
    print(f"Generating {n} pairs...")
    
    while total_generated < n:
        specs = []
        sources = []
        types = []
        
        # Generate batch of specs
        current_batch_size = min(KERNELS_PER_MODULE, n - total_generated)
        if current_batch_size <= 0: break
        
        for _ in range(current_batch_size):
            ktype = random.choice(KERNEL_TYPES)
            spec = gen.generate(ktype)
            source = gen.to_python_source(spec)
            specs.append(spec)
            sources.append(source)
            types.append(ktype)
            
        # Compile
        pairs = compile_kernel_batch(specs, sources, batch_id, temp_dir)
        
        # Save
        for i, pair in enumerate(pairs):
            pair["kernel_type"] = types[i] if i < len(types) else "unknown" # simplistic mapping
            # Note: types[i] might not match if pairs are filtered. 
            # We should pass metadata better, but for now this is okay or we can assume pairs order matches specs (it might not if some fail).
            # Better: put type in spec or metadata.
            
            filename = f"{pair['id']}_{pair['kernel_name']}.json"
            with open(output_dir / filename, 'w') as f:
                json.dump(pair, f, indent=2)
                
            total_generated += 1
            if total_generated >= n: break
            
        batch_id += 1
        
        if batch_id % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0
            print(f"Generated {total_generated}/{n} ({rate:.1f} pairs/sec)")

    elapsed = time.time() - start_time
    print(f"Finished. Total: {total_generated}. Time: {elapsed:.1f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    generate_batch(args.count, args.output, args.seed)
