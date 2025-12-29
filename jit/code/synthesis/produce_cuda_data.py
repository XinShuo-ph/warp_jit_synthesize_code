#!/usr/bin/env python3
"""
CUDA Data Production Script
Generates Python→CUDA IR training pairs WITHOUT requiring a physical GPU.
Uses Warp's codegen feature with require_device=False.
"""
import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from generator import KernelGenerator


def get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB."""
    total = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    return total / (1024 * 1024)


def hash_source(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def clear_warp_cache():
    """Clear warp compilation cache to maintain speed."""
    cache_dir = Path(os.path.expanduser("~/.cache/warp"))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


def compile_and_extract_cuda(python_source: str, kernel_name: str):
    """
    Compile kernel and extract CUDA code using codegen-only mode (no GPU required).
    """
    import warp as wp
    import warp._src.context as ctx
    import re
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        module_code = f'"""Temp kernel."""\nimport warp as wp\n\n{python_source}'
        f.write(module_code)
        temp_path = f.name
    
    try:
        spec = importlib.util.spec_from_file_location("temp_kernel", temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_kernel"] = module
        spec.loader.exec_module(module)
        
        kernel = getattr(module, kernel_name)
        wp_module = kernel.module
        
        # Use codegen-only approach - no GPU device required
        hasher = ctx.ModuleHasher(wp_module)
        
        options = wp_module.options.copy() if wp_module.options else {}
        options.setdefault("block_dim", 256)
        options.setdefault("enable_backward", True)
        options.setdefault("mode", "release")
        
        builder = ctx.ModuleBuilder(wp_module, options, hasher)
        
        # Generate CUDA code directly without loading on device
        cuda_code = builder.codegen("cuda")
        
        # Extract forward function
        mangled_name = kernel.get_mangled_name()
        
        def extract_function(pattern, code):
            match = re.search(pattern, code)
            if not match:
                return ""
            start = match.start()
            brace_count = 0
            end = start
            for i, c in enumerate(code[start:]):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break
            return code[start:end]
        
        # CUDA functions may have __global__ qualifier
        forward_pattern = rf'(?:__global__\s+)?void\s+{re.escape(mangled_name)}_cuda_kernel_forward\('
        backward_pattern = rf'(?:__global__\s+)?void\s+{re.escape(mangled_name)}_cuda_kernel_backward\('
        
        forward_ir = extract_function(forward_pattern, cuda_code)
        backward_ir = extract_function(backward_pattern, cuda_code)
        
        return {
            "forward": forward_ir,
            "backward": backward_ir,
            "full": cuda_code,
            "mangled_name": mangled_name
        }
        
    finally:
        os.unlink(temp_path)
        if "temp_kernel" in sys.modules:
            del sys.modules["temp_kernel"]


def run_cuda_production(output_dir: str, target_mb: float = 200.0, 
                        seed: int = 42, cache_clear_interval: int = 50):
    """Run CUDA production."""
    import warp as wp
    wp.init()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = KernelGenerator(seed=seed)
    kernel_types = ["arithmetic", "conditional", "loop", "math", "vector", 
                    "atomic", "nested", "multi_cond", "combined", "scalar_param"]
    
    existing_files = list(output_path.glob("*.json"))
    start_index = len(existing_files)
    current_size = get_dir_size_mb(output_path)
    
    print("=" * 60)
    print("CUDA Data Production (No GPU Required)")
    print("=" * 60)
    print(f"Target: {target_mb:.1f} MB | Current: {current_size:.2f} MB")
    print(f"Existing: {start_index} files | Cache clear every {cache_clear_interval}")
    print()
    
    if current_size >= target_mb:
        print("Already at target!")
        return
    
    stats = {
        "started_at": datetime.now().isoformat(),
        "device": "cuda",
        "target_mb": target_mb,
        "successful": 0,
        "failed": 0,
        "by_type": {k: 0 for k in kernel_types}
    }
    
    file_index = start_index
    start_time = time.time()
    since_clear = 0
    
    clear_warp_cache()
    
    try:
        while current_size < target_mb:
            if since_clear >= cache_clear_interval:
                clear_warp_cache()
                import importlib
                importlib.reload(wp)
                wp.init()
                since_clear = 0
                print(f"  [Cache cleared]")
            
            ktype = kernel_types[file_index % len(kernel_types)]
            
            try:
                spec = generator.generate(ktype)
                python_source = generator.to_python_source(spec)
                
                ir_data = compile_and_extract_cuda(python_source, spec.name)
                
                if not ir_data["forward"]:
                    raise ValueError("No forward CUDA IR extracted")
                
                pair_id = hash_source(python_source)
                pair = {
                    "id": pair_id,
                    "kernel_name": spec.name,
                    "kernel_type": ktype,
                    "python_source": python_source,
                    "cuda_forward": ir_data["forward"],
                    "cuda_backward": ir_data["backward"],
                    "device": "cuda",
                    "generated_at": datetime.now().isoformat(),
                    "metadata": {
                        "num_params": len(spec.params),
                        "num_lines": len(spec.body_lines),
                        "mangled_name": ir_data["mangled_name"]
                    }
                }
                
                filename = f"{pair_id}_{spec.name}.json"
                with open(output_path / filename, 'w') as f:
                    json.dump(pair, f, indent=2)
                
                stats["successful"] += 1
                stats["by_type"][ktype] += 1
                since_clear += 1
                
            except Exception as e:
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    print(f"  Failed ({ktype}): {e}")
            
            file_index += 1
            
            if file_index % 50 == 0:
                current_size = get_dir_size_mb(output_path)
                elapsed = time.time() - start_time
                rate = stats["successful"] / elapsed if elapsed > 0 else 0
                print(f"  {current_size:.2f}/{target_mb:.1f} MB | {stats['successful']} pairs | {rate:.2f}/sec")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        current_size = get_dir_size_mb(output_path)
        elapsed = time.time() - start_time
        
        stats["finished_at"] = datetime.now().isoformat()
        stats["final_size_mb"] = current_size
        stats["elapsed_sec"] = elapsed
        stats["total_files"] = len(list(output_path.glob("*.json")))
        
        with open(output_path / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDone: {current_size:.2f} MB | {stats['successful']} pairs | {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="/workspace/jit/data/cuda")
    parser.add_argument("-t", "--target", type=float, default=200.0)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-c", "--cache-clear", type=int, default=50)
    args = parser.parse_args()
    
    run_cuda_production(args.output, args.target, args.seed, args.cache_clear)
