#!/usr/bin/env python3
"""
Fast Data Production Script
Generates Python→IR training pairs with cache management for consistent speed.
"""
import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import importlib.util
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
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


def compile_and_extract(python_source: str, kernel_name: str, device: str = "cpu"):
    """Compile kernel and extract IR using subprocess for isolation."""
    import warp as wp
    
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
        wp_module.load(device)
        
        module_id = wp_module.get_module_identifier()
        cache_dir = Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))
        
        ext = "cpp" if device == "cpu" else "cu"
        code_file = cache_dir / module_id / f"{module_id}.{ext}"
        
        if not code_file.exists():
            raise FileNotFoundError(f"Generated code not found: {code_file}")
        
        code_full = code_file.read_text()
        
        import re
        suffix = "cpu" if device == "cpu" else "cuda"
        
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
        
        forward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_{suffix}_kernel_forward\('
        backward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_{suffix}_kernel_backward\('
        
        return {
            "forward": extract_function(forward_pattern, code_full),
            "backward": extract_function(backward_pattern, code_full),
            "module_id": module_id
        }
        
    finally:
        os.unlink(temp_path)
        if "temp_kernel" in sys.modules:
            del sys.modules["temp_kernel"]


def run_production(output_dir: str, target_mb: float = 200.0, device: str = "cpu", 
                   seed: int = 42, cache_clear_interval: int = 50):
    """Run production with periodic cache clearing."""
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
    print(f"{device.upper()} Data Production (Fast)")
    print("=" * 60)
    print(f"Target: {target_mb:.1f} MB | Current: {current_size:.2f} MB")
    print(f"Existing: {start_index} files | Cache clear every {cache_clear_interval}")
    print()
    
    if current_size >= target_mb:
        print(f"Already at target!")
        return
    
    stats = {
        "started_at": datetime.now().isoformat(),
        "device": device,
        "target_mb": target_mb,
        "successful": 0,
        "failed": 0,
        "by_type": {k: 0 for k in kernel_types}
    }
    
    file_index = start_index
    start_time = time.time()
    since_clear = 0
    
    # Clear cache at start for consistent performance
    clear_warp_cache()
    
    try:
        while current_size < target_mb:
            # Clear cache periodically
            if since_clear >= cache_clear_interval:
                clear_warp_cache()
                # Re-init warp after cache clear
                import importlib
                importlib.reload(wp)
                wp.init()
                since_clear = 0
                print(f"  [Cache cleared]")
            
            ktype = kernel_types[file_index % len(kernel_types)]
            
            try:
                spec = generator.generate(ktype)
                python_source = generator.to_python_source(spec)
                
                ir_data = compile_and_extract(python_source, spec.name, device)
                
                if not ir_data["forward"]:
                    raise ValueError("No forward IR extracted")
                
                pair_id = hash_source(python_source)
                pair = {
                    "id": pair_id,
                    "kernel_name": spec.name,
                    "kernel_type": ktype,
                    "python_source": python_source,
                    "ir_forward": ir_data["forward"],
                    "ir_backward": ir_data["backward"],
                    "device": device,
                    "generated_at": datetime.now().isoformat(),
                    "metadata": {
                        "num_params": len(spec.params),
                        "num_lines": len(spec.body_lines),
                        "module_id": ir_data["module_id"]
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
    parser.add_argument("-o", "--output", default="/workspace/jit/data/cpu")
    parser.add_argument("-t", "--target", type=float, default=200.0)
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-c", "--cache-clear", type=int, default=50)
    args = parser.parse_args()
    
    run_production(args.output, args.target, args.device, args.seed, args.cache_clear)
