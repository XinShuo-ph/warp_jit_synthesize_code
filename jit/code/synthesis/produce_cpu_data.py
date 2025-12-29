#!/usr/bin/env python3
"""
CPU Data Production Script
Generates ~200MB of Python→IR training pairs for CPU backend.

Target: 200MB ≈ 20,000-25,000 pairs (average ~8-10KB each)
"""
import os
import sys
import json
import time
import hashlib
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

from generator import KernelGenerator


def get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB."""
    total = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    return total / (1024 * 1024)


def hash_source(source: str) -> str:
    """Generate a short hash of the source code."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def compile_and_extract(python_source: str, kernel_name: str, device: str = "cpu"):
    """Compile kernel and extract IR."""
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
        
        # Get cache path and read generated code
        module_id = wp_module.get_module_identifier()
        cache_dir = Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))
        
        if device == "cpu":
            cpp_file = cache_dir / module_id / f"{module_id}.cpp"
        else:
            cpp_file = cache_dir / module_id / f"{module_id}.cu"
        
        if not cpp_file.exists():
            raise FileNotFoundError(f"Generated code not found: {cpp_file}")
        
        code_full = cpp_file.read_text()
        
        # Extract forward function
        import re
        suffix = "cpu" if device == "cpu" else "cuda"
        forward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_{suffix}_kernel_forward\('
        
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
        
        forward_ir = extract_function(forward_pattern, code_full)
        
        # Also try backward
        backward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_{suffix}_kernel_backward\('
        backward_ir = extract_function(backward_pattern, code_full)
        
        return {
            "forward": forward_ir,
            "backward": backward_ir,
            "full": code_full,
            "module_id": module_id
        }
        
    finally:
        os.unlink(temp_path)
        if "temp_kernel" in sys.modules:
            del sys.modules["temp_kernel"]


def run_production(output_dir: str, target_mb: float = 200.0, seed: int = 42):
    """Run production to generate target_mb of data."""
    import warp as wp
    wp.init()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = KernelGenerator(seed=seed)
    kernel_types = ["arithmetic", "conditional", "loop", "math", "vector", 
                    "atomic", "nested", "multi_cond", "combined", "scalar_param"]
    
    # Check existing data
    existing_files = list(output_path.glob("*.json"))
    start_index = len(existing_files)
    current_size = get_dir_size_mb(output_path)
    
    print("=" * 60)
    print("CPU Data Production")
    print("=" * 60)
    print(f"Target size: {target_mb:.1f} MB")
    print(f"Current size: {current_size:.2f} MB")
    print(f"Existing files: {start_index}")
    print(f"Output: {output_dir}")
    print()
    
    if current_size >= target_mb:
        print(f"Already at target size ({current_size:.2f} MB >= {target_mb:.1f} MB)")
        return
    
    stats = {
        "started_at": datetime.now().isoformat(),
        "target_mb": target_mb,
        "successful": 0,
        "failed": 0,
        "by_type": {k: 0 for k in kernel_types}
    }
    
    file_index = start_index
    start_time = time.time()
    
    try:
        while current_size < target_mb:
            ktype = kernel_types[file_index % len(kernel_types)]
            
            try:
                spec = generator.generate(ktype)
                python_source = generator.to_python_source(spec)
                
                ir_data = compile_and_extract(python_source, spec.name, "cpu")
                
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
                    "device": "cpu",
                    "generated_at": datetime.now().isoformat(),
                    "metadata": {
                        "num_params": len(spec.params),
                        "num_lines": len(spec.body_lines),
                        "module_id": ir_data["module_id"]
                    }
                }
                
                filename = f"{pair_id}_{spec.name}.json"
                filepath = output_path / filename
                
                with open(filepath, 'w') as f:
                    json.dump(pair, f, indent=2)
                
                stats["successful"] += 1
                stats["by_type"][ktype] += 1
                
            except Exception as e:
                stats["failed"] += 1
                print(f"  Failed ({ktype}): {e}")
            
            file_index += 1
            
            # Progress update every 100 files
            if file_index % 100 == 0:
                current_size = get_dir_size_mb(output_path)
                elapsed = time.time() - start_time
                rate = stats["successful"] / elapsed if elapsed > 0 else 0
                
                print(f"  Progress: {current_size:.2f}/{target_mb:.1f} MB | "
                      f"{stats['successful']} pairs | {rate:.1f}/sec")
                
                # Save stats periodically
                stats["current_size_mb"] = current_size
                stats["elapsed_sec"] = elapsed
                with open(output_path / "production_stats.json", 'w') as f:
                    json.dump(stats, f, indent=2)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Final stats
        current_size = get_dir_size_mb(output_path)
        elapsed = time.time() - start_time
        
        stats["finished_at"] = datetime.now().isoformat()
        stats["final_size_mb"] = current_size
        stats["total_elapsed_sec"] = elapsed
        stats["total_files"] = len(list(output_path.glob("*.json")))
        
        with open(output_path / "production_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Production Summary")
        print("=" * 60)
        print(f"Final size: {current_size:.2f} MB")
        print(f"Total pairs: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Time: {elapsed:.1f}s")
        if elapsed > 0:
            print(f"Rate: {stats['successful'] / elapsed:.1f} pairs/sec")
        print("\nBy type:")
        for ktype, count in sorted(stats["by_type"].items()):
            print(f"  {ktype}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CPU training data")
    parser.add_argument("-o", "--output", default="/workspace/jit/data/cpu",
                        help="Output directory")
    parser.add_argument("-t", "--target", type=float, default=200.0,
                        help="Target size in MB")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    run_production(args.output, args.target, args.seed)
