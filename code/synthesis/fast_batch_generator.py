"""Fast batch generator for large-scale Python→IR pair generation."""
import os
import sys
import json
import hashlib
import tempfile
import importlib.util
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from generator import KernelGenerator, KernelSpec


KERNEL_TYPES = [
    "arithmetic", "conditional", "loop", "math", "vector",
    "atomic", "nested", "multi_cond", "combined", "scalar_param"
]


def hash_source(source: str) -> str:
    """Generate a short hash of the source code."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def extract_function(pattern_start: str, code: str) -> str:
    """Extract a function from C++ code."""
    import re
    match = re.search(pattern_start, code)
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


def generate_batch(
    n: int,
    output_dir: str,
    seed: int = 42,
    kernels_per_module: int = 10,
    progress_interval: int = 100
) -> Dict[str, Any]:
    """Generate n Python→IR pairs efficiently.
    
    Groups multiple kernels per module to reduce compilation overhead.
    """
    import warp as wp
    import re
    
    wp.init()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    gen = KernelGenerator(seed=seed)
    
    stats = {
        "total_attempted": 0,
        "successful": 0,
        "failed": 0,
        "by_type": {kt: 0 for kt in KERNEL_TYPES},
        "start_time": time.time()
    }
    
    print(f"Generating {n} pairs to {output_dir}")
    print(f"Kernels per module: {kernels_per_module}")
    
    remaining = n
    batch_num = 0
    
    while remaining > 0:
        batch_size = min(kernels_per_module, remaining)
        
        # Generate kernel specs for this batch
        specs = []
        sources = []
        kernel_types = []
        
        for _ in range(batch_size):
            ktype = random.choice(KERNEL_TYPES)
            spec = gen.generate(ktype)
            source = gen.to_python_source(spec)
            specs.append(spec)
            sources.append(source)
            kernel_types.append(ktype)
        
        # Create module with all kernels
        module_source = "import warp as wp\n\n" + "\n\n".join(sources)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(module_source)
            temp_path = f.name
        
        try:
            # Load and compile module
            module_name = f"batch_module_{batch_num}"
            spec_loader = importlib.util.spec_from_file_location(module_name, temp_path)
            module = importlib.util.module_from_spec(spec_loader)
            sys.modules[module_name] = module
            spec_loader.loader.exec_module(module)
            
            # Extract IR for each kernel
            for i, (kspec, src, ktype) in enumerate(zip(specs, sources, kernel_types)):
                stats["total_attempted"] += 1
                
                try:
                    kernel = getattr(module, kspec.name, None)
                    if kernel is None:
                        stats["failed"] += 1
                        continue
                    
                    # Force compilation
                    wp_module = kernel.module
                    wp_module.load("cpu")
                    
                    # Read generated C++
                    module_id = wp_module.get_module_identifier()
                    cache_dir = Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))
                    cpp_file = cache_dir / module_id / f"{module_id}.cpp"
                    
                    if not cpp_file.exists():
                        stats["failed"] += 1
                        continue
                    
                    cpp_full = cpp_file.read_text()
                    
                    # Extract forward and backward IR
                    kernel_key = kernel.key
                    forward_pattern = rf'void\s+{re.escape(kernel_key)}_[a-f0-9]+_cpu_kernel_forward\('
                    backward_pattern = rf'void\s+{re.escape(kernel_key)}_[a-f0-9]+_cpu_kernel_backward\('
                    
                    forward_ir = extract_function(forward_pattern, cpp_full)
                    backward_ir = extract_function(backward_pattern, cpp_full)
                    
                    if not forward_ir:
                        stats["failed"] += 1
                        continue
                    
                    # Save pair
                    pair_id = hash_source(src)
                    data = {
                        "id": pair_id,
                        "kernel_name": kspec.name,
                        "kernel_type": ktype,
                        "python_source": src,
                        "cpp_ir_forward": forward_ir,
                        "cpp_ir_backward": backward_ir,
                        "generated_at": datetime.now().isoformat(),
                        "metadata": {
                            "num_params": len(kspec.params),
                            "num_lines": len(kspec.body_lines),
                            "module_id": module_id,
                            "device": "cpu"
                        }
                    }
                    
                    filename = f"{pair_id}_{kspec.name}.json"
                    filepath = output_path / filename
                    
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    stats["successful"] += 1
                    stats["by_type"][ktype] += 1
                    
                except Exception as e:
                    stats["failed"] += 1
            
            del sys.modules[module_name]
            
        except Exception as e:
            print(f"Batch {batch_num} failed: {e}")
            stats["failed"] += batch_size
            stats["total_attempted"] += batch_size
        
        finally:
            os.unlink(temp_path)
        
        remaining -= batch_size
        batch_num += 1
        
        # Progress update
        if stats["total_attempted"] % progress_interval == 0:
            elapsed = time.time() - stats["start_time"]
            rate = stats["successful"] / elapsed if elapsed > 0 else 0
            pct = (n - remaining) / n * 100
            print(f"  Progress: {stats['successful']}/{n} ({pct:.1f}%) - {rate:.1f} pairs/sec")
    
    stats["end_time"] = time.time()
    stats["elapsed_seconds"] = stats["end_time"] - stats["start_time"]
    stats["pairs_per_second"] = stats["successful"] / stats["elapsed_seconds"]
    
    # Save stats
    stats_file = output_path / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def print_stats(stats: Dict[str, Any]):
    """Print generation statistics."""
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total attempted: {stats['total_attempted']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['successful'] / max(1, stats['total_attempted']) * 100:.1f}%")
    print(f"Time: {stats['elapsed_seconds']:.1f}s")
    print(f"Rate: {stats['pairs_per_second']:.1f} pairs/sec")
    print("\nBy type:")
    for ktype, count in sorted(stats["by_type"].items()):
        if count > 0:
            print(f"  {ktype}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast batch generator for Python→IR pairs")
    parser.add_argument("-n", type=int, default=1000, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", type=str, default="data/cpu", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kernels-per-module", type=int, default=10, help="Kernels per compilation batch")
    
    args = parser.parse_args()
    
    stats = generate_batch(
        n=args.n,
        output_dir=args.output,
        seed=args.seed,
        kernels_per_module=args.kernels_per_module
    )
    
    print_stats(stats)
