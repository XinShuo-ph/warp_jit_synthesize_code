#!/usr/bin/env python3
"""
Isolated Data Production Script
Runs each generation batch in a completely fresh subprocess to avoid state buildup.
"""
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


def get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB."""
    total = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    return total / (1024 * 1024)


def clear_warp_cache():
    """Clear warp compilation cache completely."""
    cache_dir = Path(os.path.expanduser("~/.cache/warp"))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def run_batch(output_dir: str, device: str, seed: int, batch_size: int = 10):
    """Run a single batch generation in this process and return results."""
    import warp as wp
    wp.init()
    
    sys.path.insert(0, str(Path(__file__).parent))
    from generator import KernelGenerator
    import hashlib
    import tempfile
    import importlib.util
    import re
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = KernelGenerator(seed=seed)
    kernel_types = ["arithmetic", "conditional", "loop", "math", "vector", 
                    "atomic", "nested", "multi_cond", "combined", "scalar_param"]
    
    results = {"success": 0, "failed": 0, "bytes": 0}
    
    for i in range(batch_size):
        ktype = kernel_types[(seed + i) % len(kernel_types)]
        
        try:
            spec = generator.generate(ktype)
            python_source = generator.to_python_source(spec)
            
            # Compile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f'"""Temp."""\nimport warp as wp\n\n{python_source}')
                temp_path = f.name
            
            try:
                spec_mod = importlib.util.spec_from_file_location("temp_kernel", temp_path)
                module = importlib.util.module_from_spec(spec_mod)
                sys.modules["temp_kernel"] = module
                spec_mod.loader.exec_module(module)
                
                kernel = getattr(module, spec.name)
                wp_module = kernel.module
                wp_module.load(device)
                
                # Extract IR
                module_id = wp_module.get_module_identifier()
                cache_dir = Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))
                ext = "cpp" if device == "cpu" else "cu"
                code_file = cache_dir / module_id / f"{module_id}.{ext}"
                
                if not code_file.exists():
                    raise FileNotFoundError(f"No code at {code_file}")
                
                code_full = code_file.read_text()
                
                suffix = "cpu" if device == "cpu" else "cuda"
                
                def extract_function(pattern, code):
                    match = re.search(pattern, code)
                    if not match:
                        return ""
                    start = match.start()
                    brace_count = 0
                    for idx, c in enumerate(code[start:]):
                        if c == '{':
                            brace_count += 1
                        elif c == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                return code[start:start + idx + 1]
                    return ""
                
                fwd_pat = rf'void\s+{re.escape(spec.name)}_[a-f0-9]+_{suffix}_kernel_forward\('
                bwd_pat = rf'void\s+{re.escape(spec.name)}_[a-f0-9]+_{suffix}_kernel_backward\('
                
                ir_forward = extract_function(fwd_pat, code_full)
                ir_backward = extract_function(bwd_pat, code_full)
                
                if not ir_forward:
                    raise ValueError("No forward IR")
                
                pair_id = hashlib.sha256(python_source.encode()).hexdigest()[:12]
                pair = {
                    "id": pair_id,
                    "kernel_name": spec.name,
                    "kernel_type": ktype,
                    "python_source": python_source,
                    "ir_forward": ir_forward,
                    "ir_backward": ir_backward,
                    "device": device,
                    "generated_at": datetime.now().isoformat(),
                    "metadata": {
                        "num_params": len(spec.params),
                        "num_lines": len(spec.body_lines),
                        "module_id": module_id
                    }
                }
                
                filename = f"{pair_id}_{spec.name}.json"
                filepath = output_path / filename
                with open(filepath, 'w') as f:
                    json.dump(pair, f, indent=2)
                
                results["success"] += 1
                results["bytes"] += filepath.stat().st_size
                
            finally:
                os.unlink(temp_path)
                if "temp_kernel" in sys.modules:
                    del sys.modules["temp_kernel"]
                    
        except Exception as e:
            results["failed"] += 1
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="/workspace/jit/data/cpu")
    parser.add_argument("-t", "--target", type=float, default=200.0)
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-b", "--batch", type=int, default=15, help="Kernels per subprocess")
    parser.add_argument("--internal-batch", action="store_true", help="Internal: run as batch worker")
    args = parser.parse_args()
    
    # If running as internal batch worker
    if args.internal_batch:
        result = run_batch(args.output, args.device, args.seed, args.batch)
        print(json.dumps(result))
        return
    
    # Main orchestrator
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(output_path.glob("*.json")))
    current_size = get_dir_size_mb(output_path)
    
    print("=" * 60)
    print(f"{args.device.upper()} Data Production (Isolated Subprocesses)")
    print("=" * 60)
    print(f"Target: {args.target:.1f} MB | Current: {current_size:.2f} MB")
    print(f"Existing: {existing} files | Batch size: {args.batch}")
    print()
    
    if current_size >= args.target:
        print("Already at target!")
        return
    
    seed = args.seed
    total_success = 0
    total_failed = 0
    start_time = time.time()
    
    try:
        while current_size < args.target:
            # Clear cache before each batch
            clear_warp_cache()
            
            # Run batch in subprocess
            cmd = [
                sys.executable, __file__,
                "-o", args.output,
                "-d", args.device,
                "-s", str(seed),
                "-b", str(args.batch),
                "--internal-batch"
            ]
            
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, 
                    timeout=300  # 5 minute timeout per batch
                )
                
                if result.returncode == 0:
                    # Parse result from last line
                    lines = result.stdout.strip().split('\n')
                    for line in reversed(lines):
                        if line.startswith('{'):
                            data = json.loads(line)
                            total_success += data.get("success", 0)
                            total_failed += data.get("failed", 0)
                            break
                else:
                    total_failed += args.batch
                    
            except subprocess.TimeoutExpired:
                total_failed += args.batch
            except Exception as e:
                total_failed += args.batch
            
            seed += args.batch
            
            # Update stats
            current_size = get_dir_size_mb(output_path)
            file_count = len(list(output_path.glob("*.json")))
            elapsed = time.time() - start_time
            rate = total_success / elapsed if elapsed > 0 else 0
            
            print(f"  {current_size:.2f}/{args.target:.1f} MB | {file_count} files | {rate:.2f}/sec")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        current_size = get_dir_size_mb(output_path)
        file_count = len(list(output_path.glob("*.json")))
        elapsed = time.time() - start_time
        print(f"\nDone: {current_size:.2f} MB | {file_count} files | {elapsed:.1f}s")


if __name__ == "__main__":
    main()
