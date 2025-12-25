"""End-to-end pipeline for generating Python→IR training pairs."""
import sys
import os
import json
import hashlib
import tempfile
import importlib.util
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

from generator import GENERATORS, generate_random_kernel


def _write_temp_kernel(source: str) -> tuple:
    """Write kernel source to temp file and return path and kernel name."""
    # Extract kernel name from source
    kernel_name = None
    for line in source.split('\n'):
        if line.strip().startswith('def '):
            kernel_name = line.split('(')[0].replace('def ', '').strip()
            break
    
    if kernel_name is None:
        raise ValueError("Could not find kernel name in source")
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import warp as wp\n\n")
        f.write(source)
        temp_path = f.name
    
    return temp_path, kernel_name


def _load_and_compile_kernel(temp_path: str, kernel_name: str):
    """Load and compile kernel from temp file."""
    import warp as wp
    
    spec = importlib.util.spec_from_file_location("temp_kernel_mod", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    kernel = getattr(module, kernel_name)
    return kernel


def _extract_ir_from_kernel(kernel, device: str = "cpu") -> str:
    """Extract IR from compiled kernel."""
    import warp as wp
    import numpy as np
    
    # Force compile by launching with minimal data
    args = []
    for arg in kernel.adj.args:
        arg_type = arg.type
        if hasattr(arg_type, 'dtype'):
            arr = wp.zeros(1, dtype=arg_type.dtype, device=device)
            args.append(arr)
        else:
            args.append(arg_type())
    
    try:
        wp.launch(kernel, dim=1, inputs=args, device=device)
        wp.synchronize_device(wp.get_device(device))
    except Exception:
        pass
    
    # Get IR from cache
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "warp", wp.__version__)
    module = kernel.module
    ext = ".cpp" if device == "cpu" else ".cu"
    
    # Iterate over all execs to find the one for our device
    for exec_key, exec_obj in module.execs.items():
        if exec_obj.device.is_cpu if device == "cpu" else exec_obj.device.is_cuda:
            module_hash = exec_obj.module_hash.hex()
            cache_subdir = os.path.join(cache_dir, f"wp_{module.name}_{module_hash[:7]}")
            
            if os.path.isdir(cache_subdir):
                for fname in os.listdir(cache_subdir):
                    if fname.endswith(ext):
                        with open(os.path.join(cache_subdir, fname), 'r') as f:
                            return f.read()
    
    return ""


def generate_pair(kernel_type: str = None, device: str = "cpu") -> dict:
    """Generate a single Python→IR pair.
    
    Args:
        kernel_type: Specific kernel type or None for random
        device: Target device ("cpu" or "cuda")
        
    Returns:
        Dictionary with pair data or None if failed
    """
    import warp as wp
    
    # Generate kernel source
    ktype, source = generate_random_kernel(kernel_type)
    
    # Write to temp file
    temp_path, kernel_name = _write_temp_kernel(source)
    
    try:
        # Load and compile
        kernel = _load_and_compile_kernel(temp_path, kernel_name)
        
        # Extract IR
        ir_code = _extract_ir_from_kernel(kernel, device)
        
        if not ir_code:
            return None
        
        # Create pair
        pair = {
            "kernel_name": kernel_name,
            "kernel_type": ktype,
            "python_source": source,
            "ir_code": ir_code,
            "device": device,
            "ir_length": len(ir_code),
            "source_length": len(source),
        }
        
        # Add hash for deduplication
        pair["hash"] = hashlib.md5(source.encode()).hexdigest()[:8]
        
        return pair
    
    except Exception as e:
        return None
    
    finally:
        os.unlink(temp_path)


def generate_batch(count: int, output_dir: str = None, quiet: bool = False) -> list:
    """Generate a batch of Python→IR pairs.
    
    Args:
        count: Number of pairs to generate
        output_dir: Directory to save JSON files (None = don't save)
        quiet: Suppress progress output
        
    Returns:
        List of successfully generated pairs
    """
    import warp as wp
    wp.init()
    
    pairs = []
    failed = 0
    
    # Ensure balanced types
    kernel_types = list(GENERATORS.keys())
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for i in range(count):
        # Cycle through kernel types for balance
        kernel_type = kernel_types[i % len(kernel_types)]
        
        pair = generate_pair(kernel_type)
        
        if pair:
            pairs.append(pair)
            
            if output_dir:
                # Save to file
                filename = f"{pair['kernel_type']}_{pair['hash']}.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(pair, f, indent=2)
            
            if not quiet:
                print(f"[{len(pairs):4d}/{count}] Generated {pair['kernel_type']:12s} kernel: {pair['kernel_name']}")
        else:
            failed += 1
            if not quiet:
                print(f"[{len(pairs):4d}/{count}] Failed to generate kernel")
    
    if not quiet:
        print(f"\nGenerated {len(pairs)} pairs ({failed} failed)")
    
    return pairs


def validate_pairs(pairs: list) -> tuple:
    """Validate generated pairs.
    
    Args:
        pairs: List of pair dictionaries
        
    Returns:
        Tuple of (valid_count, invalid_count, issues)
    """
    valid = 0
    invalid = 0
    issues = []
    
    for pair in pairs:
        if not pair.get('python_source'):
            issues.append(f"{pair.get('kernel_name', 'unknown')}: missing python_source")
            invalid += 1
        elif not pair.get('ir_code'):
            issues.append(f"{pair.get('kernel_name', 'unknown')}: missing ir_code")
            invalid += 1
        elif len(pair.get('ir_code', '')) < 100:
            issues.append(f"{pair.get('kernel_name', 'unknown')}: IR too short")
            invalid += 1
        else:
            valid += 1
    
    return valid, invalid, issues


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()
    
    print("="*60)
    print("Python→IR Synthesis Pipeline")
    print("="*60)
    
    pairs = generate_batch(args.count, args.output, args.quiet)
    
    # Validate
    valid, invalid, issues = validate_pairs(pairs)
    print(f"\nValidation: {valid} valid, {invalid} invalid")
    
    if issues:
        print("Issues:")
        for issue in issues[:5]:
            print(f"  - {issue}")
