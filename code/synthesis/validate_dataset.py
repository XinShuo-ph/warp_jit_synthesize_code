#!/usr/bin/env python3
"""
Validate random samples from the dataset.
"""

import json
import random
from pathlib import Path

def validate_sample(filepath: Path) -> tuple[bool, str]:
    """Validate a single sample file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required = ["python_source", "kernel_name", "kernel_type", "cpp_ir_forward"]
        for field in required:
            if field not in data or not data[field]:
                return False, f"Missing or empty field: {field}"
        
        # Check Python source
        py_src = data['python_source']
        if '@wp.kernel' not in py_src and 'def ' not in py_src:
            return False, "Python source doesn't look like a kernel"
        
        # Check generated source (forward IR snippet)
        device = (data.get("device") or data.get("metadata", {}).get("device") or "cpu").lower()
        cpp_forward = data["cpp_ir_forward"]
        if "void " not in cpp_forward:
            return False, "Forward IR doesn't look like C/C++ code"
        if f"_{device}_kernel_forward" not in cpp_forward:
            return False, f"Forward IR missing expected suffix _{device}_kernel_forward"
        
        # Check sizes are reasonable
        if len(py_src) < 20:
            return False, "Python source too short"
        
        if len(cpp_forward) < 100:
            return False, "Forward IR too short"
        
        return True, "OK"
        
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_random_samples(data_dir: str = "data", sample_size: int = 20, device: str | None = None):
    """Validate random samples from dataset."""
    
    data_path = Path(data_dir)
    
    # Find all samples
    sample_files = list(data_path.rglob("*.json"))
    
    if len(sample_files) == 0:
        print("No samples found!")
        return False
    
    if device:
        device = device.lower()
        filtered = []
        for fp in sample_files:
            try:
                d = json.loads(fp.read_text())
                ddev = (d.get("device") or d.get("metadata", {}).get("device") or "cpu").lower()
                if ddev == device:
                    filtered.append(fp)
            except Exception:
                continue
        sample_files = filtered

    if len(sample_files) == 0:
        print("No samples found (after device filter)!")
        return False

    print(f"Total samples available: {len(sample_files)}")
    
    # Select random samples
    sample_size = min(sample_size, len(sample_files))
    selected = random.sample(sample_files, sample_size)
    
    print(f"Validating {sample_size} random samples...")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for filepath in selected:
        valid, msg = validate_sample(filepath)
        
        status = "✓" if valid else "✗"
        print(f"{status} {filepath.name:30s} {msg}")
        
        if valid:
            passed += 1
        else:
            failed += 1
    
    print("="*60)
    print(f"Results: {passed}/{sample_size} passed")
    
    if failed == 0:
        print("✓✓✓ All samples valid! ✓✓✓")
        return True
    else:
        print(f"✗✗✗ {failed} samples failed validation ✗✗✗")
        return False


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate a random subset of JSON samples")
    parser.add_argument("data_dir", nargs="?", default="data", help="Dataset directory (default: data)")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of random files to validate")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Only validate samples for this device")
    args = parser.parse_args()

    # Run validation
    success = validate_random_samples(data_dir=args.data_dir, sample_size=args.sample_size, device=args.device)
    
    sys.exit(0 if success else 1)
