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
        required = ['python_source', 'cpp_code', 'kernel_name']
        for field in required:
            if field not in data or not data[field]:
                return False, f"Missing or empty field: {field}"
        
        # Check Python source
        py_src = data['python_source']
        if '@wp.kernel' not in py_src and 'def ' not in py_src:
            return False, "Python source doesn't look like a kernel"
        
        # Check C++ code
        cpp_code = data['cpp_code']
        if 'void ' not in cpp_code or '_cpu_kernel_forward' not in cpp_code:
            return False, "C++ code doesn't look like valid IR"
        
        # Check sizes are reasonable
        if len(py_src) < 20:
            return False, "Python source too short"
        
        if len(cpp_code) < 100:
            return False, "C++ code too short"
        
        return True, "OK"
        
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_random_samples(data_dir: str = "/workspace/data", sample_size: int = 20):
    """Validate random samples from dataset."""
    
    data_path = Path(data_dir)
    
    # Find all samples
    sample_files = list(data_path.rglob("sample_*.json"))
    sample_files.extend(data_path.rglob("test_*.json"))
    
    if len(sample_files) == 0:
        print("No samples found!")
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
    
    # Run validation
    success = validate_random_samples(sample_size=30)
    
    sys.exit(0 if success else 1)
