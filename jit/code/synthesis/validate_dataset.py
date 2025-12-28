"""
Validate dataset quality.

Checks all samples for completeness and consistency.
"""

import json
from pathlib import Path
from collections import Counter


def validate_dataset(data_dir: str):
    """Validate dataset quality."""
    data_path = Path(data_dir)
    
    print("="*60)
    print("DATASET VALIDATION")
    print("="*60)
    print()
    
    # Find all JSON metadata files
    json_files = sorted(data_path.glob("sample_*.json"))
    
    print(f"Found {len(json_files)} samples")
    print()
    
    # Validate each sample
    valid_samples = 0
    invalid_samples = 0
    categories = Counter()
    complexities = Counter()
    python_sizes = []
    cpp_sizes = []
    
    for json_file in json_files:
        with open(json_file) as f:
            metadata = json.load(f)
        
        sample_id = metadata['sample_id']
        
        # Check files exist
        py_file = data_path / metadata['python_file']
        cpp_file = data_path / metadata['cpp_file']
        
        if not py_file.exists():
            print(f"✗ Sample {sample_id}: Missing Python file")
            invalid_samples += 1
            continue
        
        if not cpp_file.exists():
            print(f"✗ Sample {sample_id}: Missing C++ file")
            invalid_samples += 1
            continue
        
        # Check file sizes match metadata
        py_size = py_file.stat().st_size
        cpp_size = cpp_file.stat().st_size
        
        if py_size != metadata['python_size']:
            print(f"✗ Sample {sample_id}: Python size mismatch")
            invalid_samples += 1
            continue
        
        if cpp_size != metadata['cpp_size']:
            print(f"✗ Sample {sample_id}: C++ size mismatch")
            invalid_samples += 1
            continue
        
        # Collect statistics
        categories[metadata['category']] += 1
        complexities[metadata['complexity']] += 1
        python_sizes.append(py_size)
        cpp_sizes.append(cpp_size)
        
        valid_samples += 1
    
    print(f"Valid samples: {valid_samples}")
    print(f"Invalid samples: {invalid_samples}")
    print()
    
    # Print statistics
    print("Category distribution:")
    for cat, count in categories.most_common():
        print(f"  {cat:15s}: {count:3d} ({count/valid_samples*100:5.1f}%)")
    print()
    
    print("Complexity distribution:")
    for comp, count in complexities.most_common():
        print(f"  {comp:10s}: {count:3d} ({count/valid_samples*100:5.1f}%)")
    print()
    
    print("File sizes:")
    print(f"  Python: min={min(python_sizes):5d}, max={max(python_sizes):5d}, avg={sum(python_sizes)//len(python_sizes):5d} bytes")
    print(f"  C++:    min={min(cpp_sizes):5d}, max={max(cpp_sizes):5d}, avg={sum(cpp_sizes)//len(cpp_sizes):5d} bytes")
    print()
    
    # Check for duplicates (by kernel name)
    kernel_names = [json.load(open(f))['kernel_name'] for f in json_files]
    duplicates = len(kernel_names) - len(set(kernel_names))
    print(f"Duplicate kernel names: {duplicates}")
    print()
    
    print("="*60)
    if invalid_samples == 0:
        print("✓ ALL SAMPLES VALID")
    else:
        print(f"✗ {invalid_samples} SAMPLES INVALID")
    print("="*60)
    
    return valid_samples, invalid_samples


def main():
    valid, invalid = validate_dataset("/workspace/data/samples")
    
    if invalid == 0:
        print("\n✓ Dataset validation passed")
        return 0
    else:
        print(f"\n✗ Dataset validation failed ({invalid} invalid samples)")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
