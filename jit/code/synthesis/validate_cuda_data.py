#!/usr/bin/env python3
"""
Validate CUDA training data.

Checks that all generated pairs contain valid CUDA IR code.
"""
import json
import sys
from pathlib import Path
from collections import Counter


def validate_pair(filepath: Path) -> tuple[bool, str]:
    """
    Validate a single CUDA pair.
    
    Returns (is_valid, error_message)
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON parse error: {e}"
    
    # Check required fields
    if "python_source" not in data:
        return False, "Missing python_source"
    
    if "cuda_forward" not in data:
        return False, "Missing cuda_forward"
    
    cuda_code = data.get("cuda_forward", "")
    
    # Check for CUDA function signature
    if "_cuda_kernel_forward" not in cuda_code:
        return False, "Missing _cuda_kernel_forward in code"
    
    # Check for CUDA-specific patterns
    cuda_patterns = ["blockDim", "blockIdx", "threadIdx"]
    if not any(p in cuda_code for p in cuda_patterns):
        return False, "No CUDA thread indexing patterns found"
    
    # Check metadata
    metadata = data.get("metadata", {})
    if metadata.get("device") != "cuda":
        return False, f"Device is '{metadata.get('device')}', expected 'cuda'"
    
    return True, ""


def validate_dataset(data_dir: Path) -> dict:
    """Validate all pairs in a directory."""
    # Match cuda_NNNNNN.json files (6 digits), exclude stats file
    files = sorted([f for f in data_dir.glob("cuda_[0-9]*.json") 
                    if "stats" not in f.name])
    
    results = {
        "total": len(files),
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "categories": Counter(),
    }
    
    for f in files:
        is_valid, error = validate_pair(f)
        
        if is_valid:
            results["valid"] += 1
            # Count categories
            with open(f) as fp:
                data = json.load(fp)
                cat = data.get("metadata", {}).get("category", "unknown")
                results["categories"][cat] += 1
        else:
            results["invalid"] += 1
            results["errors"].append({"file": f.name, "error": error})
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CUDA training data")
    parser.add_argument("data_dir", nargs="?", default="/workspace/jit/data/cuda_training",
                        help="Directory containing CUDA data")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed error information")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return 1
    
    print(f"Validating CUDA data in: {data_dir}")
    print()
    
    results = validate_dataset(data_dir)
    
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Total files: {results['total']}")
    print(f"Valid: {results['valid']}")
    print(f"Invalid: {results['invalid']}")
    
    if results['total'] > 0:
        pct = results['valid'] / results['total'] * 100
        print(f"Success rate: {pct:.1f}%")
    
    print()
    print("Category distribution:")
    for cat, count in sorted(results['categories'].items()):
        pct = count / results['valid'] * 100 if results['valid'] > 0 else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    if results['errors'] and args.verbose:
        print()
        print("Errors:")
        for err in results['errors'][:10]:
            print(f"  {err['file']}: {err['error']}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")
    
    print()
    if results['invalid'] == 0:
        print("✓ All data is valid!")
        return 0
    else:
        print(f"✗ {results['invalid']} invalid files found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
