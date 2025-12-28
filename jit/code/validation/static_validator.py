"""
Static Validator for Generated IR Datasets.
Checks structure and content without requiring execution.
"""
import json
import argparse
from pathlib import Path
import sys

def validate_pair(filepath: Path) -> list[str]:
    """Validate a single JSON pair file. Returns list of error messages."""
    errors = []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return [f"Invalid JSON: {e}"]
    
    # Check required fields
    required_fields = ["python_source", "cpp_forward", "metadata"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing field: {field}")
            
    if errors:
        return errors
        
    # Check metadata
    metadata = data["metadata"]
    if "device" not in metadata:
        errors.append("Missing 'device' in metadata")
    elif metadata["device"] != "cuda":
        errors.append(f"Incorrect device: expected 'cuda', got '{metadata['device']}'")
        
    # Check content
    cpp_code = data["cpp_forward"]
    if not cpp_code:
        errors.append("Empty cpp_forward code")
    elif 'extern "C" __global__' not in cpp_code:
        errors.append("Missing CUDA kernel signature (extern \"C\" __global__)")
        
    if "arg_types" not in metadata:
        errors.append("Missing 'arg_types' in metadata")
        
    return errors

def validate_dataset(data_dir: Path) -> bool:
    print(f"Validating dataset in {data_dir}...")
    
    files = list(data_dir.glob("*.json"))
    stats_files = list(data_dir.glob("*stats.json"))
    data_files = [f for f in files if f not in stats_files]
    
    if not data_files:
        print("❌ No data files found.")
        return False
    
    print(f"Found {len(data_files)} data files.")
    
    passed = 0
    failed = 0
    errors_by_file = {}
    
    for f in data_files:
        errors = validate_pair(f)
        if not errors:
            passed += 1
        else:
            failed += 1
            errors_by_file[f.name] = errors
            
    print(f"\nValidation Results:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailures:")
        for name, errs in list(errors_by_file.items())[:10]: # Show first 10
            print(f"  {name}: {', '.join(errs)}")
        if failed > 10:
            print(f"  ... and {failed - 10} more.")
        return False
        
    print("\n✅ Dataset is valid.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing generated JSON files")
    args = parser.parse_args()
    
    if not validate_dataset(Path(args.data_dir)):
        sys.exit(1)
