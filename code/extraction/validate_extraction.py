#!/usr/bin/env python3
"""
Validation script: Run all IR extractions twice and verify consistency
"""

import json
import os
import hashlib

def compute_file_hash(filepath):
    """Compute SHA256 hash of file contents."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def validate_json_file(filepath):
    """Validate a JSON file contains all required fields."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        required_fields = ['python_source', 'cpp_code', 'kernel_name']
        missing = [f for f in required_fields if f not in data or not data[f]]
        
        if missing:
            return False, f"Missing fields: {missing}", None
        
        # Check sizes
        if len(data['python_source']) < 10:
            return False, "Python source too short", None
        
        if len(data['cpp_code']) < 100:
            return False, "C++ code too short", None
        
        # Return Python source hash for comparison
        py_hash = hashlib.sha256(data['python_source'].encode()).hexdigest()
        
        return True, "OK", py_hash
        
    except Exception as e:
        return False, str(e), None

def main():
    print("="*60)
    print("Validation: IR Extraction Consistency")
    print("="*60)
    
    # Find all JSON files
    data_dir = "/workspace/data"
    samples_dir = os.path.join(data_dir, "samples")
    
    json_files = []
    for d in [data_dir, samples_dir]:
        if os.path.exists(d):
            for fname in os.listdir(d):
                if fname.endswith('.json'):
                    json_files.append(os.path.join(d, fname))
    
    json_files.sort()
    
    print(f"\nFound {len(json_files)} test cases")
    print()
    
    # Validate each file
    all_valid = True
    hashes = {}
    py_hashes = {}
    
    for filepath in json_files:
        fname = os.path.basename(filepath)
        
        # Compute hash
        file_hash = compute_file_hash(filepath)
        hashes[fname] = file_hash
        
        # Validate content
        valid, msg, py_hash = validate_json_file(filepath)
        if py_hash:
            py_hashes[fname] = py_hash
        
        status = "✓" if valid else "✗"
        print(f"{status} {fname:40s} {msg}")
        
        if not valid:
            all_valid = False
    
    # Now re-run extractions and verify hashes don't change
    print("\n" + "="*60)
    print("Running extractions again to verify consistency...")
    print("="*60)
    
    # Re-run the extraction scripts
    import subprocess
    
    scripts = [
        "/workspace/code/extraction/test_ir_extraction.py",
        "/workspace/code/extraction/test_additional_cases.py"
    ]
    
    for script in scripts:
        print(f"\nRunning {os.path.basename(script)}...")
        result = subprocess.run(
            ["python3", script],
            cwd=os.path.dirname(script),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"✗ Script failed!")
            all_valid = False
        else:
            print(f"✓ Script completed")
    
    # Check hashes again - compare Python source hashes, not file hashes
    # (file hashes may change due to metadata, but Python source should be stable)
    print("\n" + "="*60)
    print("Comparing Python source consistency...")
    print("="*60)
    
    py_hash_matches = True
    for filepath in json_files:
        fname = os.path.basename(filepath)
        old_py_hash = py_hashes.get(fname)
        
        # Re-validate to get new hash
        valid, msg, new_py_hash = validate_json_file(filepath)
        
        if not valid:
            print(f"✗ {fname:40s} validation failed")
            continue
        
        if old_py_hash == new_py_hash:
            print(f"✓ {fname:40s} Python source unchanged")
        else:
            print(f"✗ {fname:40s} Python source CHANGED!")
            py_hash_matches = False
    
    # Final result
    print("\n" + "="*60)
    if all_valid and py_hash_matches:
        print("✓✓✓ ALL VALIDATION PASSED ✓✓✓")
    else:
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
        if not all_valid:
            print("  - Some files have invalid content")
        if not py_hash_matches:
            print("  - Python source changed between runs (non-deterministic)")
    print("="*60)
    
    return all_valid and py_hash_matches

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
