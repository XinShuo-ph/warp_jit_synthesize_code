"""
Validation script for generated CUDA kernels.

Checks:
1. JSON structure validity
2. Python source code syntax
3. CUDA IR patterns
4. Metadata completeness
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any


def validate_json_structure(data: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate JSON structure."""
    errors = []
    
    # Required fields
    if "python_source" not in data:
        errors.append("Missing 'python_source' field")
    if "ir_forward" not in data:
        errors.append("Missing 'ir_forward' field")
    if "metadata" not in data:
        errors.append("Missing 'metadata' field")
    
    # Metadata fields
    if "metadata" in data:
        meta = data["metadata"]
        required_meta = ["kernel_name", "category", "description", "device"]
        for field in required_meta:
            if field not in meta:
                errors.append(f"Missing metadata field: {field}")
    
    return len(errors) == 0, errors


def validate_python_syntax(python_source: str) -> tuple[bool, list[str]]:
    """Validate Python syntax."""
    errors = []
    
    try:
        compile(python_source, "<string>", "exec")
    except SyntaxError as e:
        errors.append(f"Python syntax error: {e}")
    
    # Check for @wp.kernel decorator
    if "@wp.kernel" not in python_source:
        errors.append("Missing @wp.kernel decorator")
    
    return len(errors) == 0, errors


def validate_cuda_ir(ir_code: str, device: str) -> tuple[bool, list[str]]:
    """Validate CUDA/CPU IR patterns."""
    errors = []
    
    if device == "cuda":
        # Check CUDA-specific patterns
        if "blockDim" not in ir_code:
            errors.append("Missing CUDA blockDim")
        if "blockIdx" not in ir_code or "threadIdx" not in ir_code:
            errors.append("Missing CUDA thread indexing")
        if "gridDim" not in ir_code:
            errors.append("Missing CUDA gridDim")
        
        # Check for grid-stride loop
        if "for (size_t _idx" not in ir_code:
            errors.append("Missing grid-stride loop pattern")
    
    else:  # CPU
        if "task_index" not in ir_code:
            errors.append("Missing CPU task_index")
    
    # Common checks
    if "void" not in ir_code or "kernel_forward" not in ir_code:
        errors.append("Missing kernel function signature")
    
    return len(errors) == 0, errors


def validate_backward_ir(ir_code: str, device: str) -> tuple[bool, list[str]]:
    """Validate backward/gradient IR."""
    errors = []
    
    # Check for adjoint variables
    if "adj_" not in ir_code:
        errors.append("Missing adjoint variables")
    
    # Check for backward function name
    if "kernel_backward" not in ir_code:
        errors.append("Missing backward kernel signature")
    
    return len(errors) == 0, errors


def validate_file(filepath: Path) -> Dict[str, Any]:
    """Validate a single JSON file."""
    results = {
        "file": str(filepath),
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        results["valid"] = False
        results["errors"].append(f"Invalid JSON: {e}")
        return results
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to read file: {e}")
        return results
    
    # Validate JSON structure
    valid, errors = validate_json_structure(data)
    if not valid:
        results["valid"] = False
        results["errors"].extend(errors)
        return results
    
    # Validate Python syntax
    valid, errors = validate_python_syntax(data["python_source"])
    if not valid:
        results["valid"] = False
        results["errors"].extend(errors)
    
    # Validate IR
    device = data["metadata"].get("device", "cpu")
    valid, errors = validate_cuda_ir(data["ir_forward"], device)
    if not valid:
        results["valid"] = False
        results["errors"].extend(errors)
    
    # Validate backward if present
    if "ir_backward" in data:
        valid, errors = validate_backward_ir(data["ir_backward"], device)
        if not valid:
            results["warnings"].extend(errors)
    
    return results


def validate_directory(directory: Path) -> Dict[str, Any]:
    """Validate all JSON files in a directory."""
    files = list(directory.glob("*.json"))
    
    if not files:
        return {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "errors": ["No JSON files found"],
            "results": []
        }
    
    results = []
    valid_count = 0
    
    for filepath in files:
        if filepath.name == "generation_stats.json":
            continue  # Skip stats file
        
        result = validate_file(filepath)
        results.append(result)
        if result["valid"]:
            valid_count += 1
    
    return {
        "total_files": len(results),
        "valid_files": valid_count,
        "invalid_files": len(results) - valid_count,
        "results": results
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_kernels.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    print("=" * 60)
    print("CUDA Kernel Validation")
    print("=" * 60)
    print(f"Directory: {directory}")
    print()
    
    summary = validate_directory(directory)
    
    print(f"Total files: {summary['total_files']}")
    print(f"Valid files: {summary['valid_files']}")
    print(f"Invalid files: {summary['invalid_files']}")
    print()
    
    # Show errors
    if summary['invalid_files'] > 0:
        print("Errors found:")
        for result in summary['results']:
            if not result['valid']:
                print(f"\n  {result['file']}:")
                for error in result['errors']:
                    print(f"    - {error}")
    
    # Show warnings
    warning_count = sum(len(r['warnings']) for r in summary['results'])
    if warning_count > 0:
        print(f"\nWarnings: {warning_count}")
    
    if summary['invalid_files'] == 0:
        print("\n✓ All files valid!")
        sys.exit(0)
    else:
        print(f"\n✗ {summary['invalid_files']} file(s) invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
