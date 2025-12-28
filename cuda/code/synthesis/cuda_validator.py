#!/usr/bin/env python3
"""
CUDA Code Validator

Validates CUDA code structure without requiring a GPU.
Checks for required CUDA patterns and code quality.
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """Result of validating a single CUDA pair."""
    id: str
    valid: bool
    has_forward: bool
    has_backward: bool
    has_cuda_patterns: bool
    forward_lines: int
    backward_lines: int
    errors: list[str]


# Required CUDA patterns that should appear in generated code
CUDA_PATTERNS = [
    (r"blockDim\.x", "CUDA block dimension"),
    (r"blockIdx\.x", "CUDA block index"),
    (r"threadIdx\.x", "CUDA thread index"),
]

# Optional but expected patterns
OPTIONAL_PATTERNS = [
    (r"tile_shared_storage_t", "Shared memory"),
    (r"gridDim\.x", "Grid dimension"),
]


def count_lines(code: str) -> int:
    """Count non-empty lines in code."""
    if not code:
        return 0
    return len([l for l in code.split('\n') if l.strip()])


def validate_cuda_code(code: str, code_type: str = "forward") -> tuple[bool, list[str]]:
    """
    Validate that code contains expected CUDA patterns.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not code:
        errors.append(f"Missing {code_type} code")
        return False, errors
    
    # Check for CUDA kernel function signature
    if f"cuda_kernel_{code_type}" not in code:
        errors.append(f"Missing cuda_kernel_{code_type} function")
    
    # Check for required CUDA patterns
    for pattern, description in CUDA_PATTERNS:
        if not re.search(pattern, code):
            errors.append(f"Missing {description} ({pattern})")
    
    return len(errors) == 0, errors


def validate_pair(pair: dict) -> ValidationResult:
    """Validate a single CUDA Python→IR pair."""
    errors = []
    
    # Get identifier
    pair_id = pair.get("id", pair.get("kernel_name", "unknown"))
    
    # Check required fields
    has_forward = "cuda_forward" in pair and pair["cuda_forward"]
    has_backward = "cuda_backward" in pair and pair["cuda_backward"]
    
    if not has_forward:
        errors.append("Missing cuda_forward field")
    
    # Validate forward code
    forward_valid = True
    if has_forward:
        forward_valid, forward_errors = validate_cuda_code(pair["cuda_forward"], "forward")
        errors.extend(forward_errors)
    
    # Validate backward code (if present)
    backward_valid = True
    if has_backward:
        backward_valid, backward_errors = validate_cuda_code(pair["cuda_backward"], "backward")
        errors.extend(backward_errors)
    
    # Check for python source
    if "python_source" not in pair or not pair["python_source"]:
        errors.append("Missing python_source field")
    
    # Check python source has @wp.kernel decorator
    if pair.get("python_source") and "@wp.kernel" not in pair["python_source"]:
        errors.append("Python source missing @wp.kernel decorator")
    
    # Count lines
    forward_lines = count_lines(pair.get("cuda_forward", ""))
    backward_lines = count_lines(pair.get("cuda_backward", ""))
    
    # Check for CUDA patterns
    has_cuda_patterns = forward_valid and (not has_backward or backward_valid)
    
    return ValidationResult(
        id=pair_id,
        valid=len(errors) == 0,
        has_forward=has_forward,
        has_backward=has_backward,
        has_cuda_patterns=has_cuda_patterns,
        forward_lines=forward_lines,
        backward_lines=backward_lines,
        errors=errors
    )


def validate_dataset(input_dir: Path, verbose: bool = False) -> dict:
    """
    Validate all CUDA pairs in a directory.
    
    Returns statistics dictionary.
    """
    input_dir = Path(input_dir)
    json_files = sorted(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return {"error": "No files found"}
    
    results = []
    valid_count = 0
    invalid_count = 0
    total_forward_lines = 0
    total_backward_lines = 0
    categories = {}
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                pair = json.load(f)
            
            result = validate_pair(pair)
            results.append(result)
            
            if result.valid:
                valid_count += 1
                total_forward_lines += result.forward_lines
                total_backward_lines += result.backward_lines
                
                # Track category
                cat = pair.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            else:
                invalid_count += 1
                if verbose:
                    print(f"✗ {json_file.name}: {result.errors}")
        
        except json.JSONDecodeError as e:
            invalid_count += 1
            if verbose:
                print(f"✗ {json_file.name}: Invalid JSON - {e}")
        except Exception as e:
            invalid_count += 1
            if verbose:
                print(f"✗ {json_file.name}: Error - {e}")
    
    # Calculate statistics
    stats = {
        "total_files": len(json_files),
        "valid_pairs": valid_count,
        "invalid_pairs": invalid_count,
        "validation_rate": valid_count / len(json_files) if json_files else 0,
        "avg_forward_lines": total_forward_lines / valid_count if valid_count else 0,
        "avg_backward_lines": total_backward_lines / valid_count if valid_count else 0,
        "by_category": categories,
    }
    
    return stats


def print_validation_report(stats: dict):
    """Print a formatted validation report."""
    print("=" * 60)
    print("CUDA Dataset Validation Report")
    print("=" * 60)
    
    print(f"\nTotal files: {stats['total_files']}")
    print(f"Valid pairs: {stats['valid_pairs']}")
    print(f"Invalid pairs: {stats['invalid_pairs']}")
    print(f"Validation rate: {stats['validation_rate']:.1%}")
    
    print(f"\nAverage forward code lines: {stats['avg_forward_lines']:.1f}")
    print(f"Average backward code lines: {stats['avg_backward_lines']:.1f}")
    
    if stats.get('by_category'):
        print("\nCategory distribution:")
        for cat, count in sorted(stats['by_category'].items()):
            print(f"  {cat}: {count}")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CUDA dataset")
    parser.add_argument("--input", "-i", required=True, help="Input directory with JSON files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show validation errors")
    parser.add_argument("--output", "-o", help="Save stats to JSON file")
    
    args = parser.parse_args()
    
    stats = validate_dataset(Path(args.input), verbose=args.verbose)
    print_validation_report(stats)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {args.output}")
