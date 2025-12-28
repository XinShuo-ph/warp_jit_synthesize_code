#!/usr/bin/env python3
"""
CUDA Dataset Statistics and Validation Tool

Analyzes and validates generated CUDA IR datasets.

Features:
- Category distribution analysis
- Forward/backward pass validation
- CUDA-specific marker verification
- IR structure validation
- Summary statistics

Usage:
    python3 cuda_dataset_stats.py data/cuda_production
    python3 cuda_dataset_stats.py data/cuda_production --validate
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any
from collections import Counter


def load_dataset(dataset_dir: Path) -> list[dict]:
    """Load all JSON files from dataset directory."""
    pairs = []
    json_files = sorted(dataset_dir.glob("*.json"))
    
    # Skip checkpoint file
    json_files = [f for f in json_files if not f.name.startswith(".")]
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data["_filename"] = filepath.name
                pairs.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return pairs


def validate_cuda_markers(ir_code: str) -> dict[str, bool]:
    """Check for CUDA-specific markers in IR code."""
    return {
        "has_cuda_suffix": "_cuda_kernel_" in ir_code,
        "has_blockDim": "blockDim" in ir_code,
        "has_blockIdx": "blockIdx" in ir_code,
        "has_threadIdx": "threadIdx" in ir_code,
        "has_tile_shared": "tile_shared_storage" in ir_code,
    }


def validate_pair(pair: dict) -> dict[str, Any]:
    """Validate a single CUDA IR pair."""
    issues = []
    
    # Check required fields
    required_fields = ["python_source", "ir_forward", "metadata"]
    for field in required_fields:
        if field not in pair:
            issues.append(f"Missing field: {field}")
    
    # Check metadata
    metadata = pair.get("metadata", {})
    if metadata.get("device") != "cuda":
        issues.append("Device is not 'cuda'")
    if metadata.get("ir_type") != "cuda":
        issues.append("IR type is not 'cuda'")
    
    # Validate forward code
    ir_forward = pair.get("ir_forward", "")
    if ir_forward:
        forward_markers = validate_cuda_markers(ir_forward)
        if not forward_markers["has_cuda_suffix"]:
            issues.append("Forward code missing CUDA suffix")
        if not any([forward_markers["has_blockDim"], forward_markers["has_threadIdx"]]):
            issues.append("Forward code missing CUDA threading markers")
    else:
        issues.append("Empty forward code")
    
    # Validate backward code if expected
    if metadata.get("has_backward", False):
        ir_backward = pair.get("ir_backward", "")
        if ir_backward:
            backward_markers = validate_cuda_markers(ir_backward)
            if not backward_markers["has_cuda_suffix"]:
                issues.append("Backward code missing CUDA suffix")
            if "adj_" not in ir_backward:
                issues.append("Backward code missing adjoint variables")
        else:
            issues.append("Expected backward code but not found")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "filename": pair.get("_filename", "unknown")
    }


def analyze_dataset(pairs: list[dict], validate: bool = False) -> dict[str, Any]:
    """Analyze dataset and return statistics."""
    stats = {
        "total_pairs": len(pairs),
        "categories": Counter(),
        "has_backward": 0,
        "forward_code_lengths": [],
        "backward_code_lengths": [],
        "validation_results": [],
    }
    
    for pair in pairs:
        metadata = pair.get("metadata", {})
        
        # Category distribution
        category = metadata.get("category", "unknown")
        stats["categories"][category] += 1
        
        # Backward pass count
        if metadata.get("has_backward", False):
            stats["has_backward"] += 1
        
        # Code lengths
        ir_forward = pair.get("ir_forward", "")
        if ir_forward:
            stats["forward_code_lengths"].append(len(ir_forward))
        
        ir_backward = pair.get("ir_backward", "")
        if ir_backward:
            stats["backward_code_lengths"].append(len(ir_backward))
        
        # Validation
        if validate:
            result = validate_pair(pair)
            stats["validation_results"].append(result)
    
    return stats


def print_stats(stats: dict[str, Any], validate: bool = False):
    """Print dataset statistics."""
    print("=" * 60)
    print("CUDA Dataset Statistics")
    print("=" * 60)
    print()
    
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"With backward pass: {stats['has_backward']}")
    print()
    
    print("Category distribution:")
    for cat, count in sorted(stats["categories"].items()):
        pct = count / stats["total_pairs"] * 100 if stats["total_pairs"] > 0 else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")
    print()
    
    if stats["forward_code_lengths"]:
        avg_forward = sum(stats["forward_code_lengths"]) / len(stats["forward_code_lengths"])
        print(f"Forward code avg length: {avg_forward:.0f} chars")
    
    if stats["backward_code_lengths"]:
        avg_backward = sum(stats["backward_code_lengths"]) / len(stats["backward_code_lengths"])
        print(f"Backward code avg length: {avg_backward:.0f} chars")
    
    if validate and stats["validation_results"]:
        print()
        print("=" * 60)
        print("Validation Results")
        print("=" * 60)
        
        valid_count = sum(1 for r in stats["validation_results"] if r["valid"])
        invalid_count = len(stats["validation_results"]) - valid_count
        
        print(f"Valid pairs: {valid_count}")
        print(f"Invalid pairs: {invalid_count}")
        
        if invalid_count > 0:
            print()
            print("Invalid pairs:")
            for result in stats["validation_results"]:
                if not result["valid"]:
                    print(f"  {result['filename']}:")
                    for issue in result["issues"]:
                        print(f"    - {issue}")
        else:
            print("All pairs are valid! ✓")


def main():
    parser = argparse.ArgumentParser(
        description="CUDA Dataset Statistics and Validation Tool"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Directory containing CUDA IR JSON files"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Perform detailed validation of each pair"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics as JSON"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Error: Directory not found: {dataset_dir}")
        sys.exit(1)
    
    print(f"Loading dataset from: {dataset_dir}")
    pairs = load_dataset(dataset_dir)
    
    if len(pairs) == 0:
        print("No JSON files found in dataset directory")
        sys.exit(1)
    
    print(f"Loaded {len(pairs)} pairs")
    print()
    
    stats = analyze_dataset(pairs, validate=args.validate)
    
    if args.json:
        # Convert Counter to dict for JSON serialization
        stats["categories"] = dict(stats["categories"])
        # Remove large lists
        del stats["forward_code_lengths"]
        del stats["backward_code_lengths"]
        print(json.dumps(stats, indent=2))
    else:
        print_stats(stats, validate=args.validate)


if __name__ == "__main__":
    main()
