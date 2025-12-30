"""Validation and analysis tools for generated datasets."""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import jax.numpy as jnp

sys.path.insert(0, '/workspace/jax_jit/code/synthesis')
from pipeline import SynthesisPipeline


def validate_dataset(data_dir: str = "/workspace/jax_jit/data/samples") -> Dict[str, Any]:
    """
    Validate all pairs in the dataset.
    
    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    print("=" * 80)
    print(f"VALIDATING DATASET: {len(json_files)} files")
    print("=" * 80)
    
    valid_count = 0
    invalid_count = 0
    errors = defaultdict(int)
    
    pipeline = SynthesisPipeline(output_dir=data_dir)
    
    for i, json_file in enumerate(json_files):
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{len(json_files)}")
        
        try:
            with open(json_file, 'r') as f:
                pair = json.load(f)
            
            if pipeline.validate_pair(pair):
                valid_count += 1
            else:
                invalid_count += 1
                errors['validation_failed'] += 1
        except json.JSONDecodeError as e:
            invalid_count += 1
            errors['json_decode_error'] += 1
        except Exception as e:
            invalid_count += 1
            error_type = type(e).__name__
            errors[error_type] += 1
    
    print(f"\nValidation complete: {valid_count} valid, {invalid_count} invalid")
    
    if errors:
        print("\nError breakdown:")
        for error_type, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
    
    return {
        'total_files': len(json_files),
        'valid': valid_count,
        'invalid': invalid_count,
        'errors': dict(errors),
        'validation_rate': valid_count / len(json_files) if json_files else 0,
    }


def analyze_dataset(data_dir: str = "/workspace/jax_jit/data/samples") -> Dict[str, Any]:
    """
    Analyze dataset statistics and characteristics.
    
    Returns:
        Dictionary with analysis results
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        return {'error': 'No JSON files found'}
    
    print("\n" + "=" * 80)
    print(f"ANALYZING DATASET: {len(json_files)} files")
    print("=" * 80)
    
    # Statistics accumulators
    categories = defaultdict(int)
    python_line_counts = []
    ir_line_counts = []
    flops_counts = []
    ir_operations = defaultdict(int)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                pair = json.load(f)
            
            # Category
            cat = pair.get('category', 'unknown')
            categories[cat] += 1
            
            # Line counts
            python_lines = len(pair['python_source'].splitlines())
            ir_lines = len(pair['stablehlo_ir'].splitlines())
            python_line_counts.append(python_lines)
            ir_line_counts.append(ir_lines)
            
            # FLOPs
            flops = pair.get('cost_analysis', {}).get('flops', 0)
            flops_counts.append(flops)
            
            # IR operations (count unique StableHLO ops)
            ir_text = pair['stablehlo_ir'].lower()
            for op in ['add', 'subtract', 'multiply', 'divide', 'dot', 'reduce', 
                      'compare', 'select', 'broadcast', 'convert']:
                if f'stablehlo.{op}' in ir_text:
                    ir_operations[op] += 1
        
        except Exception as e:
            continue
    
    # Compute statistics
    def stats(data):
        if not data:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        sorted_data = sorted(data)
        return {
            'min': sorted_data[0],
            'max': sorted_data[-1],
            'avg': sum(data) / len(data),
            'median': sorted_data[len(sorted_data) // 2],
        }
    
    analysis = {
        'total_pairs': len(json_files),
        'categories': dict(categories),
        'python_lines': stats(python_line_counts),
        'ir_lines': stats(ir_line_counts),
        'flops': stats(flops_counts),
        'ir_operations': dict(sorted(ir_operations.items(), key=lambda x: -x[1])),
    }
    
    # Print analysis
    print(f"\nTotal pairs: {analysis['total_pairs']}")
    
    print(f"\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(json_files)
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    print(f"\nPython lines:")
    print(f"  Min: {analysis['python_lines']['min']}")
    print(f"  Max: {analysis['python_lines']['max']}")
    print(f"  Avg: {analysis['python_lines']['avg']:.1f}")
    print(f"  Median: {analysis['python_lines']['median']}")
    
    print(f"\nStableHLO IR lines:")
    print(f"  Min: {analysis['ir_lines']['min']}")
    print(f"  Max: {analysis['ir_lines']['max']}")
    print(f"  Avg: {analysis['ir_lines']['avg']:.1f}")
    print(f"  Median: {analysis['ir_lines']['median']}")
    
    print(f"\nFLOPs:")
    print(f"  Min: {analysis['flops']['min']:.0f}")
    print(f"  Max: {analysis['flops']['max']:.0f}")
    print(f"  Avg: {analysis['flops']['avg']:.1f}")
    print(f"  Total: {sum(flops_counts):.0f}")
    
    print(f"\nTop IR operations:")
    for op, count in list(sorted(ir_operations.items(), key=lambda x: -x[1]))[:10]:
        pct = 100 * count / len(json_files)
        print(f"  {op}: {count} ({pct:.1f}%)")
    
    return analysis


def sample_pairs(data_dir: str = "/workspace/jax_jit/data/samples", n: int = 3) -> None:
    """Display a few sample pairs from the dataset."""
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))[:n]
    
    print("\n" + "=" * 80)
    print(f"SAMPLE PAIRS (showing {min(n, len(json_files))} of {len(list(data_path.glob('*.json')))})")
    print("=" * 80)
    
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r') as f:
                pair = json.load(f)
            
            print(f"\n[{i+1}] {pair['function_name']} (Category: {pair.get('category', 'unknown')})")
            print("-" * 80)
            print("Python Source:")
            print(pair['python_source'])
            print("\nStableHLO IR (first 10 lines):")
            ir_lines = pair['stablehlo_ir'].splitlines()
            for line in ir_lines[:10]:
                print(line)
            if len(ir_lines) > 10:
                print(f"... ({len(ir_lines) - 10} more lines)")
            print(f"\nCost: {pair.get('cost_analysis', {})}")
        
        except Exception as e:
            print(f"\n[{i+1}] Error loading {json_file.name}: {e}")


def check_duplicates(data_dir: str = "/workspace/jax_jit/data/samples") -> Dict[str, int]:
    """Check for duplicate pairs in the dataset."""
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    print("\n" + "=" * 80)
    print("CHECKING FOR DUPLICATES")
    print("=" * 80)
    
    seen_python = {}
    seen_ir = {}
    duplicates = {'python': 0, 'ir': 0, 'both': 0}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                pair = json.load(f)
            
            python_hash = hash(pair['python_source'])
            ir_hash = hash(pair['stablehlo_ir'])
            
            python_dup = python_hash in seen_python
            ir_dup = ir_hash in seen_ir
            
            if python_dup and ir_dup:
                duplicates['both'] += 1
            elif python_dup:
                duplicates['python'] += 1
            elif ir_dup:
                duplicates['ir'] += 1
            
            seen_python[python_hash] = json_file.name
            seen_ir[ir_hash] = json_file.name
        
        except Exception:
            continue
    
    total_dup = duplicates['python'] + duplicates['ir'] + duplicates['both']
    
    print(f"Total files: {len(json_files)}")
    print(f"Duplicates: {total_dup}")
    print(f"  Python only: {duplicates['python']}")
    print(f"  IR only: {duplicates['ir']}")
    print(f"  Both: {duplicates['both']}")
    print(f"Unique pairs: {len(json_files) - total_dup}")
    
    return duplicates


def run_full_validation():
    """Run complete validation and analysis."""
    print("=" * 80)
    print("DATASET VALIDATION AND ANALYSIS")
    print("=" * 80)
    
    # Validate
    validation_results = validate_dataset()
    
    # Analyze
    analysis_results = analyze_dataset()
    
    # Check duplicates
    duplicate_results = check_duplicates()
    
    # Show samples
    sample_pairs(n=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total pairs: {validation_results['total_files']}")
    print(f"Valid pairs: {validation_results['valid']} ({validation_results['validation_rate']*100:.1f}%)")
    print(f"Invalid pairs: {validation_results['invalid']}")
    print(f"Unique pairs: {validation_results['total_files'] - sum(duplicate_results.values())}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_validation()
