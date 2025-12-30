"""
Analyze and compute statistics for generated dataset
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def analyze_dataset(dataset_dir):
    """Analyze dataset and compute statistics."""
    dataset_path = Path(dataset_dir)
    
    print("=" * 80)
    print(f"Dataset Analysis: {dataset_dir}")
    print("=" * 80)
    
    # Get all JSON files (excluding checkpoint)
    json_files = [f for f in dataset_path.glob('*.json') if f.name != 'checkpoint.json']
    
    print(f"\nTotal files: {len(json_files)}")
    
    if len(json_files) == 0:
        print("No data files found!")
        return
    
    # Statistics containers
    stats = {
        'total_samples': len(json_files),
        'by_category': defaultdict(int),
        'by_complexity': defaultdict(int),
        'ir_length_stats': [],
        'python_length_stats': [],
        'dialects': Counter(),
        'unique_operations': set()
    }
    
    # Analyze each file
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Category
            category = data.get('category', 'unknown')
            stats['by_category'][category] += 1
            
            # Complexity
            complexity = data.get('complexity', 'unknown')
            stats['by_complexity'][complexity] += 1
            
            # IR code length
            ir_code = data.get('ir_code', '')
            stats['ir_length_stats'].append(len(ir_code))
            
            # Python source length
            python_source = data.get('python_source', '')
            stats['python_length_stats'].append(len(python_source))
            
            # Dialect
            dialect = data.get('dialect', 'unknown')
            stats['dialects'][dialect] += 1
            
            # Extract operations from metadata
            if 'operation' in data:
                stats['unique_operations'].add(data['operation'])
            if 'function' in data:
                stats['unique_operations'].add(data['function'])
            if 'pattern' in data:
                stats['unique_operations'].add(data['pattern'])
        
        except Exception as e:
            print(f"Error analyzing {filepath.name}: {e}")
    
    # Compute summary statistics
    ir_lengths = stats['ir_length_stats']
    py_lengths = stats['python_length_stats']
    
    print("\n" + "-" * 80)
    print("Distribution by Category:")
    print("-" * 80)
    for cat in sorted(stats['by_category'].keys()):
        count = stats['by_category'][cat]
        pct = 100 * count / stats['total_samples']
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n" + "-" * 80)
    print("Distribution by Complexity:")
    print("-" * 80)
    for comp in sorted(stats['by_complexity'].keys()):
        count = stats['by_complexity'][comp]
        pct = 100 * count / stats['total_samples']
        print(f"  {comp:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n" + "-" * 80)
    print("IR Code Statistics:")
    print("-" * 80)
    print(f"  Min length:     {min(ir_lengths):6d} chars")
    print(f"  Max length:     {max(ir_lengths):6d} chars")
    print(f"  Mean length:    {sum(ir_lengths) / len(ir_lengths):6.0f} chars")
    print(f"  Median length:  {sorted(ir_lengths)[len(ir_lengths)//2]:6d} chars")
    
    print("\n" + "-" * 80)
    print("Python Source Statistics:")
    print("-" * 80)
    print(f"  Min length:     {min(py_lengths):6d} chars")
    print(f"  Max length:     {max(py_lengths):6d} chars")
    print(f"  Mean length:    {sum(py_lengths) / len(py_lengths):6.0f} chars")
    print(f"  Median length:  {sorted(py_lengths)[len(py_lengths)//2]:6d} chars")
    
    print("\n" + "-" * 80)
    print("Dialects:")
    print("-" * 80)
    for dialect, count in stats['dialects'].most_common():
        print(f"  {dialect}: {count}")
    
    print("\n" + "-" * 80)
    print(f"Unique Operations: {len(stats['unique_operations'])}")
    print("-" * 80)
    for op in sorted(stats['unique_operations']):
        print(f"  - {op}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    
    return stats


def save_stats_to_file(stats, output_file):
    """Save statistics to markdown file."""
    with open(output_file, 'w') as f:
        f.write("# JAX Dataset Statistics\n\n")
        
        f.write(f"## Overview\n")
        f.write(f"- **Total Samples**: {stats['total_samples']:,}\n")
        f.write(f"- **Dialects**: {', '.join(stats['dialects'].keys())}\n")
        f.write(f"- **Unique Operations**: {len(stats['unique_operations'])}\n\n")
        
        f.write(f"## Distribution by Category\n")
        f.write(f"| Category | Count | Percentage |\n")
        f.write(f"|----------|-------|------------|\n")
        for cat in sorted(stats['by_category'].keys()):
            count = stats['by_category'][cat]
            pct = 100 * count / stats['total_samples']
            f.write(f"| {cat:20s} | {count:5,d} | {pct:5.1f}% |\n")
        
        f.write(f"\n## Code Length Statistics\n")
        ir_lengths = stats['ir_length_stats']
        py_lengths = stats['python_length_stats']
        
        f.write(f"### IR Code\n")
        f.write(f"- Min: {min(ir_lengths):,} chars\n")
        f.write(f"- Max: {max(ir_lengths):,} chars\n")
        f.write(f"- Mean: {sum(ir_lengths) / len(ir_lengths):.0f} chars\n")
        f.write(f"- Median: {sorted(ir_lengths)[len(ir_lengths)//2]:,} chars\n\n")
        
        f.write(f"### Python Source\n")
        f.write(f"- Min: {min(py_lengths):,} chars\n")
        f.write(f"- Max: {max(py_lengths):,} chars\n")
        f.write(f"- Mean: {sum(py_lengths) / len(py_lengths):.0f} chars\n")
        f.write(f"- Median: {sorted(py_lengths)[len(py_lengths)//2]:,} chars\n")
    
    print(f"\nStatistics saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze JAX dataset')
    parser.add_argument('dataset_dir', type=str,
                       help='Dataset directory to analyze')
    parser.add_argument('--save', type=str, default=None,
                       help='Save statistics to file')
    
    args = parser.parse_args()
    
    stats = analyze_dataset(args.dataset_dir)
    
    if args.save and stats:
        save_stats_to_file(stats, args.save)


if __name__ == "__main__":
    main()
