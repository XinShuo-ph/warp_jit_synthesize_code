#!/usr/bin/env python3
"""
Analyze and generate statistics for the dataset.
"""

import json
import os
from pathlib import Path
from collections import Counter
import sys

def analyze_dataset(data_dir: str = "data"):
    """Analyze all JSON files in the dataset."""
    
    data_path = Path(data_dir)
    
    # Find all JSON sample files
    sample_files = list(data_path.rglob("*.json"))
    
    print(f"Found {len(sample_files)} sample files")
    
    if len(sample_files) == 0:
        print("No samples found!")
        return
    
    # Statistics
    stats = {
        'total_samples': len(sample_files),
        'total_size_mb': 0,
        'template_counts': Counter(),
        'device_counts': Counter(),
        'python_lines': [],
        'forward_ir_lines': [],
        'kernel_names': set(),
        'by_directory': Counter(),
    }
    
    # Analyze each file
    for i, filepath in enumerate(sample_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Size
            stats['total_size_mb'] += os.path.getsize(filepath) / (1024 * 1024)
            
            # Template type
            if 'template_type' in data:
                stats['template_counts'][data['template_type']] += 1
            elif "kernel_type" in data:
                stats["template_counts"][data["kernel_type"]] += 1
            elif 'kernel_name' in data:
                stats['template_counts'][data['kernel_name'].split('_')[0]] += 1

            # Device
            device = (data.get("device") or data.get("metadata", {}).get("device") or "cpu").lower()
            stats["device_counts"][device] += 1
            
            # Line counts
            if 'python_source' in data:
                stats['python_lines'].append(len(data['python_source'].split('\n')))
            
            if "cpp_ir_forward" in data:
                stats["forward_ir_lines"].append(len(data["cpp_ir_forward"].split("\n")))
            
            # Kernel names
            if 'kernel_name' in data:
                stats['kernel_names'].add(data['kernel_name'])
            
            # Directory
            stats['by_directory'][filepath.parent.name] += 1
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Compute averages
    if stats['python_lines']:
        stats['avg_python_lines'] = sum(stats['python_lines']) / len(stats['python_lines'])
        stats['min_python_lines'] = min(stats['python_lines'])
        stats['max_python_lines'] = max(stats['python_lines'])
    
    if stats["forward_ir_lines"]:
        stats["avg_forward_ir_lines"] = sum(stats["forward_ir_lines"]) / len(stats["forward_ir_lines"])
        stats["min_forward_ir_lines"] = min(stats["forward_ir_lines"])
        stats["max_forward_ir_lines"] = max(stats["forward_ir_lines"])
    
    stats['unique_kernels'] = len(stats['kernel_names'])
    
    # Print report
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Total size: {stats['total_size_mb']:.1f} MB")
    print(f"Unique kernels: {stats['unique_kernels']}")
    
    print(f"\nTemplate distribution:")
    for template, count in sorted(stats['template_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_samples']
        print(f"  {template:12s}: {count:4d} ({pct:5.1f}%)")

    print("\nDevice distribution:")
    for dev, count in sorted(stats["device_counts"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["total_samples"]
        print(f"  {dev:12s}: {count:4d} ({pct:5.1f}%)")
    
    if stats['python_lines']:
        print(f"\nPython source lines:")
        print(f"  Average: {stats['avg_python_lines']:.1f}")
        print(f"  Range: {stats['min_python_lines']}-{stats['max_python_lines']}")
    
    if stats.get("forward_ir_lines"):
        print("\nForward IR lines:")
        print(f"  Average: {stats['avg_forward_ir_lines']:.1f}")
        print(f"  Range: {stats['min_forward_ir_lines']}-{stats['max_forward_ir_lines']}")
    
    print(f"\nBy directory:")
    for directory, count in sorted(stats['by_directory'].items()):
        print(f"  {directory:20s}: {count:4d}")
    
    print("="*60)
    
    return stats


def create_stats_markdown(stats: dict, output_file: str = "code/notes/data_stats.md"):
    """Create markdown documentation of statistics."""
    
    content = f"""# Dataset Statistics

**Total Samples**: {stats['total_samples']}  
**Total Size**: {stats['total_size_mb']:.1f} MB  
**Unique Kernels**: {stats['unique_kernels']}

## Template Distribution
"""
    
    for template, count in sorted(stats['template_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_samples']
        content += f"- **{template}**: {count} ({pct:.1f}%)\n"
    
    if stats.get('avg_python_lines'):
        content += f"""
## Code Size
- Python lines: {stats['min_python_lines']}-{stats['max_python_lines']} (avg: {stats['avg_python_lines']:.1f})
- Forward IR lines: {stats.get('min_forward_ir_lines','?')}-{stats.get('max_forward_ir_lines','?')} (avg: {stats.get('avg_forward_ir_lines',0):.1f})
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\nStatistics saved to: {output_file}")
    print(f"Total lines: {len(content.split(chr(10)))}")


if __name__ == "__main__":
    stats = analyze_dataset()
    if stats:
        create_stats_markdown(stats)
