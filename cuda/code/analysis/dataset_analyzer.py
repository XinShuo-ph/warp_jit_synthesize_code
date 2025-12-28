"""
Dataset Analyzer

Analyzes generated CUDA kernel datasets and produces statistics:
- Category distribution
- Complexity metrics
- CUDA pattern coverage
- IR size distribution
- Quality metrics
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import re


class DatasetAnalyzer:
    """Analyzer for CUDA kernel datasets."""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.kernels = []
        self.load_kernels()
    
    def load_kernels(self):
        """Load all kernel JSON files."""
        json_files = list(self.dataset_dir.glob("kernel_*.json"))
        json_files += list(self.dataset_dir.glob("pair_*.json"))
        json_files += list(self.dataset_dir.glob("synth_*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.kernels.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        print(f"Loaded {len(self.kernels)} kernels from {self.dataset_dir}")
    
    def analyze_categories(self) -> Dict[str, int]:
        """Analyze category distribution."""
        categories = Counter()
        for kernel in self.kernels:
            cat = kernel.get("metadata", {}).get("category", "unknown")
            categories[cat] += 1
        return dict(categories)
    
    def analyze_backward_coverage(self) -> Dict[str, Any]:
        """Analyze backward pass coverage."""
        total = len(self.kernels)
        with_backward = sum(1 for k in self.kernels if "ir_backward" in k)
        
        return {
            "total": total,
            "with_backward": with_backward,
            "coverage_pct": (with_backward / total * 100) if total > 0 else 0
        }
    
    def analyze_cuda_patterns(self) -> Dict[str, Any]:
        """Analyze CUDA pattern usage."""
        patterns = {
            "blockDim": 0,
            "blockIdx": 0,
            "threadIdx": 0,
            "gridDim": 0,
            "syncthreads": 0,
            "shared_memory": 0,
            "atomic_ops": 0,
            "grid_stride": 0,
        }
        
        for kernel in self.kernels:
            ir = kernel.get("ir_forward", "")
            
            if "blockDim" in ir:
                patterns["blockDim"] += 1
            if "blockIdx" in ir:
                patterns["blockIdx"] += 1
            if "threadIdx" in ir:
                patterns["threadIdx"] += 1
            if "gridDim" in ir:
                patterns["gridDim"] += 1
            if "syncthreads" in ir or "__syncthreads" in ir:
                patterns["syncthreads"] += 1
            if "shared_" in ir or "__shared__" in ir or "tile_mem" in ir:
                patterns["shared_memory"] += 1
            if "atomic" in ir.lower():
                patterns["atomic_ops"] += 1
            if "for (size_t _idx" in ir:
                patterns["grid_stride"] += 1
        
        total = len(self.kernels)
        patterns_pct = {k: (v / total * 100) if total > 0 else 0 for k, v in patterns.items()}
        
        return {
            "counts": patterns,
            "percentages": patterns_pct
        }
    
    def analyze_complexity(self) -> Dict[str, Any]:
        """Analyze kernel complexity."""
        python_lengths = []
        ir_lengths = []
        
        for kernel in self.kernels:
            py_src = kernel.get("python_source", "")
            ir_fwd = kernel.get("ir_forward", "")
            
            python_lengths.append(len(py_src))
            ir_lengths.append(len(ir_fwd))
        
        def stats(values):
            if not values:
                return {}
            return {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
            }
        
        return {
            "python_source_chars": stats(python_lengths),
            "ir_forward_chars": stats(ir_lengths),
            "total_kernels": len(self.kernels)
        }
    
    def analyze_devices(self) -> Dict[str, int]:
        """Analyze device distribution."""
        devices = Counter()
        for kernel in self.kernels:
            device = kernel.get("metadata", {}).get("device", "unknown")
            devices[device] += 1
        return dict(devices)
    
    def analyze_operations(self) -> Dict[str, Any]:
        """Analyze operations used in kernels."""
        operations = Counter()
        
        for kernel in self.kernels:
            py_src = kernel.get("python_source", "")
            
            # Count common warp operations
            ops = [
                "wp.add", "wp.sub", "wp.mul", "wp.div",
                "wp.sin", "wp.cos", "wp.sqrt", "wp.abs",
                "wp.dot", "wp.cross", "wp.length", "wp.normalize",
                "wp.atomic_add", "wp.atomic_min", "wp.atomic_max",
                "wp.clamp", "wp.min", "wp.max",
                "wp.load", "wp.store", "wp.address"
            ]
            
            for op in ops:
                if op in py_src:
                    operations[op] += 1
        
        return dict(operations.most_common(20))
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("CUDA KERNEL DATASET ANALYSIS")
        report.append("=" * 80)
        report.append(f"Dataset: {self.dataset_dir}")
        report.append(f"Total kernels: {len(self.kernels)}")
        report.append("")
        
        # Category distribution
        report.append("CATEGORY DISTRIBUTION")
        report.append("-" * 80)
        categories = self.analyze_categories()
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(self.kernels) * 100 if self.kernels else 0
            report.append(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")
        report.append("")
        
        # Backward coverage
        report.append("BACKWARD PASS COVERAGE")
        report.append("-" * 80)
        backward = self.analyze_backward_coverage()
        report.append(f"  Total kernels:        {backward['total']}")
        report.append(f"  With backward pass:   {backward['with_backward']}")
        report.append(f"  Coverage:             {backward['coverage_pct']:.1f}%")
        report.append("")
        
        # CUDA patterns
        report.append("CUDA PATTERN USAGE")
        report.append("-" * 80)
        patterns = self.analyze_cuda_patterns()
        for pattern, count in sorted(patterns['counts'].items(), key=lambda x: x[1], reverse=True):
            pct = patterns['percentages'][pattern]
            report.append(f"  {pattern:20s}: {count:4d} ({pct:5.1f}%)")
        report.append("")
        
        # Device distribution
        report.append("DEVICE DISTRIBUTION")
        report.append("-" * 80)
        devices = self.analyze_devices()
        for device, count in sorted(devices.items()):
            pct = count / len(self.kernels) * 100 if self.kernels else 0
            report.append(f"  {device:20s}: {count:4d} ({pct:5.1f}%)")
        report.append("")
        
        # Complexity metrics
        report.append("COMPLEXITY METRICS")
        report.append("-" * 80)
        complexity = self.analyze_complexity()
        report.append(f"  Python source length (chars):")
        for k, v in complexity['python_source_chars'].items():
            report.append(f"    {k:10s}: {v:8.0f}")
        report.append(f"  IR forward length (chars):")
        for k, v in complexity['ir_forward_chars'].items():
            report.append(f"    {k:10s}: {v:8.0f}")
        report.append("")
        
        # Top operations
        report.append("TOP 10 OPERATIONS")
        report.append("-" * 80)
        operations = self.analyze_operations()
        for i, (op, count) in enumerate(list(operations.items())[:10], 1):
            report.append(f"  {i:2d}. {op:25s}: {count:4d}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, output_file: Path):
        """Save analysis report to file."""
        report = self.generate_report()
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python dataset_analyzer.py <dataset_directory> [output_report.txt]")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else dataset_dir / "analysis_report.txt"
    
    if not dataset_dir.exists():
        print(f"Error: Directory {dataset_dir} does not exist")
        sys.exit(1)
    
    analyzer = DatasetAnalyzer(dataset_dir)
    
    if len(analyzer.kernels) == 0:
        print("Error: No kernels found in dataset")
        sys.exit(1)
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    analyzer.save_report(output_file)


if __name__ == "__main__":
    main()
