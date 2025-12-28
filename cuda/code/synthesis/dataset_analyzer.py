"""
Dataset Analyzer for CUDA IR Pairs

Analyzes generated CUDA IR dataset and produces comprehensive statistics:
- Category distribution and balance
- Code complexity metrics
- Pattern analysis
- Size statistics
- Visualizations
"""
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import sys


class DatasetAnalyzer:
    """Analyze CUDA IR dataset comprehensively."""
    
    def __init__(self):
        self.pairs = []
        self.stats = {
            'total_pairs': 0,
            'categories': defaultdict(int),
            'complexity': defaultdict(list),
            'code_sizes': defaultdict(list),
            'patterns': defaultdict(int),
            'operations': defaultdict(int),
        }
    
    def load_dataset(self, data_dir: str):
        """Load all pairs from dataset directory."""
        data_path = Path(data_dir)
        json_files = list(data_path.rglob("*.json"))
        json_files = [f for f in json_files if 'stats' not in f.name and 'report' not in f.name and 'progress' not in f.name]
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    pair = json.load(f)
                    self.pairs.append(pair)
            except:
                continue
        
        self.stats['total_pairs'] = len(self.pairs)
    
    def analyze_complexity(self, pair: dict) -> Dict:
        """Analyze code complexity metrics."""
        python_src = pair.get('python_source', '')
        cuda_code = pair.get('cuda_forward', '')
        
        metrics = {
            'python_lines': len([l for l in python_src.split('\n') if l.strip()]),
            'cuda_lines': len([l for l in cuda_code.split('\n') if l.strip()]),
            'python_chars': len(python_src),
            'cuda_chars': len(cuda_code),
        }
        
        # Count operations in Python code
        operations = []
        
        # Arithmetic
        operations.extend(re.findall(r'\s+[\+\-\*/]\s+', python_src))
        operations.extend(re.findall(r'wp\.(min|max|abs|sqrt)', python_src))
        
        # Math functions
        operations.extend(re.findall(r'wp\.(sin|cos|exp|log|tan)', python_src))
        
        # Vector operations
        operations.extend(re.findall(r'wp\.(dot|cross|length|normalize)', python_src))
        
        # Atomic operations
        operations.extend(re.findall(r'wp\.atomic_(add|min|max|sub)', python_src))
        
        # Control flow
        if_count = len(re.findall(r'\bif\b', python_src))
        for_count = len(re.findall(r'\bfor\b', python_src))
        operations.extend(['if'] * if_count)
        operations.extend(['for'] * for_count)
        
        metrics['operation_count'] = len(operations)
        metrics['operations'] = operations
        
        return metrics
    
    def analyze_patterns(self, pair: dict) -> List[str]:
        """Detect specific CUDA patterns."""
        cuda_code = pair.get('cuda_forward', '')
        patterns = []
        
        if 'extern "C" __global__' in cuda_code:
            patterns.append('extern_c_global')
        if 'blockDim.x' in cuda_code:
            patterns.append('grid_stride_loop')
        if 'tile_shared_storage' in cuda_code:
            patterns.append('shared_memory')
        if 'atomic_' in cuda_code:
            patterns.append('atomics')
        if 'wp::vec' in cuda_code:
            patterns.append('vectors')
        if 'wp::mat' in cuda_code:
            patterns.append('matrices')
        
        return patterns
    
    def analyze_all(self):
        """Analyze entire dataset."""
        print(f"Analyzing {len(self.pairs)} pairs...")
        
        for i, pair in enumerate(self.pairs):
            if (i + 1) % 200 == 0:
                print(f"  Analyzed {i+1}/{len(self.pairs)}...")
            
            # Category
            category = pair.get('metadata', {}).get('category', 'unknown')
            self.stats['categories'][category] += 1
            
            # Complexity
            metrics = self.analyze_complexity(pair)
            for key in ['python_lines', 'cuda_lines', 'python_chars', 'cuda_chars', 'operation_count']:
                self.stats['complexity'][key].append(metrics[key])
            
            # Operations
            for op in metrics['operations']:
                self.stats['operations'][op] += 1
            
            # Patterns
            patterns = self.analyze_patterns(pair)
            for pattern in patterns:
                self.stats['patterns'][pattern] += 1
        
        print(f"  Analysis complete!")
    
    def compute_statistics(self) -> Dict:
        """Compute summary statistics."""
        summary = {
            'total_pairs': self.stats['total_pairs'],
            'categories': dict(self.stats['categories']),
            'patterns': dict(self.stats['patterns']),
            'operations': dict(sorted(self.stats['operations'].items(), key=lambda x: -x[1])[:20]),  # Top 20
        }
        
        # Compute complexity stats
        for metric_name, values in self.stats['complexity'].items():
            if values:
                summary[metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'median': sorted(values)[len(values) // 2],
                }
        
        # Category balance
        if summary['categories']:
            counts = list(summary['categories'].values())
            summary['category_balance'] = {
                'min': min(counts),
                'max': max(counts),
                'std_dev': (sum((x - sum(counts)/len(counts))**2 for x in counts) / len(counts)) ** 0.5,
            }
        
        return summary
    
    def print_report(self, summary: Dict):
        """Print analysis report."""
        print("\n" + "=" * 70)
        print("CUDA IR Dataset Analysis Report")
        print("=" * 70)
        
        print(f"\nDataset Size: {summary['total_pairs']} pairs")
        
        # Category distribution
        print("\nCategory Distribution:")
        total = summary['total_pairs']
        for cat, count in sorted(summary['categories'].items()):
            pct = 100 * count / total
            bar = '█' * int(pct / 2)
            print(f"  {cat:15s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Category balance
        if 'category_balance' in summary:
            balance = summary['category_balance']
            print(f"\nCategory Balance:")
            print(f"  Min count:  {balance['min']}")
            print(f"  Max count:  {balance['max']}")
            print(f"  Std dev:    {balance['std_dev']:.1f}")
            balance_score = 1 - (balance['std_dev'] / (total / len(summary['categories'])))
            print(f"  Balance:    {balance_score:.1%} (1.0 = perfectly balanced)")
        
        # Complexity metrics
        print("\nCode Complexity:")
        metrics = ['python_lines', 'cuda_lines', 'operation_count']
        for metric in metrics:
            if metric in summary:
                m = summary[metric]
                print(f"  {metric:20s}: min={m['min']:3.0f}, max={m['max']:3.0f}, avg={m['avg']:5.1f}, med={m['median']:3.0f}")
        
        # Code sizes
        print("\nCode Sizes (characters):")
        for metric in ['python_chars', 'cuda_chars']:
            if metric in summary:
                m = summary[metric]
                print(f"  {metric:20s}: min={m['min']:4.0f}, max={m['max']:4.0f}, avg={m['avg']:6.1f}")
        
        # Patterns
        print("\nCUDA Patterns Coverage:")
        for pattern, count in sorted(summary['patterns'].items()):
            pct = 100 * count / total
            print(f"  {pattern:25s}: {count:4d} ({pct:5.1f}%)")
        
        # Top operations
        print("\nTop 10 Operations:")
        ops = list(summary['operations'].items())[:10]
        for op, count in ops:
            pct = 100 * count / total
            print(f"  {op:25s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n" + "=" * 70)
        print("Analysis Complete")
        print("=" * 70)


def main():
    """Main analysis routine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze CUDA IR dataset")
    parser.add_argument('data_dir', help="Directory containing CUDA IR pairs")
    parser.add_argument('-o', '--output', help="Save analysis to JSON file")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = DatasetAnalyzer()
    analyzer.load_dataset(args.data_dir)
    
    if analyzer.stats['total_pairs'] == 0:
        print(f"✗ No valid pairs found in {args.data_dir}")
        return 1
    
    analyzer.analyze_all()
    summary = analyzer.compute_statistics()
    analyzer.print_report(summary)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nDetailed analysis saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
