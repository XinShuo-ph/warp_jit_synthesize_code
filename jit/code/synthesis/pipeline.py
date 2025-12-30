"""
JAX Synthesis Pipeline
End-to-end Python→IR pair generation
"""

import jax.numpy as jnp
import json
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extraction.ir_extractor import IRExtractor
from synthesis.generator import KernelGenerator


class SynthesisPipeline:
    """Pipeline for generating Python→IR training pairs."""
    
    def __init__(self, output_dir='data/generated', dialect='stablehlo', seed=None):
        """
        Initialize synthesis pipeline.
        
        Args:
            output_dir: Directory to save generated pairs
            dialect: IR dialect ('hlo' or 'stablehlo')
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = IRExtractor(dialect=dialect)
        self.generator = KernelGenerator(seed=seed)
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'by_category': {}
        }
    
    def generate_pair(self, category=None, **kwargs):
        """
        Generate a single Python→IR pair.
        
        Args:
            category: Category of kernel to generate (None for random)
            **kwargs: Additional arguments for generation
        
        Returns:
            Dictionary with pair data or None if failed
        """
        try:
            # Generate kernel
            if category == 'arithmetic':
                func, args, meta = self.generator.generate_arithmetic(kwargs.get('operation', 'add'))
            elif category == 'math':
                func, args, meta = self.generator.generate_math_function(kwargs.get('function', 'sin'))
            elif category == 'array':
                func, args, meta = self.generator.generate_array_op(kwargs.get('operation', 'dot'))
            elif category == 'control_flow':
                func, args, meta = self.generator.generate_control_flow(kwargs.get('type', 'where'))
            elif category == 'combined':
                func, args, meta = self.generator.generate_combined(kwargs.get('pattern', 'linear'))
            else:
                # Random generation
                func, args, meta = self.generator.generate_random()
            
            # Extract IR
            ir_data = self.extractor.extract_with_metadata(func, *args)
            
            # Combine with generation metadata
            pair_data = {
                **ir_data,
                **meta,
                'generator_info': {
                    'category': meta['category'],
                    'seed': self.generator.function_counter
                }
            }
            
            # Update stats
            self.stats['success'] += 1
            cat = meta['category']
            self.stats['by_category'][cat] = self.stats['by_category'].get(cat, 0) + 1
            
            return pair_data
            
        except Exception as e:
            self.stats['failed'] += 1
            print(f"Error generating pair: {e}")
            return None
    
    def save_pair(self, pair_data, filename=None):
        """
        Save a pair to JSON file.
        
        Args:
            pair_data: Pair dictionary
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            category = pair_data.get('category', 'unknown')
            func_name = pair_data.get('function_name', 'func')
            filename = f"{category}_{func_name}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(pair_data, f, indent=2)
        
        return filepath
    
    def generate_batch(self, count, category=None, save=True, verbose=True):
        """
        Generate multiple Python→IR pairs.
        
        Args:
            count: Number of pairs to generate
            category: Category filter (None for mixed)
            save: Whether to save to disk
            verbose: Print progress
        
        Returns:
            List of generated pairs
        """
        pairs = []
        
        for i in range(count):
            self.stats['total'] += 1
            
            pair = self.generate_pair(category=category)
            
            if pair is not None:
                pairs.append(pair)
                
                if save:
                    filepath = self.save_pair(pair)
                    if verbose and (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{count} pairs...")
            
            if verbose and (i + 1) == count:
                print(f"Completed: {i + 1}/{count} pairs")
        
        return pairs
    
    def generate_balanced_dataset(self, count_per_category, save=True, verbose=True):
        """
        Generate balanced dataset with equal samples per category.
        
        Args:
            count_per_category: Number of samples per category
            save: Whether to save to disk
            verbose: Print progress
        
        Returns:
            List of all generated pairs
        """
        categories = [
            ('arithmetic', ['add', 'sub', 'mul', 'div']),
            ('math', ['sin', 'cos', 'exp', 'tanh', 'sqrt']),
            ('array', ['dot', 'matmul', 'sum', 'mean', 'transpose']),
            ('control_flow', ['where', 'maximum', 'minimum']),
            ('combined', ['linear', 'quadratic', 'sigmoid', 'softmax', 'relu', 'mse'])
        ]
        
        all_pairs = []
        
        for category, options in categories:
            if verbose:
                print(f"\nGenerating {category} kernels...")
            
            # Distribute count across options
            per_option = count_per_category // len(options)
            remainder = count_per_category % len(options)
            
            for idx, option in enumerate(options):
                count = per_option + (1 if idx < remainder else 0)
                
                for _ in range(count):
                    self.stats['total'] += 1
                    
                    if category == 'arithmetic':
                        pair = self.generate_pair(category=category, operation=option)
                    elif category == 'math':
                        pair = self.generate_pair(category=category, function=option)
                    elif category == 'array':
                        pair = self.generate_pair(category=category, operation=option)
                    elif category == 'control_flow':
                        pair = self.generate_pair(category=category, type=option)
                    elif category == 'combined':
                        pair = self.generate_pair(category=category, pattern=option)
                    
                    if pair is not None:
                        all_pairs.append(pair)
                        if save:
                            self.save_pair(pair)
            
            if verbose:
                print(f"  Completed {category}: {self.stats['by_category'].get(category, 0)} pairs")
        
        return all_pairs
    
    def print_stats(self):
        """Print generation statistics."""
        print("\n" + "=" * 80)
        print("Pipeline Statistics")
        print("=" * 80)
        print(f"Total attempted: {self.stats['total']}")
        print(f"Successful: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {100 * self.stats['success'] / max(1, self.stats['total']):.1f}%")
        
        print("\nBy Category:")
        for cat, count in sorted(self.stats['by_category'].items()):
            print(f"  {cat}: {count}")
        
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='JAX IR synthesis pipeline')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of pairs to generate')
    parser.add_argument('--output', type=str, default='data/generated',
                       help='Output directory')
    parser.add_argument('--category', type=str, default=None,
                       help='Category filter (arithmetic, math, array, control_flow, combined)')
    parser.add_argument('--balanced', action='store_true',
                       help='Generate balanced dataset')
    parser.add_argument('--per-category', type=int, default=20,
                       help='Samples per category for balanced generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--dialect', type=str, default='stablehlo',
                       choices=['hlo', 'stablehlo'],
                       help='IR dialect')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("JAX IR Synthesis Pipeline")
    print("=" * 80)
    print(f"Output directory: {args.output}")
    print(f"IR dialect: {args.dialect}")
    print(f"Random seed: {args.seed}")
    
    # Create pipeline
    pipeline = SynthesisPipeline(
        output_dir=args.output,
        dialect=args.dialect,
        seed=args.seed
    )
    
    # Generate pairs
    if args.balanced:
        print(f"\nGenerating balanced dataset ({args.per_category} per category)...")
        pairs = pipeline.generate_balanced_dataset(
            count_per_category=args.per_category,
            save=True,
            verbose=True
        )
    else:
        print(f"\nGenerating {args.count} pairs...")
        if args.category:
            print(f"Category filter: {args.category}")
        pairs = pipeline.generate_batch(
            count=args.count,
            category=args.category,
            save=True,
            verbose=True
        )
    
    # Print statistics
    pipeline.print_stats()
    
    # Sample output
    if pairs:
        print("\nSample Output (first pair):")
        print("-" * 80)
        sample = pairs[0]
        print(f"Function: {sample['function_name']}")
        print(f"Category: {sample['category']}")
        print(f"Python source:\n{sample['python_source']}")
        print(f"IR code (first 200 chars):\n{sample['ir_code'][:200]}...")


if __name__ == "__main__":
    main()
