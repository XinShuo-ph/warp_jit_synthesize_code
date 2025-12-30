"""Batch generation for large-scale Python→StableHLO dataset creation."""
import argparse
import time
from pathlib import Path
import json
from typing import Optional

import sys
sys.path.insert(0, '/workspace/jax_jit/code/synthesis')

from pipeline import SynthesisPipeline


class BatchGenerator:
    """High-throughput batch generator for training data."""
    
    def __init__(self, output_dir: str = "/workspace/jax_jit/data/samples", 
                 seed: Optional[int] = None, verbose: bool = True):
        self.pipeline = SynthesisPipeline(output_dir=output_dir, seed=seed)
        self.verbose = verbose
        self.total_generated = 0
        self.total_failed = 0
        self.start_time = None
    
    def generate_dataset(self, target_count: int, batch_size: int = 100) -> dict:
        """
        Generate a large dataset in batches.
        
        Args:
            target_count: Target number of pairs to generate
            batch_size: Number of pairs per batch
        
        Returns:
            Statistics dictionary
        """
        self.start_time = time.time()
        
        if self.verbose:
            print("=" * 80)
            print(f"BATCH GENERATION: Target {target_count} pairs")
            print("=" * 80)
        
        generated = 0
        batch_num = 0
        
        while generated < target_count:
            batch_num += 1
            remaining = target_count - generated
            current_batch_size = min(batch_size, remaining)
            
            if self.verbose:
                print(f"\nBatch {batch_num}: Generating {current_batch_size} pairs...")
            
            batch_start = time.time()
            pairs = self.pipeline.generate_batch(
                count=current_batch_size,
                save=True,
                verbose=False
            )
            batch_time = time.time() - batch_start
            
            generated += len(pairs)
            self.total_generated = generated
            
            if self.verbose:
                rate = len(pairs) / batch_time if batch_time > 0 else 0
                print(f"  Generated: {len(pairs)}/{current_batch_size} pairs")
                print(f"  Time: {batch_time:.2f}s ({rate:.1f} pairs/sec)")
                print(f"  Total: {generated}/{target_count}")
        
        total_time = time.time() - self.start_time
        
        # Get final statistics
        stats = self.pipeline.get_statistics()
        stats['total_time'] = total_time
        stats['rate'] = generated / total_time if total_time > 0 else 0
        
        if self.verbose:
            self._print_final_stats(stats)
        
        return stats
    
    def generate_balanced_dataset(self, target_count: int, 
                                 categories: Optional[list] = None) -> dict:
        """
        Generate a balanced dataset with equal representation from each category.
        
        Args:
            target_count: Target number of pairs to generate
            categories: List of categories to include (default: all)
        
        Returns:
            Statistics dictionary
        """
        if categories is None:
            categories = ['arithmetic', 'conditional', 'reduction', 'matrix', 
                         'elementwise', 'broadcasting', 'composite']
        
        per_category = target_count // len(categories)
        remainder = target_count % len(categories)
        
        if self.verbose:
            print("=" * 80)
            print(f"BALANCED GENERATION: {target_count} pairs across {len(categories)} categories")
            print(f"Per category: {per_category} pairs")
            print("=" * 80)
        
        self.start_time = time.time()
        total_generated = 0
        
        for i, category in enumerate(categories):
            count = per_category + (1 if i < remainder else 0)
            
            if self.verbose:
                print(f"\n[{i+1}/{len(categories)}] Category: {category} ({count} pairs)")
            
            generated = 0
            failed = 0
            
            while generated < count:
                try:
                    pair = self.pipeline.generate_single(category=category, save=True)
                    if pair is not None:
                        generated += 1
                        if self.verbose and generated % 10 == 0:
                            print(f"  Progress: {generated}/{count}")
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    if self.verbose and failed % 10 == 0:
                        print(f"  Failed: {failed}")
            
            total_generated += generated
            self.total_generated = total_generated
            
            if self.verbose:
                print(f"  Completed: {generated} pairs ({failed} failed)")
        
        total_time = time.time() - self.start_time
        
        stats = self.pipeline.get_statistics()
        stats['total_time'] = total_time
        stats['rate'] = total_generated / total_time if total_time > 0 else 0
        
        if self.verbose:
            self._print_final_stats(stats)
        
        return stats
    
    def _print_final_stats(self, stats: dict) -> None:
        """Print final generation statistics."""
        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total pairs: {stats['total_pairs']}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Rate: {stats['rate']:.2f} pairs/sec")
        print(f"\nCategories:")
        for cat, count in sorted(stats['categories'].items()):
            print(f"  {cat}: {count}")
        print(f"\nAverage Python lines: {stats['avg_python_lines']:.1f}")
        print(f"Average IR lines: {stats['avg_ir_lines']:.1f}")
        print(f"Average FLOPs: {stats['avg_flops']:.1f}")
        print(f"Total FLOPs: {stats['total_flops']:.0f}")
        print("=" * 80)
    
    def save_metadata(self, stats: dict, filename: str = "dataset_metadata.json") -> None:
        """Save dataset metadata to file."""
        metadata_path = Path(self.pipeline.output_dir) / filename
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if self.verbose:
            print(f"\nMetadata saved to: {metadata_path}")


def main():
    """Command-line interface for batch generation."""
    parser = argparse.ArgumentParser(description="Generate Python→StableHLO training dataset")
    parser.add_argument('--count', type=int, default=100, help='Number of pairs to generate')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for generation')
    parser.add_argument('--balanced', action='store_true', help='Generate balanced dataset')
    parser.add_argument('--output-dir', type=str, default='/workspace/jax_jit/data/samples',
                       help='Output directory for samples')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    generator = BatchGenerator(
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    if args.balanced:
        stats = generator.generate_balanced_dataset(target_count=args.count)
    else:
        stats = generator.generate_dataset(
            target_count=args.count,
            batch_size=args.batch_size
        )
    
    generator.save_metadata(stats)


if __name__ == "__main__":
    # If no arguments, run a demo
    if len(sys.argv) == 1:
        print("Running demo with 50 pairs...")
        generator = BatchGenerator(seed=42)
        stats = generator.generate_dataset(target_count=50, batch_size=25)
        generator.save_metadata(stats)
    else:
        main()
