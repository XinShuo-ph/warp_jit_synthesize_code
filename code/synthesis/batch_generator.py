"""
Batch Generator for Large-Scale Dataset Creation

Optimized pipeline for generating thousands of Python→IR pairs efficiently.
"""

import warp as wp
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthesis.generator import KernelGenerator, KernelSpec, OpType
from synthesis.pipeline import SynthesisPipeline

wp.init()


class BatchGenerator:
    """
    Optimized batch generator for large-scale dataset creation
    
    Features:
    - Progress tracking
    - Error recovery
    - Incremental saving
    - Statistics collection
    """
    
    def __init__(self, output_dir: str = "data/large_dataset", num_workers: int = 1):
        """
        Args:
            output_dir: Directory to save generated pairs
            num_workers: Number of parallel workers (1 = sequential, good for stability)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        
        self.stats = {
            'total_generated': 0,
            'total_failed': 0,
            'by_type': {},
            'start_time': None,
            'end_time': None
        }
    
    def generate_large_batch(self, target_count: int, 
                            batch_size: int = 100,
                            checkpoint_every: int = 100,
                            verbose: bool = True) -> Dict:
        """
        Generate a large dataset in batches with checkpointing
        
        Args:
            target_count: Total number of samples to generate
            batch_size: Samples per batch
            checkpoint_every: Save statistics every N samples
            verbose: Print progress
            
        Returns:
            Final statistics
        """
        self.stats['start_time'] = time.time()
        
        if verbose:
            print("=" * 80)
            print(f"LARGE-SCALE DATASET GENERATION")
            print("=" * 80)
            print(f"Target count: {target_count}")
            print(f"Batch size: {batch_size}")
            print(f"Output directory: {self.output_dir}")
            print(f"Workers: {self.num_workers}")
            print()
        
        # Create pipeline (one per worker if parallel, else single)
        pipeline = SynthesisPipeline(output_dir=str(self.output_dir))
        
        total_attempts = 0
        checkpoint_counter = 0
        
        while self.stats['total_generated'] < target_count:
            # Determine batch size for this iteration
            remaining = target_count - self.stats['total_generated']
            current_batch = min(batch_size, remaining)
            
            if verbose:
                print(f"\nGenerating batch ({self.stats['total_generated']}/{target_count})...")
            
            # Generate batch
            batch_start = time.time()
            
            for i in range(current_batch):
                total_attempts += 1
                pair = pipeline.generate_single_pair(save=True)
                
                if pair:
                    self.stats['total_generated'] += 1
                    op_type = pair['metadata']['op_type']
                    self.stats['by_type'][op_type] = self.stats['by_type'].get(op_type, 0) + 1
                else:
                    self.stats['total_failed'] += 1
                
                checkpoint_counter += 1
                
                # Progress update
                if verbose and (i + 1) % 10 == 0:
                    elapsed = time.time() - batch_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (current_batch - i - 1) / rate if rate > 0 else 0
                    print(f"  Progress: {i+1}/{current_batch} ({rate:.1f} samples/sec, ETA: {eta:.0f}s)")
                
                # Checkpoint
                if checkpoint_counter >= checkpoint_every:
                    self._save_checkpoint(verbose=verbose)
                    checkpoint_counter = 0
            
            batch_elapsed = time.time() - batch_start
            if verbose:
                print(f"  Batch complete in {batch_elapsed:.1f}s")
                print(f"  Success rate: {100 * self.stats['total_generated'] / total_attempts:.1f}%")
        
        self.stats['end_time'] = time.time()
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        if verbose:
            print("\n" + "=" * 80)
            print("GENERATION COMPLETE")
            print("=" * 80)
            print(f"Total generated: {self.stats['total_generated']}")
            print(f"Total failed: {self.stats['total_failed']}")
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"Average rate: {self.stats['total_generated'] / total_time:.2f} samples/sec")
            print(f"\nDistribution by type:")
            for op_type, count in sorted(self.stats['by_type'].items()):
                pct = 100 * count / self.stats['total_generated']
                print(f"  {op_type:15s}: {count:4d} ({pct:.1f}%)")
        
        # Final checkpoint
        self._save_checkpoint(verbose=verbose)
        
        return self.stats
    
    def _save_checkpoint(self, verbose: bool = False):
        """Save current statistics to checkpoint file"""
        checkpoint_file = self.output_dir / "generation_stats.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        if verbose:
            print(f"  Checkpoint saved: {self.stats['total_generated']} samples")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if it exists"""
        checkpoint_file = self.output_dir / "generation_stats.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                return json.load(f)
        return None
    
    def resume_generation(self, target_count: int, **kwargs) -> Dict:
        """Resume generation from checkpoint"""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            self.stats = checkpoint
            remaining = target_count - self.stats.get('total_generated', 0)
            print(f"Resuming from checkpoint: {self.stats['total_generated']} samples already generated")
            print(f"Generating {remaining} more samples to reach {target_count}")
        
        return self.generate_large_batch(target_count, **kwargs)


def generate_dataset(target_count: int = 1000, output_dir: str = "data/large_dataset"):
    """
    Convenience function to generate a dataset
    
    Args:
        target_count: Number of samples to generate
        output_dir: Output directory
    """
    generator = BatchGenerator(output_dir=output_dir, num_workers=1)
    stats = generator.generate_large_batch(
        target_count=target_count,
        batch_size=100,
        checkpoint_every=50,
        verbose=True
    )
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate large-scale training dataset")
    parser.add_argument("--count", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/large_dataset", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    generator = BatchGenerator(output_dir=args.output)
    
    if args.resume:
        stats = generator.resume_generation(target_count=args.count)
    else:
        stats = generator.generate_large_batch(target_count=args.count)
    
    print("\n✓ Dataset generation complete!")
