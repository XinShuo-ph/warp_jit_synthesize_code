"""Batch Generator for Large-Scale Dataset Generation.

Generates large datasets (10k+) efficiently with progress tracking.

Features:
- Batched generation with progress reporting
- Checkpoint/resume capability
- Error handling and retry logic
- Performance optimization
"""

import warp as wp
import json
import os
import time
from typing import List, Dict, Any
from datetime import datetime

from pipeline import SynthesisPipeline


class BatchGenerator:
    """Generate large-scale datasets with batching and checkpointing."""
    
    def __init__(self, output_dir: str = "/workspace/data", seed: int = 42):
        """Initialize batch generator.
        
        Args:
            output_dir: Directory for output files
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.seed = seed
        self.pipeline = SynthesisPipeline(seed=seed)
        self.checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    def _save_checkpoint(self, pairs: List[Dict[str, Any]], count: int):
        """Save checkpoint for resume capability.
        
        Args:
            pairs: Current list of pairs
            count: Target count
        """
        checkpoint = {
            'pairs': pairs,
            'target_count': count,
            'current_count': len(pairs),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists.
        
        Returns:
            Checkpoint data or None
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def generate_large_dataset(
        self,
        count: int = 10000,
        batch_size: int = 100,
        checkpoint_interval: int = 500,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate large-scale dataset with batching.
        
        Args:
            count: Total number of pairs to generate
            batch_size: Size of each batch
            checkpoint_interval: Save checkpoint every N pairs
            verbose: Print progress
            
        Returns:
            List of all generated pairs
        """
        # Check for checkpoint
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint['target_count'] == count:
            pairs = checkpoint['pairs']
            start_count = len(pairs)
            if verbose:
                print(f"Resuming from checkpoint: {start_count}/{count} pairs")
        else:
            pairs = []
            start_count = 0
        
        if verbose:
            print(f"Generating {count} pairs in batches of {batch_size}")
            print("=" * 70)
        
        start_time = time.time()
        
        while len(pairs) < count:
            # Calculate batch size for this iteration
            remaining = count - len(pairs)
            current_batch_size = min(batch_size, remaining)
            
            # Generate batch
            try:
                batch_pairs = self.pipeline.generate_dataset(
                    count=current_batch_size,
                    verbose=False
                )
                pairs.extend(batch_pairs)
                
                if verbose:
                    elapsed = time.time() - start_time
                    rate = len(pairs) / elapsed if elapsed > 0 else 0
                    eta = (count - len(pairs)) / rate if rate > 0 else 0
                    print(f"Progress: {len(pairs)}/{count} pairs "
                          f"({100*len(pairs)/count:.1f}%) "
                          f"- Rate: {rate:.1f} pairs/sec "
                          f"- ETA: {eta/60:.1f} min")
                
                # Save checkpoint
                if len(pairs) % checkpoint_interval < batch_size:
                    self._save_checkpoint(pairs, count)
                    if verbose:
                        print(f"  Checkpoint saved at {len(pairs)} pairs")
            
            except Exception as e:
                print(f"Error generating batch: {e}")
                print(f"Saving checkpoint and continuing...")
                self._save_checkpoint(pairs, count)
        
        if verbose:
            elapsed = time.time() - start_time
            print("=" * 70)
            print(f"Generation complete!")
            print(f"Total time: {elapsed/60:.1f} minutes")
            print(f"Average rate: {count/elapsed:.1f} pairs/sec")
        
        # Clean up checkpoint
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        return pairs
    
    def save_large_dataset(self, pairs: List[Dict[str, Any]], name: str = "large_dataset"):
        """Save large dataset with compression and statistics.
        
        Args:
            pairs: List of data pairs
            name: Base name for output files
        """
        filepath = os.path.join(self.output_dir, f"{name}.json")
        self.pipeline.save_dataset(pairs, filepath)
        
        print(f"\n✓ Dataset saved successfully!")
        print(f"  Location: {filepath}")
        print(f"  Size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate large-scale Python→IR training data")
    parser.add_argument('--count', type=int, default=10000, help="Number of pairs to generate")
    parser.add_argument('--batch-size', type=int, default=100, help="Batch size")
    parser.add_argument('--checkpoint-interval', type=int, default=500, 
                       help="Save checkpoint every N pairs")
    parser.add_argument('--output', type=str, default='/workspace/data', 
                       help="Output directory")
    parser.add_argument('--name', type=str, default='large_dataset', 
                       help="Dataset name")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--quiet', action='store_true', help="Suppress output")
    
    args = parser.parse_args()
    
    # Initialize warp
    wp.init()
    
    # Create batch generator
    generator = BatchGenerator(output_dir=args.output, seed=args.seed)
    
    # Generate dataset
    pairs = generator.generate_large_dataset(
        count=args.count,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        verbose=not args.quiet
    )
    
    # Save
    generator.save_large_dataset(pairs, name=args.name)
    
    print("\n🎉 Batch generation completed successfully!")
