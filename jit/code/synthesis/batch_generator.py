#!/usr/bin/env python3
"""
Batch Generator: Large-scale training data generation

Generates thousands of Python→IR pairs with chunking, progress tracking,
and error recovery.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pipeline import SynthesisPipeline
import json
from typing import List, Dict, Any
import time


class BatchGenerator:
    """Generate large batches of training pairs efficiently"""
    
    def __init__(self, output_dir: str = "../../data", ir_type: str = "both"):
        """
        Initialize batch generator
        
        Args:
            output_dir: Directory to save generated pairs
            ir_type: Type of IR to extract
        """
        self.output_dir = output_dir
        self.ir_type = ir_type
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_chunk(self, n: int, seed: int, chunk_id: int) -> List[Dict[str, Any]]:
        """
        Generate a chunk of training pairs
        
        Args:
            n: Number of pairs to generate
            seed: Random seed
            chunk_id: Chunk identifier
        
        Returns:
            List of training pairs
        """
        pipeline = SynthesisPipeline(ir_type=self.ir_type, seed=seed)
        pairs = pipeline.generate_batch(n, verbose=False)
        return pairs
    
    def generate_large_dataset(
        self,
        total_pairs: int,
        chunk_size: int = 500,
        start_seed: int = 42,
        prefix: str = "dataset"
    ):
        """
        Generate large dataset in chunks
        
        Args:
            total_pairs: Total number of pairs to generate
            chunk_size: Number of pairs per chunk
            start_seed: Starting random seed
            prefix: Output file prefix
        """
        print("=" * 70)
        print(f"Generating {total_pairs} training pairs")
        print("=" * 70)
        
        n_chunks = (total_pairs + chunk_size - 1) // chunk_size
        all_pairs = []
        total_generated = 0
        start_time = time.time()
        
        for chunk_idx in range(n_chunks):
            # Calculate pairs for this chunk
            remaining = total_pairs - total_generated
            n_pairs = min(chunk_size, remaining)
            
            print(f"\nChunk {chunk_idx + 1}/{n_chunks}: Generating {n_pairs} pairs...")
            
            # Generate chunk
            seed = start_seed + chunk_idx
            chunk_pairs = self.generate_chunk(n_pairs, seed, chunk_idx)
            
            # Update statistics
            total_generated += len(chunk_pairs)
            all_pairs.extend(chunk_pairs)
            
            # Progress
            elapsed = time.time() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0
            eta = (total_pairs - total_generated) / rate if rate > 0 else 0
            
            print(f"  Generated: {len(chunk_pairs)} pairs")
            print(f"  Total: {total_generated}/{total_pairs} ({100*total_generated/total_pairs:.1f}%)")
            print(f"  Rate: {rate:.1f} pairs/sec")
            print(f"  ETA: {eta:.1f} seconds")
            
            # Save intermediate checkpoint every 5 chunks
            if (chunk_idx + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.output_dir,
                    f"{prefix}_checkpoint_{total_generated}.json"
                )
                with open(checkpoint_path, 'w') as f:
                    json.dump(all_pairs, f, indent=2)
                print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save final dataset
        final_path = os.path.join(self.output_dir, f"{prefix}_final.json")
        with open(final_path, 'w') as f:
            json.dump(all_pairs, f, indent=2)
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total pairs: {len(all_pairs)}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Average rate: {len(all_pairs)/elapsed:.1f} pairs/sec")
        print(f"Saved to: {final_path}")
        
        return all_pairs
    
    def generate_diverse_large_dataset(
        self,
        n_per_category: int = 2000,
        prefix: str = "diverse_dataset"
    ):
        """
        Generate large diverse dataset with balanced categories
        
        Args:
            n_per_category: Number of pairs per category
            prefix: Output file prefix
        """
        categories = [
            "arithmetic", "array_op", "math_func",
            "reduction", "linalg", "composite"
        ]
        
        print("=" * 70)
        print(f"Generating diverse dataset ({n_per_category} per category)")
        print("=" * 70)
        
        all_pairs = []
        start_time = time.time()
        
        for cat_idx, category in enumerate(categories):
            print(f"\n[{cat_idx + 1}/{len(categories)}] Category: {category}")
            print("-" * 70)
            
            pipeline = SynthesisPipeline(
                ir_type=self.ir_type,
                seed=42 + cat_idx * 1000
            )
            
            pairs = pipeline.generate_batch(
                n_per_category,
                category=category,
                verbose=True
            )
            
            all_pairs.extend(pairs)
            
            # Save category checkpoint
            cat_path = os.path.join(
                self.output_dir,
                f"{prefix}_{category}.json"
            )
            with open(cat_path, 'w') as f:
                json.dump(pairs, f, indent=2)
            print(f"  Saved: {cat_path}")
        
        # Save combined dataset
        final_path = os.path.join(self.output_dir, f"{prefix}_final.json")
        with open(final_path, 'w') as f:
            json.dump(all_pairs, f, indent=2)
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total pairs: {len(all_pairs)}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Average rate: {len(all_pairs)/elapsed:.1f} pairs/sec")
        print(f"Saved to: {final_path}")
        
        # Category breakdown
        print("\nCategory distribution:")
        category_counts = {}
        for pair in all_pairs:
            cat = pair['metadata']['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat:15s}: {count:5d}")
        
        return all_pairs


# Main
if __name__ == "__main__":
    print("Batch Generator Test")
    
    generator = BatchGenerator(output_dir="../../data")
    
    # Quick test: generate small batch
    print("\nTest: Generate 50 pairs in chunks")
    pairs = generator.generate_large_dataset(
        total_pairs=50,
        chunk_size=20,
        prefix="test_batch"
    )
    
    print(f"\nSuccess! Generated {len(pairs)} pairs")
