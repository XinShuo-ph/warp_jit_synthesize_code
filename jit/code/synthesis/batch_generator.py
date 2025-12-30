"""
JAX Batch Generator
Efficient large-scale generation of Python→IR pairs
"""

import jax.numpy as jnp
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from synthesis.pipeline import SynthesisPipeline


class BatchGenerator:
    """Efficient batch generation for large datasets."""
    
    def __init__(self, output_dir='data/large_dataset', dialect='stablehlo', 
                 seed=42, checkpoint_every=1000):
        """
        Initialize batch generator.
        
        Args:
            output_dir: Output directory
            dialect: IR dialect
            seed: Random seed
            checkpoint_every: Save checkpoint every N samples
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_every = checkpoint_every
        self.checkpoint_file = self.output_dir / 'checkpoint.json'
        
        self.pipeline = SynthesisPipeline(
            output_dir=str(self.output_dir),
            dialect=dialect,
            seed=seed
        )
        
        self.start_time = None
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint if it exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'total_generated': 0,
            'last_checkpoint': 0,
            'start_time': None
        }
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint_data, f, indent=2)
    
    def generate_large_dataset(self, target_count, balanced=True, 
                               verbose=True, resume=False):
        """
        Generate large dataset with checkpointing.
        
        Args:
            target_count: Target number of samples
            balanced: Use balanced generation
            verbose: Print progress
            resume: Resume from checkpoint
        
        Returns:
            Total number of generated samples
        """
        if not resume:
            self.checkpoint_data = {
                'total_generated': 0,
                'last_checkpoint': 0,
                'start_time': datetime.now().isoformat()
            }
        
        start_count = self.checkpoint_data['total_generated']
        remaining = target_count - start_count
        
        if remaining <= 0:
            print(f"Target already reached: {start_count}/{target_count}")
            return start_count
        
        print("=" * 80)
        print("JAX Batch Generator - Large Scale Generation")
        print("=" * 80)
        print(f"Target count: {target_count}")
        print(f"Already generated: {start_count}")
        print(f"Remaining: {remaining}")
        print(f"Output directory: {self.output_dir}")
        print(f"Checkpoint every: {self.checkpoint_every} samples")
        print("=" * 80)
        
        self.start_time = time.time()
        
        if balanced:
            # Calculate samples per category
            num_categories = 5  # arithmetic, math, array, control_flow, combined
            per_category = remaining // num_categories
            
            print(f"\nBalanced generation: ~{per_category} per category")
            
            # Generate in chunks
            chunk_size = min(self.checkpoint_every // num_categories, per_category)
            chunks = (per_category + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(chunks):
                current_per_cat = min(chunk_size, per_category - chunk_idx * chunk_size)
                
                if current_per_cat <= 0:
                    break
                
                if verbose:
                    print(f"\nChunk {chunk_idx + 1}/{chunks}: Generating {current_per_cat} per category...")
                
                self.pipeline.generate_balanced_dataset(
                    count_per_category=current_per_cat,
                    save=True,
                    verbose=verbose
                )
                
                # Update checkpoint
                generated_this_chunk = current_per_cat * num_categories
                self.checkpoint_data['total_generated'] += generated_this_chunk
                self.checkpoint_data['last_checkpoint'] = self.checkpoint_data['total_generated']
                self._save_checkpoint()
                
                # Print progress
                elapsed = time.time() - self.start_time
                rate = self.checkpoint_data['total_generated'] / elapsed
                eta = (target_count - self.checkpoint_data['total_generated']) / rate
                
                print(f"\nProgress: {self.checkpoint_data['total_generated']}/{target_count} "
                      f"({100 * self.checkpoint_data['total_generated'] / target_count:.1f}%)")
                print(f"Rate: {rate:.1f} samples/sec")
                print(f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        else:
            # Simple sequential generation
            for i in range(start_count, target_count, self.checkpoint_every):
                batch_size = min(self.checkpoint_every, target_count - i)
                
                if verbose:
                    print(f"\nGenerating batch {i}-{i+batch_size}...")
                
                self.pipeline.generate_batch(
                    count=batch_size,
                    category=None,
                    save=True,
                    verbose=verbose
                )
                
                # Update checkpoint
                self.checkpoint_data['total_generated'] = i + batch_size
                self.checkpoint_data['last_checkpoint'] = self.checkpoint_data['total_generated']
                self._save_checkpoint()
                
                # Print progress
                elapsed = time.time() - self.start_time
                rate = (self.checkpoint_data['total_generated'] - start_count) / elapsed
                eta = (target_count - self.checkpoint_data['total_generated']) / rate
                
                print(f"\nProgress: {self.checkpoint_data['total_generated']}/{target_count}")
                print(f"Rate: {rate:.1f} samples/sec")
                print(f"ETA: {eta:.1f}s")
        
        total_time = time.time() - self.start_time
        final_count = self.checkpoint_data['total_generated']
        
        print("\n" + "=" * 80)
        print("Generation Complete!")
        print("=" * 80)
        print(f"Total generated: {final_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average rate: {final_count / total_time:.1f} samples/sec")
        print("=" * 80)
        
        # Print final stats
        self.pipeline.print_stats()
        
        return final_count
    
    def validate_dataset(self, sample_size=100):
        """
        Validate generated dataset by sampling.
        
        Args:
            sample_size: Number of samples to validate
        
        Returns:
            Validation statistics
        """
        import random
        
        print("\n" + "=" * 80)
        print("Dataset Validation")
        print("=" * 80)
        
        # Get all JSON files
        json_files = list(self.output_dir.glob('*.json'))
        json_files = [f for f in json_files if f.name != 'checkpoint.json']
        
        print(f"Total files: {len(json_files)}")
        
        if len(json_files) == 0:
            print("No files to validate!")
            return
        
        # Sample files
        sample_files = random.sample(json_files, min(sample_size, len(json_files)))
        
        stats = {
            'total_checked': 0,
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        for filepath in sample_files:
            stats['total_checked'] += 1
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Check required fields
                required = ['function_name', 'ir_code', 'category', 'dialect']
                if all(k in data for k in required):
                    stats['valid'] += 1
                else:
                    stats['invalid'] += 1
                    missing = [k for k in required if k not in data]
                    stats['errors'].append(f"{filepath.name}: missing {missing}")
            
            except Exception as e:
                stats['invalid'] += 1
                stats['errors'].append(f"{filepath.name}: {str(e)}")
        
        print(f"\nValidated {stats['total_checked']} samples:")
        print(f"  Valid: {stats['valid']}")
        print(f"  Invalid: {stats['invalid']}")
        
        if stats['errors']:
            print(f"\nErrors found:")
            for error in stats['errors'][:10]:  # Show first 10
                print(f"  - {error}")
        
        print("=" * 80)
        
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='JAX batch generator')
    parser.add_argument('--target', type=int, default=10000,
                       help='Target number of samples')
    parser.add_argument('--output', type=str, default='data/large_dataset',
                       help='Output directory')
    parser.add_argument('--balanced', action='store_true', default=True,
                       help='Use balanced generation')
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                       help='Checkpoint frequency')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset after generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--dialect', type=str, default='stablehlo',
                       choices=['hlo', 'stablehlo'],
                       help='IR dialect')
    
    args = parser.parse_args()
    
    # Create batch generator
    generator = BatchGenerator(
        output_dir=args.output,
        dialect=args.dialect,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every
    )
    
    # Generate dataset
    total = generator.generate_large_dataset(
        target_count=args.target,
        balanced=args.balanced,
        verbose=True,
        resume=args.resume
    )
    
    # Validate if requested
    if args.validate:
        generator.validate_dataset(sample_size=100)


if __name__ == "__main__":
    main()
