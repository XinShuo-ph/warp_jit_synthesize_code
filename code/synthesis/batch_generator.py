#!/usr/bin/env python3
"""
Batch Generator for Large-Scale Dataset Creation

Efficiently generates 10k+ Python→IR training pairs with:
- Progress tracking and checkpointing
- Memory-efficient processing
- Error recovery
- Parallel-ready architecture
"""

import warp as wp
import json
import os
import time
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from synthesis.generator import KernelGenerator
from extraction.ir_extractor import IRExtractor, IRExtractorError

wp.init()

class BatchGenerator:
    """
    Large-scale batch generator for training data.
    
    Features:
    - Checkpointing: Resume from interruption
    - Progress tracking: Real-time statistics
    - Memory efficient: Process one at a time
    - Error recovery: Continue on failures
    """
    
    def __init__(self, 
                 output_dir: str, 
                 seed: int = 42,
                 checkpoint_interval: int = 100,
                 verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        
        self.generator = KernelGenerator(seed=seed)
        self.extractor = IRExtractor()
        self.extractor.set_verbose(False)  # Reduce noise
        
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.stats_file = self.output_dir / "generation_stats.json"
        
        # Load checkpoint if exists
        self.checkpoint = self._load_checkpoint()
        
        # Statistics
        self.stats = {
            'total_requested': 0,
            'generated': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
            'template_counts': {},
            'errors': []
        }
    
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint from file if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'last_index': -1, 'completed': []}
    
    def _save_checkpoint(self):
        """Save current progress."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f)
    
    def _save_stats(self):
        """Save generation statistics."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def generate_single(self, index: int) -> bool:
        """
        Generate a single sample.
        
        Returns:
            True if successful, False otherwise
        """
        # Check if already generated
        output_file = self.output_dir / f"sample_{index:05d}.json"
        if output_file.exists():
            if self.verbose and index % 100 == 0:
                print(f"[{index}] Already exists, skipping")
            return True
        
        try:
            # Generate kernel
            kernel_func, params, source = self.generator.generate_random()
            
            # Launch to compile
            wp.launch(kernel=kernel_func, **params['launch_args'])
            wp.synchronize()
            
            # Extract IR
            ir = self.extractor.extract_ir(kernel_func, trigger_compile=False)
            
            # Validate
            valid, msg = ir.validate()
            if not valid:
                raise ValueError(f"Validation failed: {msg}")
            
            # Determine template type from kernel name
            template_type = kernel_func.key.split('_')[0]
            
            # Save
            data = {
                'index': index,
                'seed': self.seed,
                'template_type': template_type,
                'kernel_name': ir.kernel_name,
                'python_source': ir.python_source,
                'cpp_code': ir.cpp_code,
                'meta': ir.meta,
                'module_hash': ir.module_hash,
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update stats
            self.stats['generated'] += 1
            self.stats['template_counts'][template_type] = \
                self.stats['template_counts'].get(template_type, 0) + 1
            
            if self.verbose and index % 100 == 0:
                elapsed = time.time() - self.stats['start_time']
                rate = self.stats['generated'] / elapsed
                print(f"[{index}] Generated {template_type} (rate: {rate:.1f}/s)")
            
            return True
            
        except Exception as e:
            self.stats['failed'] += 1
            error_info = {'index': index, 'error': str(e)}
            self.stats['errors'].append(error_info)
            
            if self.verbose and index % 100 == 0:
                print(f"[{index}] Failed: {e}")
            
            return False
    
    def generate_batch(self, count: int, start_index: int = 0) -> Dict:
        """
        Generate a batch of samples.
        
        Args:
            count: Number of samples to generate
            start_index: Starting index
            
        Returns:
            Statistics dictionary
        """
        self.stats['total_requested'] = count
        self.stats['start_time'] = time.time()
        
        if self.verbose:
            print("="*60)
            print(f"Batch Generation: {count} samples")
            print(f"Output: {self.output_dir}")
            print(f"Starting from index: {start_index}")
            print("="*60)
        
        # Resume from checkpoint if available
        if self.checkpoint['last_index'] >= start_index:
            resume_index = self.checkpoint['last_index'] + 1
            if self.verbose:
                print(f"Resuming from checkpoint: index {resume_index}")
            start_index = resume_index
        
        for i in range(count):
            index = start_index + i
            
            # Generate sample
            success = self.generate_single(index)
            
            # Update checkpoint
            if success:
                self.checkpoint['last_index'] = index
                self.checkpoint['completed'].append(index)
            
            # Save checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint()
                self._save_stats()
                
                if self.verbose:
                    elapsed = time.time() - self.stats['start_time']
                    rate = self.stats['generated'] / elapsed
                    remaining = (count - i - 1) / rate if rate > 0 else 0
                    print(f"\nCheckpoint: {i+1}/{count} samples")
                    print(f"  Success: {self.stats['generated']}")
                    print(f"  Failed: {self.stats['failed']}")
                    print(f"  Rate: {rate:.1f} samples/sec")
                    print(f"  Estimated remaining: {remaining:.0f}s")
                    print()
        
        # Final save
        self.stats['end_time'] = time.time()
        self._save_checkpoint()
        self._save_stats()
        
        # Summary
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        if self.verbose:
            print("\n" + "="*60)
            print("Batch Generation Complete")
            print("="*60)
            print(f"Total time: {total_time:.1f}s")
            print(f"Success: {self.stats['generated']}/{count}")
            print(f"Failed: {self.stats['failed']}")
            print(f"Average rate: {self.stats['generated']/total_time:.2f} samples/sec")
            print("\nTemplate distribution:")
            for template, count in sorted(self.stats['template_counts'].items()):
                print(f"  {template}: {count}")
            print("="*60)
        
        return self.stats


def main():
    """Generate large-scale dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch generate training data')
    parser.add_argument('--count', type=int, default=10000,
                       help='Number of samples to generate (default: 10000)')
    parser.add_argument('--output', type=str, default='/workspace/data/large_dataset',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting index')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Checkpoint interval')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create batch generator
    generator = BatchGenerator(
        output_dir=args.output,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        verbose=not args.quiet
    )
    
    # Generate samples
    stats = generator.generate_batch(args.count, start_index=args.start)
    
    # Exit with error code if any failures
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
