#!/usr/bin/env python3
"""
End-to-end pipeline: Generate kernels → Compile → Extract IR → Save

This pipeline automates the complete workflow for creating training data.
"""

import warp as wp
import json
import os
from pathlib import Path
from typing import List, Dict
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from synthesis.generator import KernelGenerator
from extraction.ir_extractor import IRExtractor, IRExtractorError

wp.init()

class DataSynthesisPipeline:
    """
    End-to-end pipeline for generating Python→IR training data.
    
    Workflow:
    1. Generate varied Python kernels
    2. Compile kernels (by launching them)
    3. Extract IR (C++ code)
    4. Save paired data (Python source + IR)
    """
    
    def __init__(self, output_dir: str, seed: int = 42, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = KernelGenerator(seed=seed)
        self.extractor = IRExtractor()
        self.extractor.set_verbose(verbose)
        
        self.verbose = verbose
        self.generated = []
        self.failed = []
    
    def generate_single(self, index: int) -> bool:
        """
        Generate a single kernel and extract its IR.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate kernel
            kernel_func, params, source = self.generator.generate_random()
            
            if self.verbose:
                print(f"[{index}] Generated: {kernel_func.key}")
            
            # Launch to compile
            wp.launch(kernel=kernel_func, **params['launch_args'])
            wp.synchronize()
            
            # Extract IR
            ir = self.extractor.extract_ir(kernel_func, trigger_compile=False)
            
            # Validate
            valid, msg = ir.validate()
            if not valid:
                raise ValueError(f"Validation failed: {msg}")
            
            # Save
            output_file = self.output_dir / f"sample_{index:04d}.json"
            data = {
                'index': index,
                'kernel_name': ir.kernel_name,
                'python_source': ir.python_source,
                'cpp_code': ir.cpp_code,
                'meta': ir.meta,
                'module_hash': ir.module_hash,
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.generated.append(output_file.name)
            
            if self.verbose:
                print(f"  ✓ Saved: {output_file.name}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Failed: {e}")
            self.failed.append({'index': index, 'error': str(e)})
            return False
    
    def generate_batch(self, count: int, start_index: int = 0) -> Dict:
        """
        Generate multiple samples.
        
        Args:
            count: Number of samples to generate
            start_index: Starting index for naming
            
        Returns:
            Statistics dictionary
        """
        if self.verbose:
            print("="*60)
            print(f"Generating {count} samples...")
            print("="*60)
        
        success_count = 0
        
        for i in range(count):
            index = start_index + i
            if self.generate_single(index):
                success_count += 1
        
        stats = {
            'total': count,
            'success': success_count,
            'failed': len(self.failed),
            'output_dir': str(self.output_dir),
            'generated_files': self.generated,
            'failures': self.failed
        }
        
        if self.verbose:
            print("\n" + "="*60)
            print(f"Generation complete:")
            print(f"  Success: {success_count}/{count}")
            print(f"  Failed: {len(self.failed)}")
            print("="*60)
        
        return stats


def main():
    """Run the pipeline to generate samples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Python→IR training data')
    parser.add_argument('--count', type=int, default=10, 
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='/workspace/data/pipeline',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DataSynthesisPipeline(
        output_dir=args.output,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # Generate samples
    stats = pipeline.generate_batch(args.count)
    
    # Save statistics
    stats_file = Path(args.output) / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_file}")
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
