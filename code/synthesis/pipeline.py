"""Synthesis Pipeline: Generate kernels and extract IR pairs.

End-to-end pipeline:
1. Generate diverse warp kernels
2. Compile each kernel
3. Extract Python source and C++ IR
4. Save pairs to JSON

Usage:
    from synthesis.pipeline import SynthesisPipeline
    
    pipeline = SynthesisPipeline()
    pairs = pipeline.generate_dataset(count=100)
    pipeline.save_dataset(pairs, "output.json")
"""

import warp as wp
import json
import os
import sys
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.ir_extractor import extract_ir_pair
from synthesis.generator import KernelGenerator


class SynthesisPipeline:
    """End-to-end pipeline for generating Python→IR training data."""
    
    def __init__(self, seed=42):
        """Initialize pipeline.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.generator = KernelGenerator(seed=seed)
        self.pairs = []
        
    def _create_test_inputs(self, kernel):
        """Create appropriate test inputs for a kernel based on its signature.
        
        Args:
            kernel: Warp kernel function
            
        Returns:
            Tuple of (inputs, dim) for wp.launch
        """
        import inspect
        sig = inspect.signature(kernel.func)
        
        inputs = []
        dim = 10  # Default array size
        
        for param_name, param in sig.parameters.items():
            # Check annotation to determine type
            annotation_str = str(param.annotation)
            
            if 'vec3' in annotation_str:
                # Vector array
                arr = wp.array([wp.vec3(float(i), float(i+1), float(i+2)) 
                               for i in range(dim)], dtype=wp.vec3)
                inputs.append(arr)
            elif 'int' in annotation_str and 'array' in annotation_str:
                # Integer array
                arr = wp.array([i % 5 for i in range(dim)], dtype=int)
                inputs.append(arr)
            elif 'float' in annotation_str and 'array' in annotation_str:
                # Float array
                arr = wp.array([float(i) * 0.1 for i in range(dim)], dtype=float)
                inputs.append(arr)
        
        return inputs, dim
    
    def _compile_kernel(self, kernel) -> bool:
        """Compile a kernel by launching it with test data.
        
        Args:
            kernel: Warp kernel to compile
            
        Returns:
            True if compilation successful, False otherwise
        """
        try:
            inputs, dim = self._create_test_inputs(kernel)
            wp.launch(kernel, dim=dim, inputs=inputs)
            wp.synchronize()
            
            # Give warp time to write cache files
            import time
            time.sleep(0.1)
            
            return True
        except Exception as e:
            print(f"Warning: Failed to compile kernel: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_single_pair(self, kernel, description: str) -> Dict[str, Any]:
        """Generate a single Python→IR pair.
        
        Args:
            kernel: Warp kernel function
            description: Human-readable description
            
        Returns:
            Dictionary with Python source, C++ IR, and metadata
        """
        # Compile the kernel
        if not self._compile_kernel(kernel):
            return None
        
        # Extract IR pair
        python_src, cpp_ir = extract_ir_pair(kernel)
        
        if cpp_ir is None:
            print(f"Warning: Failed to extract IR for {description}")
            return None
        
        return {
            'description': description,
            'python_source': python_src,
            'cpp_ir': cpp_ir,
            'kernel_name': kernel.func.__name__,
            'python_length': len(python_src),
            'cpp_length': len(cpp_ir),
        }
    
    def generate_dataset(self, count: int = 100, verbose: bool = True) -> List[Dict[str, Any]]:
        """Generate a dataset of Python→IR pairs.
        
        Args:
            count: Number of pairs to generate
            verbose: Print progress
            
        Returns:
            List of data pairs
        """
        if verbose:
            print(f"Generating {count} Python→IR pairs...")
            print("=" * 70)
        
        pairs = []
        success_count = 0
        attempt_count = 0
        max_attempts = count * 2  # Allow some failures
        
        while success_count < count and attempt_count < max_attempts:
            # Generate a kernel
            kernels = self.generator.generate_batch(count=1)
            if not kernels:
                attempt_count += 1
                continue
            
            kernel, desc = kernels[0]
            attempt_count += 1
            
            # Generate pair
            pair = self.generate_single_pair(kernel, desc)
            
            if pair is not None:
                pairs.append(pair)
                success_count += 1
                
                if verbose and success_count % 10 == 0:
                    print(f"Generated {success_count}/{count} pairs...")
        
        if verbose:
            print("=" * 70)
            print(f"Successfully generated {len(pairs)} pairs")
            print(f"Average Python length: {np.mean([p['python_length'] for p in pairs]):.0f} chars")
            print(f"Average C++ length: {np.mean([p['cpp_length'] for p in pairs]):.0f} chars")
        
        self.pairs = pairs
        return pairs
    
    def save_dataset(self, pairs: List[Dict[str, Any]], filepath: str):
        """Save dataset to JSON file.
        
        Args:
            pairs: List of data pairs
            filepath: Output file path
        """
        # Add metadata
        dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'count': len(pairs),
                'generator': 'SynthesisPipeline',
                'warp_version': wp.__version__,
            },
            'pairs': pairs,
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(pairs)} pairs to {filepath}")
        
        # Also save statistics
        stats_file = filepath.replace('.json', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Dataset Statistics\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Total pairs: {len(pairs)}\n")
            f.write(f"Generated at: {dataset['metadata']['generated_at']}\n")
            f.write(f"\nPython source:\n")
            f.write(f"  Min length: {min(p['python_length'] for p in pairs)}\n")
            f.write(f"  Max length: {max(p['python_length'] for p in pairs)}\n")
            f.write(f"  Mean length: {np.mean([p['python_length'] for p in pairs]):.0f}\n")
            f.write(f"\nC++ IR:\n")
            f.write(f"  Min length: {min(p['cpp_length'] for p in pairs)}\n")
            f.write(f"  Max length: {max(p['cpp_length'] for p in pairs)}\n")
            f.write(f"  Mean length: {np.mean([p['cpp_length'] for p in pairs]):.0f}\n")
            f.write(f"\nSample descriptions:\n")
            for i, pair in enumerate(pairs[:10], 1):
                f.write(f"  {i}. {pair['description']}\n")
        
        print(f"Saved statistics to {stats_file}")
    
    def load_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file.
        
        Args:
            filepath: Input file path
            
        Returns:
            List of data pairs
        """
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        
        self.pairs = dataset['pairs']
        return self.pairs


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→IR training data")
    parser.add_argument('--count', type=int, default=100, help="Number of pairs to generate")
    parser.add_argument('--output', type=str, default='/workspace/data/samples/dataset.json', 
                       help="Output file path")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--quiet', action='store_true', help="Suppress output")
    
    args = parser.parse_args()
    
    # Initialize warp
    wp.init()
    
    # Create pipeline
    pipeline = SynthesisPipeline(seed=args.seed)
    
    # Generate dataset
    pairs = pipeline.generate_dataset(count=args.count, verbose=not args.quiet)
    
    # Save
    pipeline.save_dataset(pairs, args.output)
    
    print("\n✓ Pipeline completed successfully!")
