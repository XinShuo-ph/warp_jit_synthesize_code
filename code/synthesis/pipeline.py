"""
End-to-End Synthesis Pipeline

Generates Python kernels → Compiles → Extracts IR → Saves pairs
"""

import warp as wp
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import traceback

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthesis.generator import KernelGenerator, KernelSpec, OpType
from extraction.ir_extractor import IRExtractor

wp.init()


class SynthesisPipeline:
    """End-to-end pipeline for generating training data"""
    
    def __init__(self, output_dir: str = "data/samples"):
        self.generator = KernelGenerator()
        self.extractor = IRExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'by_type': {}
        }
    
    def _create_dummy_inputs(self, spec: KernelSpec):
        """Create dummy inputs based on kernel spec"""
        inputs = []
        n = 10  # Default array size
        
        if spec.op_type == OpType.VECTOR:
            # Vector arrays
            inputs.append(wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3))  # positions
            inputs.append(wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3))  # velocities
            inputs.append(wp.zeros(n, dtype=wp.vec3))  # forces
            if spec.has_scalar_param:
                inputs.append(0.01)  # dt
        
        elif spec.op_type == OpType.LOOP:
            # 2D array
            rows, cols = 5, 8
            inputs.append(wp.array(np.random.randn(rows, cols).astype(np.float32), dtype=wp.float32))
            inputs.append(wp.zeros(rows, dtype=wp.float32))
        
        elif spec.op_type == OpType.CONDITIONAL:
            # Arrays for categorization
            inputs.append(wp.array(np.linspace(-2, 2, n).astype(np.float32)))
            inputs.append(wp.zeros(n, dtype=wp.int32))
            inputs.append(wp.zeros(n, dtype=wp.float32))
        
        elif spec.op_type == OpType.ATOMIC:
            # Arrays for atomic operations
            inputs.append(wp.array(np.random.randn(n).astype(np.float32)))
            inputs.append(wp.zeros(2, dtype=wp.float32))
            if spec.has_scalar_param:
                inputs.append(0.5)  # threshold
        
        elif spec.op_type == OpType.ARITHMETIC:
            # Regular float arrays
            for _ in range(spec.num_inputs):
                inputs.append(wp.array(np.random.randn(n).astype(np.float32)))
            for _ in range(spec.num_outputs):
                inputs.append(wp.zeros(n, dtype=wp.float32))
            if spec.has_scalar_param:
                inputs.append(2.0)  # scale
        
        elif spec.op_type == OpType.TRIGONOMETRY:
            # Float arrays for trig functions
            inputs.append(wp.array(np.linspace(-3.14, 3.14, n).astype(np.float32)))
            inputs.append(wp.zeros(n, dtype=wp.float32))
            if spec.has_scalar_param:
                inputs.append(1.0)  # freq
        
        else:
            # Default: single array
            inputs.append(wp.array(np.random.randn(n).astype(np.float32)))
            inputs.append(wp.zeros(n, dtype=wp.float32))
        
        return inputs
    
    def generate_single_pair(self, spec: Optional[KernelSpec] = None, 
                            save: bool = True) -> Optional[Dict]:
        """
        Generate a single Python→IR pair
        
        Args:
            spec: Kernel specification (if None, generates random)
            save: Whether to save to disk
            
        Returns:
            Dictionary with python_code, ir_code, and metadata, or None on failure
        """
        self.stats['total_attempted'] += 1
        
        try:
            # Generate kernel specification
            if spec is None:
                spec = self.generator.generate_random_spec()
            
            # Generate Python code
            python_code = self.generator.generate_kernel(spec)
            
            # Write to temporary file (warp requires files, not exec)
            temp_file = self.output_dir / f"_temp_{spec.name}.py"
            with open(temp_file, 'w') as f:
                f.write("import warp as wp\n\n")
                f.write(python_code)
            
            # Import the kernel
            import importlib.util
            spec_module = importlib.util.spec_from_file_location(f"temp_{spec.name}", temp_file)
            module = importlib.util.module_from_spec(spec_module)
            spec_module.loader.exec_module(module)
            kernel = getattr(module, spec.name)
            
            # Create dummy inputs and compile
            inputs = self._create_dummy_inputs(spec)
            dim = 10 if spec.op_type != OpType.LOOP else 5
            
            # Launch to force compilation
            wp.launch(kernel, dim=dim, inputs=inputs)
            
            # Extract IR
            ir_data = self.extractor.extract_ir(kernel)
            
            # Clean up temp file
            temp_file.unlink()
            
            # Create pair
            pair = {
                'python_code': python_code,
                'ir_code': ir_data['forward_function'],
                'metadata': {
                    'kernel_name': spec.name,
                    'op_type': spec.op_type.value,
                    'complexity': spec.complexity,
                    'num_inputs': spec.num_inputs,
                    'num_outputs': spec.num_outputs,
                    'has_scalar_param': spec.has_scalar_param,
                    'python_lines': len(python_code.split('\n')),
                    'ir_lines': len(ir_data['forward_function'].split('\n')) if ir_data['forward_function'] else 0
                }
            }
            
            # Save if requested
            if save:
                output_file = self.output_dir / f"{spec.name}.json"
                with open(output_file, 'w') as f:
                    json.dump(pair, f, indent=2)
            
            # Update stats
            self.stats['successful'] += 1
            op_type = spec.op_type.value
            self.stats['by_type'][op_type] = self.stats['by_type'].get(op_type, 0) + 1
            
            return pair
            
        except Exception as e:
            self.stats['failed'] += 1
            print(f"Failed to generate pair for {spec.name if spec else 'unknown'}: {e}")
            # Clean up temp file if it exists
            try:
                temp_file.unlink()
            except:
                pass
            # Optionally print traceback for debugging
            # traceback.print_exc()
            return None
    
    def generate_batch(self, count: int, verbose: bool = True) -> List[Dict]:
        """
        Generate multiple pairs
        
        Args:
            count: Number of pairs to generate
            verbose: Print progress
            
        Returns:
            List of successfully generated pairs
        """
        pairs = []
        
        if verbose:
            print(f"Generating {count} kernel pairs...")
            print("=" * 80)
        
        for i in range(count):
            if verbose and (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{count} ({self.stats['successful']} successful, {self.stats['failed']} failed)")
            
            pair = self.generate_single_pair()
            if pair:
                pairs.append(pair)
        
        if verbose:
            print(f"\nGeneration complete!")
            print(f"Successful: {self.stats['successful']}")
            print(f"Failed: {self.stats['failed']}")
            print(f"\nBy operation type:")
            for op_type, count in sorted(self.stats['by_type'].items()):
                print(f"  {op_type:15s}: {count}")
        
        return pairs
    
    def get_stats(self) -> Dict:
        """Return current statistics"""
        return self.stats.copy()


def main():
    """Test the pipeline"""
    print("SYNTHESIS PIPELINE TEST")
    print("=" * 80)
    
    # Create pipeline
    pipeline = SynthesisPipeline(output_dir="data/samples")
    
    # Generate a small batch
    print("\n1. Generating 20 sample pairs...")
    pairs = pipeline.generate_batch(20, verbose=True)
    
    print(f"\n2. Sample pair inspection:")
    if pairs:
        sample = pairs[0]
        print(f"\nPython code (first 300 chars):")
        print(sample['python_code'][:300])
        print(f"\nIR code (first 300 chars):")
        print(sample['ir_code'][:300] if sample['ir_code'] else "None")
        print(f"\nMetadata: {sample['metadata']}")
    
    print("\n" + "=" * 80)
    print("✓ Pipeline test complete")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
