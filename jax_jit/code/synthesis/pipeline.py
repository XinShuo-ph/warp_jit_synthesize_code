"""End-to-end pipeline: generate function → compile → extract IR → save pair."""
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

import jax.numpy as jnp

import sys
sys.path.insert(0, '/workspace/jax_jit/code/extraction')
sys.path.insert(0, '/workspace/jax_jit/code/synthesis')

from ir_extractor import extract_ir, IRPair
from generator import FunctionGenerator, FunctionSpec, spec_to_callable, generate_example_inputs, spec_to_code


class SynthesisPipeline:
    """Pipeline for generating Python→StableHLO training pairs."""
    
    def __init__(self, output_dir: str = "/workspace/jax_jit/data/samples", seed: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = FunctionGenerator(seed=seed)
        self.generated_count = 0
    
    def generate_single(self, category: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
        """
        Generate a single Python→StableHLO pair.
        
        Args:
            category: Optional category for generation
            save: Whether to save to disk
        
        Returns:
            Dictionary containing the complete pair
        """
        # Generate function spec
        spec = self.generator.generate(category)
        
        # Convert to callable
        func = spec_to_callable(spec)
        
        # Generate example inputs
        inputs = generate_example_inputs(spec.params)
        
        # Extract IR
        try:
            ir_pair = extract_ir(func, *inputs)
        except Exception as e:
            print(f"Warning: Failed to extract IR for {spec.name}: {e}")
            return None
        
        # Create the complete data pair
        python_code = spec_to_code(spec)
        
        pair_data = {
            'function_name': spec.name,
            'category': category or 'mixed',
            'python_source': python_code,
            'stablehlo_ir': ir_pair.stablehlo_ir,
            'cost_analysis': ir_pair.cost_analysis,
            'params': [(name, desc) for name, desc in spec.params],
            'docstring': spec.docstring,
        }
        
        if save:
            self._save_pair(pair_data, spec.name, category)
        
        self.generated_count += 1
        return pair_data
    
    def generate_batch(self, count: int, categories: Optional[list] = None, 
                      save: bool = True, verbose: bool = True) -> list:
        """
        Generate a batch of Python→StableHLO pairs.
        
        Args:
            count: Number of pairs to generate
            categories: Optional list of categories to sample from
            save: Whether to save to disk
            verbose: Whether to print progress
        
        Returns:
            List of pair dictionaries
        """
        pairs = []
        failed = 0
        
        for i in range(count):
            if verbose and (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} pairs (failed: {failed})")
            
            try:
                pair = self.generate_single(category=None, save=save)
                if pair is not None:
                    pairs.append(pair)
                else:
                    failed += 1
            except Exception as e:
                if verbose:
                    print(f"Error generating pair {i + 1}: {e}")
                failed += 1
                continue
        
        if verbose:
            print(f"\nCompleted: {len(pairs)} pairs generated, {failed} failed")
        
        return pairs
    
    def _save_pair(self, pair_data: Dict[str, Any], name: str, category: Optional[str]) -> None:
        """Save a pair to disk as JSON."""
        # Create a unique filename
        content_hash = hashlib.md5(pair_data['python_source'].encode()).hexdigest()[:12]
        category_tag = category or "mixed"
        filename = f"{content_hash}_{category_tag}_{name}.json"
        filepath = self.output_dir / filename
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(pair_data, f, indent=2)
    
    def load_pair(self, filename: str) -> Dict[str, Any]:
        """Load a pair from disk."""
        filepath = self.output_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated pairs."""
        json_files = list(self.output_dir.glob("*.json"))
        
        if not json_files:
            return {
                'total_pairs': 0,
                'categories': {},
                'avg_python_lines': 0,
                'avg_ir_lines': 0,
                'total_flops': 0,
            }
        
        categories = {}
        total_python_lines = 0
        total_ir_lines = 0
        total_flops = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                category = data.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                total_python_lines += len(data['python_source'].splitlines())
                total_ir_lines += len(data['stablehlo_ir'].splitlines())
                total_flops += data.get('cost_analysis', {}).get('flops', 0)
            except Exception:
                continue
        
        count = len(json_files)
        
        return {
            'total_pairs': count,
            'categories': categories,
            'avg_python_lines': total_python_lines / count if count > 0 else 0,
            'avg_ir_lines': total_ir_lines / count if count > 0 else 0,
            'total_flops': total_flops,
            'avg_flops': total_flops / count if count > 0 else 0,
        }
    
    def validate_pair(self, pair_data: Dict[str, Any]) -> bool:
        """
        Validate that a pair is well-formed and the function executes.
        
        Args:
            pair_data: The pair dictionary to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required = ['function_name', 'python_source', 'stablehlo_ir', 'params']
            for field in required:
                if field not in pair_data:
                    print(f"Missing field: {field}")
                    return False
            
            # Check that Python source is not empty
            if not pair_data['python_source'].strip():
                print("Empty Python source")
                return False
            
            # Check that IR is not empty
            if not pair_data['stablehlo_ir'].strip():
                print("Empty StableHLO IR")
                return False
            
            # Try to compile the Python source
            namespace = {'jnp': jnp}
            exec(pair_data['python_source'], namespace)
            func = namespace[pair_data['function_name']]
            
            # Try to execute with example inputs
            inputs = generate_example_inputs(pair_data['params'])
            result = func(*inputs)
            
            # Check result is valid
            if result is None:
                print("Function returned None")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False


def run_pipeline_demo():
    """Run a demo of the synthesis pipeline."""
    print("=" * 80)
    print("SYNTHESIS PIPELINE DEMO")
    print("=" * 80)
    
    pipeline = SynthesisPipeline(seed=42)
    
    # Generate pairs from different categories
    categories = ['arithmetic', 'conditional', 'reduction', 'matrix', 'elementwise', 'broadcasting', 'composite']
    
    print("\nGenerating one pair from each category:")
    print("-" * 80)
    
    for category in categories:
        print(f"\nCategory: {category}")
        pair = pipeline.generate_single(category=category, save=True)
        
        if pair:
            print(f"  Function: {pair['function_name']}")
            print(f"  Python lines: {len(pair['python_source'].splitlines())}")
            print(f"  IR lines: {len(pair['stablehlo_ir'].splitlines())}")
            print(f"  FLOPs: {pair['cost_analysis'].get('flops', 0)}")
            print(f"  Valid: {pipeline.validate_pair(pair)}")
    
    # Get statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = pipeline.get_statistics()
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Categories: {stats['categories']}")
    print(f"Avg Python lines: {stats['avg_python_lines']:.1f}")
    print(f"Avg IR lines: {stats['avg_ir_lines']:.1f}")
    print(f"Avg FLOPs: {stats['avg_flops']:.1f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_demo()
