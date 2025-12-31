"""End-to-end pipeline for generating Python→HLO training pairs."""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
from generator import FunctionGenerator
from ir_extractor import extract_all_ir, extract_xla_hlo


class SynthesisPipeline:
    """Pipeline for generating Python→HLO training data pairs."""
    
    def __init__(self, output_dir: str = "../../data/samples", seed: int = None):
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = FunctionGenerator(seed=seed)
        self.stats = {
            "generated": 0,
            "successful": 0,
            "failed": 0,
            "categories": {}
        }
    
    def generate_pair(self) -> Dict[str, Any]:
        """Generate a single Python→HLO pair.
        
        Returns:
            Dict with keys: name, python_source, jaxpr, stablehlo, xla_hlo, shapes
        """
        fn_name, code, fn, args = self.generator.generate_random()
        
        # Extract all IR representations
        ir = extract_all_ir(fn, *args, include_source=False)
        
        # Get input/output shapes
        input_shapes = [list(a.shape) for a in args]
        output = jax.jit(fn)(*args)
        output_shape = list(output.shape) if hasattr(output, 'shape') else []
        
        return {
            "name": fn_name,
            "python_source": code,
            "jaxpr": ir.jaxpr,
            "stablehlo": ir.stablehlo,
            "xla_hlo": ir.xla_hlo,
            "input_shapes": input_shapes,
            "output_shape": output_shape,
        }
    
    def generate_batch(self, n: int, verbose: bool = True) -> List[Dict[str, Any]]:
        """Generate a batch of pairs.
        
        Args:
            n: Number of pairs to generate
            verbose: Print progress
        
        Returns:
            List of pair dictionaries
        """
        pairs = []
        
        for i in range(n):
            self.stats["generated"] += 1
            
            try:
                pair = self.generate_pair()
                pairs.append(pair)
                self.stats["successful"] += 1
                
                # Track category
                category = pair["name"].split("_")[0]
                self.stats["categories"][category] = \
                    self.stats["categories"].get(category, 0) + 1
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{n} pairs...")
                    
            except Exception as e:
                self.stats["failed"] += 1
                if verbose:
                    print(f"Failed to generate pair {i + 1}: {e}")
        
        return pairs
    
    def save_pairs(self, pairs: List[Dict[str, Any]], filename: str = None) -> str:
        """Save pairs to JSON file.
        
        Args:
            pairs: List of pair dictionaries
            filename: Optional filename, defaults to timestamp
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pairs_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(pairs, f, indent=2)
        
        return str(filepath)
    
    def save_individual_pairs(self, pairs: List[Dict[str, Any]]) -> List[str]:
        """Save each pair as an individual JSON file.
        
        Args:
            pairs: List of pair dictionaries
        
        Returns:
            List of saved file paths
        """
        paths = []
        
        for i, pair in enumerate(pairs):
            filename = f"pair_{i:04d}_{pair['name']}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, "w") as f:
                json.dump(pair, f, indent=2)
            
            paths.append(str(filepath))
        
        return paths
    
    def print_stats(self):
        """Print generation statistics."""
        print("\n" + "=" * 50)
        print("GENERATION STATISTICS")
        print("=" * 50)
        print(f"Total attempted: {self.stats['generated']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['successful']/max(1, self.stats['generated'])*100:.1f}%")
        print("\nBy category:")
        for cat, count in sorted(self.stats["categories"].items()):
            print(f"  {cat}: {count}")
    
    def run(self, n: int = 100, save_individual: bool = True) -> List[Dict[str, Any]]:
        """Run the full pipeline.
        
        Args:
            n: Number of pairs to generate
            save_individual: Whether to save individual files
        
        Returns:
            List of generated pairs
        """
        print(f"Generating {n} Python→HLO pairs...")
        
        pairs = self.generate_batch(n)
        
        # Save as single batch file
        batch_path = self.save_pairs(pairs)
        print(f"\nSaved batch to: {batch_path}")
        
        # Optionally save individual files
        if save_individual:
            paths = self.save_individual_pairs(pairs)
            print(f"Saved {len(paths)} individual pair files")
        
        self.print_stats()
        
        return pairs


def demo():
    """Demonstrate the pipeline with a small batch."""
    print("Running synthesis pipeline demo...\n")
    
    pipeline = SynthesisPipeline(seed=42)
    
    # Generate a few pairs
    pairs = pipeline.run(n=10, save_individual=True)
    
    # Show a sample
    print("\n" + "=" * 50)
    print("SAMPLE PAIR")
    print("=" * 50)
    sample = pairs[0]
    print(f"Name: {sample['name']}")
    print(f"\nPython Source:\n{sample['python_source']}")
    print(f"\nInput shapes: {sample['input_shapes']}")
    print(f"Output shape: {sample['output_shape']}")
    print(f"\nXLA HLO (first 400 chars):\n{sample['xla_hlo'][:400]}...")


if __name__ == "__main__":
    demo()
