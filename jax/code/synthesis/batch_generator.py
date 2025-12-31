"""Batch generator for large-scale Python→HLO pair generation."""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
from generator import FunctionGenerator
from ir_extractor import extract_all_ir


def generate_single_pair(args):
    """Generate a single pair (for use in parallel execution).
    
    Args:
        args: Tuple of (index, seed)
    
    Returns:
        Tuple of (index, pair_dict) or (index, None) on failure
    """
    idx, seed = args
    
    try:
        gen = FunctionGenerator(seed=seed + idx)
        fn_name, code, fn, example_args = gen.generate_random()
        
        ir = extract_all_ir(fn, *example_args, include_source=False)
        
        input_shapes = [list(a.shape) for a in example_args]
        output = jax.jit(fn)(*example_args)
        output_shape = list(output.shape) if hasattr(output, 'shape') else []
        
        pair = {
            "id": idx,
            "name": fn_name,
            "python_source": code,
            "jaxpr": ir.jaxpr,
            "stablehlo": ir.stablehlo,
            "xla_hlo": ir.xla_hlo,
            "input_shapes": input_shapes,
            "output_shape": output_shape,
        }
        
        return idx, pair
        
    except Exception as e:
        return idx, None


class BatchGenerator:
    """High-throughput batch generator for Python→HLO pairs."""
    
    def __init__(self, output_dir: str = "../../data", seed: int = 42):
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "categories": {}
        }
    
    def generate_batch_sequential(self, n: int, start_id: int = 0,
                                    progress_interval: int = 100) -> List[Dict]:
        """Generate pairs sequentially (simpler, works reliably with JAX).
        
        Args:
            n: Number of pairs to generate
            start_id: Starting ID for pairs
            progress_interval: Print progress every N pairs
        
        Returns:
            List of generated pairs
        """
        pairs = []
        gen = FunctionGenerator(seed=self.seed)
        
        for i in range(n):
            idx = start_id + i
            self.stats["total"] += 1
            
            try:
                fn_name, code, fn, example_args = gen.generate_random()
                ir = extract_all_ir(fn, *example_args, include_source=False)
                
                input_shapes = [list(a.shape) for a in example_args]
                output = jax.jit(fn)(*example_args)
                output_shape = list(output.shape) if hasattr(output, 'shape') else []
                
                pair = {
                    "id": idx,
                    "name": fn_name,
                    "python_source": code,
                    "jaxpr": ir.jaxpr,
                    "stablehlo": ir.stablehlo,
                    "xla_hlo": ir.xla_hlo,
                    "input_shapes": input_shapes,
                    "output_shape": output_shape,
                }
                
                pairs.append(pair)
                self.stats["successful"] += 1
                
                # Track category
                category = fn_name.split("_")[0]
                self.stats["categories"][category] = \
                    self.stats["categories"].get(category, 0) + 1
                
            except Exception as e:
                self.stats["failed"] += 1
            
            if (i + 1) % progress_interval == 0:
                pct = (i + 1) / n * 100
                print(f"Progress: {i + 1}/{n} ({pct:.1f}%) - "
                      f"Success rate: {self.stats['successful']/self.stats['total']*100:.1f}%")
        
        return pairs
    
    def save_batch(self, pairs: List[Dict], filename: str = None,
                   as_jsonl: bool = True) -> str:
        """Save batch of pairs.
        
        Args:
            pairs: List of pair dictionaries
            filename: Output filename (optional)
            as_jsonl: If True, save as JSONL (one JSON per line)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "jsonl" if as_jsonl else "json"
            filename = f"batch_{len(pairs)}_{timestamp}.{ext}"
        
        filepath = self.output_dir / filename
        
        if as_jsonl:
            with open(filepath, "w") as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
        else:
            with open(filepath, "w") as f:
                json.dump(pairs, f)
        
        return str(filepath)
    
    def print_stats(self):
        """Print generation statistics."""
        print("\n" + "=" * 60)
        print("BATCH GENERATION STATISTICS")
        print("=" * 60)
        print(f"Total attempted: {self.stats['total']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        if self.stats['total'] > 0:
            print(f"Success rate: {self.stats['successful']/self.stats['total']*100:.1f}%")
        
        print("\nCategory distribution:")
        for cat, count in sorted(self.stats["categories"].items(), 
                                  key=lambda x: -x[1]):
            pct = count / max(1, self.stats['successful']) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")
    
    def get_stats_summary(self) -> Dict:
        """Get statistics as dictionary."""
        return {
            "total_attempted": self.stats['total'],
            "successful": self.stats['successful'],
            "failed": self.stats['failed'],
            "success_rate": self.stats['successful'] / max(1, self.stats['total']),
            "categories": self.stats['categories'],
        }
    
    def run(self, n: int = 10000, save: bool = True) -> List[Dict]:
        """Run batch generation.
        
        Args:
            n: Number of pairs to generate
            save: Whether to save to disk
        
        Returns:
            List of generated pairs
        """
        print(f"Starting batch generation of {n} pairs...")
        start_time = datetime.now()
        
        pairs = self.generate_batch_sequential(n, progress_interval=max(100, n // 20))
        
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = len(pairs) / elapsed
        
        print(f"\nGeneration complete in {elapsed:.1f}s ({rate:.1f} pairs/sec)")
        
        if save and pairs:
            filepath = self.save_batch(pairs)
            print(f"Saved to: {filepath}")
        
        self.print_stats()
        
        return pairs


def main():
    """Main entry point for batch generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→HLO pairs")
    parser.add_argument("-n", "--num", type=int, default=10000,
                        help="Number of pairs to generate")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("-o", "--output", type=str, default="../../data",
                        help="Output directory")
    
    args = parser.parse_args()
    
    generator = BatchGenerator(output_dir=args.output, seed=args.seed)
    generator.run(n=args.num)


if __name__ == "__main__":
    main()
