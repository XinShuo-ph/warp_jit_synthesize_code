"""End-to-end synthesis pipeline: generate functions → extract IR → save pairs."""
import os
import sys
import json
import random
from typing import List, Dict, Optional

# Add extraction module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extraction'))

from generator import generate_batch, generate_random_function, GENERATORS
from ir_extractor import extract_jaxpr, extract_hlo


def create_pair(name: str, code: str, fn: callable, example_args: tuple) -> Optional[Dict]:
    """Create a Python→IR pair from a generated function.
    
    Returns None if extraction fails.
    """
    try:
        jaxpr = extract_jaxpr(fn, *example_args)
        hlo = extract_hlo(fn, *example_args)
        
        # Get input shapes
        input_shapes = []
        for arg in example_args:
            if hasattr(arg, 'shape'):
                input_shapes.append(list(arg.shape))
            else:
                input_shapes.append(None)
        
        return {
            "name": name,
            "source": code,
            "jaxpr": jaxpr,
            "hlo": hlo,
            "input_shapes": input_shapes,
        }
    except Exception as e:
        return None


def generate_pairs(n: int, seed: int = None) -> List[Dict]:
    """Generate n Python→IR pairs.
    
    Args:
        n: Number of pairs to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of successfully generated pairs
    """
    if seed is not None:
        random.seed(seed)
    
    pairs = []
    attempts = 0
    max_attempts = n * 3  # Allow some failures
    
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        try:
            name, code, fn, args = generate_random_function()
            pair = create_pair(name, code, fn, args)
            if pair is not None:
                pairs.append(pair)
        except Exception:
            continue
    
    return pairs


def save_pairs(pairs: List[Dict], output_dir: str) -> int:
    """Save pairs to individual JSON files.
    
    Returns number of files saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    
    for pair in pairs:
        filepath = os.path.join(output_dir, f"{pair['name']}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
            saved += 1
        except Exception:
            continue
    
    return saved


def run_pipeline(n: int, output_dir: str, seed: int = None) -> Dict:
    """Run the complete synthesis pipeline.
    
    Args:
        n: Number of pairs to generate
        output_dir: Directory to save output
        seed: Random seed
    
    Returns:
        Statistics dict
    """
    print(f"Generating {n} Python→IR pairs...")
    pairs = generate_pairs(n, seed)
    print(f"Generated {len(pairs)} pairs")
    
    saved = save_pairs(pairs, output_dir)
    print(f"Saved {saved} files to {output_dir}")
    
    # Compute stats
    stats = {
        "requested": n,
        "generated": len(pairs),
        "saved": saved,
        "output_dir": output_dir,
    }
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→IR pairs")
    parser.add_argument("-n", type=int, default=100, help="Number of pairs")
    parser.add_argument("-o", "--output", default="../../data/samples", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    stats = run_pipeline(args.n, args.output, args.seed)
    print(f"\nStats: {stats}")
