"""End-to-end pipeline: generate function → compile → extract IR → save pair."""

import os
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from generator import generate_random_function, GENERATORS


def extract_ir_pair(fn, source: str, args: tuple) -> dict:
    """Extract jaxpr and HLO from a function."""
    # Get jaxpr
    jaxpr = jax.make_jaxpr(fn)(*args)
    
    # Get HLO text
    lowered = jax.jit(fn).lower(*args)
    hlo_text = lowered.as_text()
    
    # Get shape info
    input_shapes = []
    for arg in args:
        if hasattr(arg, 'shape'):
            input_shapes.append(f"{arg.dtype}{list(arg.shape)}")
        else:
            input_shapes.append(str(type(arg).__name__))
    
    return {
        "python_source": source,
        "jaxpr": str(jaxpr),
        "hlo_text": hlo_text,
        "input_shapes": input_shapes,
    }


def generate_and_extract(seed: int = None) -> dict:
    """Generate a random function and extract its IR."""
    if seed is not None:
        random.seed(seed)
    
    fn, source, args = generate_random_function()
    pair = extract_ir_pair(fn, source, args)
    
    # Add metadata
    pair["generator"] = fn.__name__ if hasattr(fn, '__name__') else "lambda"
    pair["seed"] = seed
    
    return pair


def save_pair(pair: dict, output_dir: str, prefix: str = "pair") -> str:
    """Save a pair to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique ID from content
    content_hash = hashlib.md5(pair["python_source"].encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{prefix}_{content_hash}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(pair, f, indent=2)
    
    return filepath


def run_pipeline(n_samples: int, output_dir: str, verbose: bool = True) -> list:
    """Run the full pipeline to generate n samples."""
    pairs = []
    failed = 0
    
    for i in range(n_samples):
        try:
            pair = generate_and_extract(seed=i)
            filepath = save_pair(pair, output_dir)
            pairs.append(pair)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_samples} pairs")
                
        except Exception as e:
            failed += 1
            if verbose:
                print(f"Failed at {i}: {e}")
    
    if verbose:
        print(f"\nCompleted: {len(pairs)} pairs, {failed} failed")
    
    return pairs


def validate_pairs(pairs: list) -> dict:
    """Validate generated pairs."""
    stats = {
        "total": len(pairs),
        "has_jaxpr": 0,
        "has_hlo": 0,
        "has_source": 0,
        "avg_jaxpr_len": 0,
        "avg_hlo_len": 0,
    }
    
    jaxpr_lens = []
    hlo_lens = []
    
    for pair in pairs:
        if pair.get("jaxpr"):
            stats["has_jaxpr"] += 1
            jaxpr_lens.append(len(pair["jaxpr"]))
        if pair.get("hlo_text"):
            stats["has_hlo"] += 1
            hlo_lens.append(len(pair["hlo_text"]))
        if pair.get("python_source"):
            stats["has_source"] += 1
    
    if jaxpr_lens:
        stats["avg_jaxpr_len"] = sum(jaxpr_lens) // len(jaxpr_lens)
    if hlo_lens:
        stats["avg_hlo_len"] = sum(hlo_lens) // len(hlo_lens)
    
    return stats


if __name__ == "__main__":
    import sys
    
    output_dir = "../../data/samples"
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print(f"Generating {n_samples} Python→IR pairs...")
    pairs = run_pipeline(n_samples, output_dir)
    
    stats = validate_pairs(pairs)
    print(f"\nValidation stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Show sample
    if pairs:
        print(f"\nSample pair:")
        print(f"  Source: {pairs[0]['python_source'][:80]}...")
        print(f"  Jaxpr: {pairs[0]['jaxpr'][:80]}...")
