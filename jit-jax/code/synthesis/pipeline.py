"""End-to-end pipeline for generating Python→IR training data."""
import json
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'extraction'))

import jax
import jax.numpy as jnp
from generator import FunctionGenerator
from ir_extractor import extract_ir, create_ir_pair


def generate_pair(
    fn,
    source: str,
    meta: dict,
    ir_format: str = "stablehlo"
) -> Optional[dict]:
    """Generate a single Python→IR pair.
    
    Args:
        fn: The JAX function
        source: Python source code
        meta: Metadata including args and type
        ir_format: IR format to extract
        
    Returns:
        Dictionary with pair data or None if extraction fails
    """
    try:
        # Extract IR
        ir = extract_ir(fn, *meta['args'], format=ir_format)
        
        # Build pair record
        pair = {
            'python': source,
            'ir': ir,
            'ir_format': ir_format,
            'function_type': meta['type'],
            'arg_shapes': [
                {'shape': list(arg.shape), 'dtype': str(arg.dtype)}
                for arg in meta['args']
            ],
        }
        
        return pair
        
    except Exception as e:
        print(f"Error extracting IR: {e}")
        return None


def save_pair(pair: dict, output_dir: Path, index: int) -> str:
    """Save a pair to a JSON file.
    
    Args:
        pair: The pair data
        output_dir: Directory to save to
        index: Pair index for filename
        
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"pair_{index:05d}.json"
    
    with open(filepath, 'w') as f:
        json.dump(pair, f, indent=2)
    
    return str(filepath)


def run_pipeline(
    n_pairs: int = 100,
    output_dir: str = "data/samples",
    ir_format: str = "stablehlo",
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """Run the full synthesis pipeline.
    
    Args:
        n_pairs: Number of pairs to generate
        output_dir: Output directory
        ir_format: IR format to extract
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Summary statistics
    """
    output_path = Path(__file__).parent.parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = FunctionGenerator(seed=seed)
    
    stats = {
        'total_attempted': 0,
        'total_generated': 0,
        'by_type': {},
        'failures': 0,
        'output_dir': str(output_path),
    }
    
    if verbose:
        print(f"Generating {n_pairs} Python→IR pairs...")
        print(f"Output directory: {output_path}")
        print(f"IR format: {ir_format}")
        print("-" * 50)
    
    for i in range(n_pairs):
        stats['total_attempted'] += 1
        
        try:
            # Generate function
            fn, source, meta = generator.generate_random()
            
            # Generate pair
            pair = generate_pair(fn, source, meta, ir_format)
            
            if pair is not None:
                # Save pair
                save_pair(pair, output_path, i)
                stats['total_generated'] += 1
                
                # Update type stats
                fn_type = meta['type']
                stats['by_type'][fn_type] = stats['by_type'].get(fn_type, 0) + 1
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{n_pairs} pairs...")
            else:
                stats['failures'] += 1
                
        except Exception as e:
            stats['failures'] += 1
            if verbose:
                print(f"Error at pair {i}: {e}")
    
    if verbose:
        print("-" * 50)
        print(f"Done! Generated {stats['total_generated']}/{stats['total_attempted']} pairs")
        print(f"By type: {stats['by_type']}")
        if stats['failures'] > 0:
            print(f"Failures: {stats['failures']}")
    
    # Save summary
    summary_path = output_path / "summary.json"
    stats['timestamp'] = datetime.now().isoformat()
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def load_pair(filepath: str) -> dict:
    """Load a pair from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def validate_pairs(output_dir: str = "data/samples", sample_size: int = 10) -> bool:
    """Validate generated pairs."""
    output_path = Path(__file__).parent.parent.parent / output_dir
    
    pair_files = sorted(output_path.glob("pair_*.json"))[:sample_size]
    
    print(f"Validating {len(pair_files)} pairs...")
    
    valid = 0
    for filepath in pair_files:
        pair = load_pair(filepath)
        
        # Check required fields
        has_python = 'python' in pair and pair['python']
        has_ir = 'ir' in pair and pair['ir']
        has_format = 'ir_format' in pair
        
        if has_python and has_ir and has_format:
            valid += 1
            print(f"✓ {filepath.name}: {pair['function_type']}")
        else:
            print(f"✗ {filepath.name}: missing fields")
    
    print(f"\nValid: {valid}/{len(pair_files)}")
    return valid == len(pair_files)


if __name__ == "__main__":
    # Generate 10 pairs as test
    print("=" * 60)
    print("Testing pipeline with 10 pairs")
    print("=" * 60)
    
    stats = run_pipeline(n_pairs=10, seed=42)
    
    print("\n" + "=" * 60)
    print("Validating generated pairs")
    print("=" * 60)
    
    validate_pairs(sample_size=10)
    
    print("\n✓ Pipeline test complete!")
