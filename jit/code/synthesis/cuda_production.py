"""
CUDA Production Pipeline
========================

Entry point for mass production of Python-to-CUDA-IR pairs.
Wrapper around batch_generator with CUDA configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow importing batch_generator
sys.path.insert(0, str(Path(__file__).parent))

from batch_generator import run_large_scale_generation

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA IR Production Pipeline")
    parser.add_argument("-n", type=int, default=1000, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="/workspace/jit/data/cuda_production", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=12345, help="Random seed")
    
    args = parser.parse_args()
    
    print("Starting CUDA Production Run...")
    run_large_scale_generation(
        n=args.n,
        output_dir=args.output,
        seed=args.seed,
        device="cuda"
    )

if __name__ == "__main__":
    main()
