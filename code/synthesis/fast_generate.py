#!/usr/bin/env python3
"""Fast batch generation script - generates multiple small batches."""

import sys
sys.path.insert(0, '/workspace/code')

import warp as wp
from synthesis.pipeline import SynthesisPipeline
import json
import os

wp.init()

def generate_batch(batch_num, count, seed):
    """Generate a single batch."""
    pipeline = SynthesisPipeline(seed=seed)
    pairs = pipeline.generate_dataset(count=count, verbose=False)
    
    output_file = f'/workspace/data/batch_{batch_num}.json'
    pipeline.save_dataset(pairs, output_file)
    return len(pairs)

# Generate 5 batches of 1800 pairs each = 9000 pairs
# Plus existing 1200 = 10,200 total
total_generated = 0
for i in range(2, 7):
    print(f"Generating batch {i}...")
    count = generate_batch(i, 1800, seed=42 + i * 100)
    total_generated += count
    print(f"  Batch {i}: {count} pairs")

print(f"\nTotal new pairs: {total_generated}")
print(f"Combined with existing 1200: {total_generated + 1200} pairs")
