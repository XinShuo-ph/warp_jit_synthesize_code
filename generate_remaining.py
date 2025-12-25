#!/usr/bin/env python3
"""Generate remaining pairs to reach 10k target."""
import sys
sys.path.insert(0, '/workspace/code')
import warp as wp
from synthesis.pipeline import SynthesisPipeline

wp.init()

# We have 1400, need 8600 more
# Generate in one batch
print("Generating 8600 pairs to reach 10k target...")
print("This will take approximately 14-15 minutes at 10 pairs/sec")
print("=" * 70)

pipeline = SynthesisPipeline(seed=999)
pairs = pipeline.generate_dataset(count=8600, verbose=True)

print("\n" + "=" * 70)
pipeline.save_dataset(pairs, '/workspace/data/batch_large.json')
print(f"✓ Generated {len(pairs)} pairs successfully")
