# Dataset Statistics

## Current (data/generated/): 266 pairs
- Generation rate: ~0.3-0.7 pairs/s (CPU-only)
- Unique: 266 (no duplicates)

## Type Distribution (balanced: 38 each)
arithmetic, math, loop, conditional, vector, matrix, combined

## Size Stats
- Avg source: 178 chars | Avg IR: 384k chars
- IR range: [5k, 913k] chars

## Scaling to 10k+
Time: ~9h at 0.3 pairs/s
Command: `python3 batch_generator.py --count 10000 --output ../../data/generated --resume`
