# Branch 25e7 Analysis

## Quick Stats
- **Milestone**: M5 complete (per `README.md`)
- **Data generated**: 10,000 pairs stored as a single JSON (`data/dataset_10k.json`, ~18.9MB)

## Unique Features
- **Fast generation utility**: `code/synthesis/fast_generate.py` + top-level scripts (`create_10k_dataset.py`, `generate_remaining.py`) aimed at scaling quickly.
- **“Single-file dataset” format**: JSON with top-level `metadata` and `pairs` list (good for streaming into training, but heavy for git).

## Recommended for Merge
- [ ] Mine `fast_generate.py` / dataset scripting ideas if we want a “one-command generate 10k” mode.

## Skip / Handle Carefully
- **Large single-file datasets** (`data/dataset_10k.json`, etc.) should not be merged into the final repo; keep ≤100 samples.
- Exclude `__pycache__/` and other generated artifacts.

