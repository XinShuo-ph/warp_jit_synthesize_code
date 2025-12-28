# Tier 3 Branches Analysis (25e7, 5d09, a4fd)

## 25e7 Analysis
- Milestone: M5 ✓
- Pipeline works: **NO - ModuleNotFoundError: extraction**
- Generator: Module imports OK but can't run

## 5d09 Analysis
- Milestone: M5 ✓
- Data generated: 0 pairs
- Pipeline works: **NO - ModuleNotFoundError: jit**
- Generator: Has hardcoded import paths

## a4fd Analysis
- Milestone: M5 ✓
- Data generated: 1 pair
- Pipeline works: **PARTIAL - Runs but fails to generate**
```
$ python3 pipeline.py --count 3 --output output
Failed to generate pair 0: No module named 'ir_extractor'
Failed to generate pair 1: No module named 'ir_extractor'
Failed to generate pair 2: No module named 'ir_extractor'
Total: 0 pairs generated, 3 failed
```

## Summary
All Tier 3 branches have broken pipelines due to hardcoded import paths.
None are suitable for merge.

## Recommended for Merge
- [ ] None - All have import path issues

## Skip
- All: Broken pipelines, limited unique features
