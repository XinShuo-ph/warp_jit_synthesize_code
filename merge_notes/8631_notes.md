# Branch 8631 Analysis

## Quick Stats
- Milestone: M4 ✓ (IR extraction + synthesis pipeline)
- Data generated: 10,000 pairs
- Pipeline works: Yes
- Generation speed: ~380 samples/second

## Unique Features
- **Expression tree generator**: Random expression trees with depth up to 3
- **Simpler approach**: Single kernel type with varied expressions
- **Fixed signature**: `data: wp.array(dtype=float)` for all kernels
- **Random statements**: 3-8 statements per kernel
- Poisson solver present (similar to 12c4)

## Code Quality
- Clean: Yes
- Tests: Yes (test_extractor.py)
- Docs: Yes (README)

## Generator Approach
- **Philosophy**: Generate variety through random expression trees rather than distinct kernel types
- Uses: +, -, *, /, wp.sin, wp.cos, wp.exp, wp.abs
- Variables: v0, v1, v2, tmp
- Random depth and composition

## File Structure
Similar to 12c4/9177:
```
jit/
├── code/
│   ├── extraction/
│   │   ├── ir_extractor.py
│   │   ├── test_extractor.py
│   │   └── debug_extraction.py
│   ├── synthesis/
│   │   ├── generator.py          # Expression tree approach
│   │   ├── pipeline.py
│   │   └── batch_generator.py
│   └── examples/
│       ├── poisson_solver.py
│       ├── test_poisson.py
│       └── example_*.py
├── data/samples/ (10,000)
```

## Recommended for Merge
- [ ] Expression tree approach - interesting but less systematic than 12c4/9177's typed generators

## Skip
- Generator approach - 12c4's typed generators are more systematic
- Rest similar to 12c4

## Comparison with 12c4/9177
| Feature | 12c4 | 9177 | 8631 |
|---------|------|------|------|
| Generator | 7 typed | 10 typed | Expression trees |
| Data count | 10,500 | 10,270 | 10,000 |
| Diversity | High (types) | Higher (types) | Medium (depth) |
| Systematic | Yes | Yes | No |

## Verdict
**Skip** - 12c4 and 9177's typed generator approach is more systematic and produces more diverse kernel types. Expression tree approach is interesting but less structured.
