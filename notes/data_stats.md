# Dataset Statistics

## Overview
- **Total**: 10,000 Python→IR pairs (18.9 MB)
- **Original**: 1,400 unique kernels
- **Variations**: 8,600 variants
- **Rate**: ~10 pairs/second

## Characteristics
- Python: 195-326 chars (mean: 260)
- C++ IR: 1171-1820 chars (mean: 1528)

## Kernel Types (7 patterns)
1. Arithmetic (mul, add, sub, div)
2. Array indexing + scaling
3. Conditional logic (if/else)
4. Loops (iteration)
5. Vector ops (vec3: length, dot, normalize)
6. Math functions (sin, cos, exp, sqrt, abs)
7. Multi-op (combined operations)

## Infrastructure
Batch generator with checkpointing supports large-scale generation.

