# Dataset Statistics

## Overview
- **Total samples**: 200+ Python→IR pairs generated
- **Dataset size**: 400+ KB (JSON format)
- **Generation rate**: ~10 pairs/second

## Python Source
- Min length: 195 chars
- Max length: 326 chars
- Mean length: ~260 chars

## C++ IR
- Min length: ~1171 chars
- Max length: ~1820 chars
- Mean length: ~1528 chars

## Kernel Types (7 patterns)
1. Arithmetic operations (mul, add, sub, div)
2. Array indexing with scaling
3. Conditional logic (if/else)
4. Loops (iteration patterns)
5. Vector operations (vec3: length, dot, normalize)
6. Math functions (sin, cos, exp, sqrt, abs)
7. Multi-operation kernels (combined operations)

## Infrastructure
- Batch generator supports 10k+ with checkpointing
- Pipeline handles module caching and IR extraction
- Automated diversity through randomized generation
