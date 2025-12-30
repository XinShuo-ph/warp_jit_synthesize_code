# JAX Dataset Statistics

## Overview
- **Total pairs**: 11,538
- **Format**: Python code → Jaxpr + StableHLO
- **Generation method**: Automated synthesis pipeline

## Category Distribution
- Arithmetic: 2,000 (17.3%)
- Array operations: 2,000 (17.3%)
- Math functions: 2,000 (17.3%)
- Reductions: 2,000 (17.3%)
- Linear algebra: 2,000 (17.3%)
- Composite operations: 1,538 (13.3%)

## IR Size Statistics
- **Jaxpr**: mean 130 chars, range [35, 375]
- **StableHLO**: mean 451 chars, range [230, 883]

## Operation Diversity
- 36 unique operations
- Top operations: dot, outer, normalize, squared_norm, softmax
