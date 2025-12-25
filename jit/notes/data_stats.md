# Dataset Statistics

## Overview
- **Total Samples**: 10,000
- **Format**: JSON pairs (Python Source + Warp IR)
- **Location**: `jit/data/samples/`

## Generation Performance
- **Time**: ~26 seconds
- **Rate**: ~380 samples/second (single threaded)

## Content
- **Input**: Randomly generated Python kernels using basic arithmetic (+, -, *, /) and math functions (sin, cos, exp, abs).
- **Output**: Corresponding Intermediate Representation (IR) extracted from Warp's compilation pipeline.
- **Structure**:
    - `python_source`: Valid `@wp.kernel` code.
    - `ir`: List of C++ statements from `Adjoint.blocks`.
    - `args`: Kernel argument types.

## Diversity
- Random expression trees with depth up to 3.
- Random number of statements (3-8).
- Variable reuse and constant literals.
