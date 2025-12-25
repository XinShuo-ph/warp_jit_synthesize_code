# Dataset Statistics

**Dataset**: `jit/data/large_scale/dataset_large.jsonl`
**Size**: 10,000 samples

## Code Metrics

| Metric | Average Lines | Description |
|--------|---------------|-------------|
| Python Source | ~14.5 | High-level kernel definition |
| C++ Forward | ~54.3 | Generated forward pass code |
| C++ Backward | ~100.5 | Generated adjoint/backward pass code |

## Content Analysis

- **Diversity**: Kernels contain a mix of arithmetic operations (`+`, `-`, `*`), built-in functions (`min`, `max`), type casts (`int()`, `float()`), and array access patterns.
- **Coverage**:
  - ~55% of kernels use `wp.min` / `wp.max`.
  - ~72% involve floating point constants/casts.
  - ~55% involve integer casts.
- **Structure**: All kernels include valid Forward and Backward C++ implementations, providing a rich dataset for learning the autograd transformation logic.

## Generation Speed
- The synthesis pipeline achieves **~1400 samples/second** on a 4-core CPU, leveraging Warp's AST-to-C++ string generation without invoking the backend compiler.
