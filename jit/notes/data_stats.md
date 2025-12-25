# Dataset Statistics

- **Total Samples**: 10,000
- **Average IR Lines**: 48.4
- **Format**: JSONL (id, kernel_name, python_code, ir_code)

## Strategy Distribution
- **elementwise**: 28.0%
- **nested_loop**: 14.7%
- **vec3_op**: 14.5%
- **loop**: 14.4%
- **conditional**: 13.7%
- **atomic_accumulate**: 13.6%
- **complex_math**: 1.1% (detected via strict heuristic)

## Notes
- Generated using `batch_generator.py` with 4 workers.
- IR includes CUDA C++ kernel code extracted via `warp.codegen`.
