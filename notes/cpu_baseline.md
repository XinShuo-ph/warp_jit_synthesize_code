# CPU Baseline Documentation

## Overview
Production-ready CPU-based Python→IR pipeline from branch `cursor/agent-work-merge-process-6964`.

## Pipeline Components

### 1. IR Extractor (`code/extraction/ir_extractor.py`)
- **Function**: `extract_ir(kernel, device="cpu", include_backward=True)`
- **Capabilities**: 
  - Device parameter support (cpu/cuda)
  - Forward and backward pass extraction
  - Metadata extraction (kernel name, arg types, etc.)
- **Output**: Dict with python_source, cpp_code, forward_code, backward_code, metadata

### 2. Kernel Generator (`code/synthesis/generator.py`)
- **10 Kernel Types**:
  1. `arithmetic` - Basic arithmetic ops (+, -, *, /, min, max)
  2. `vector` - Vector operations (dot, cross, length, normalize)
  3. `matrix` - Matrix operations (mat-vec, mat-mat multiply)
  4. `control_flow` - Conditionals (if/elif/else)
  5. `math` - Math functions (sin, cos, exp, sqrt, abs, log)
  6. `atomic` - Atomic operations (add, min, max)
  7. `nested_loop` - Nested loops
  8. `multi_conditional` - Multiple conditionals
  9. `combined` - Multi-pattern kernels
  10. `scalar_param` - Kernels with scalar parameters

### 3. Pipeline (`code/synthesis/pipeline.py`)
- Generates kernel specs → compiles → extracts IR → saves JSON
- Command: `python3 code/synthesis/pipeline.py -n <count> -o <dir>`
- Default device: CPU

## CPU IR Output Format

### JSON Structure
```json
{
  "python_source": "...",     // Python kernel source code
  "cpp_forward": "...",        // C++ forward kernel function
  "metadata": {
    "kernel_name": "...",
    "category": "...",
    "description": "...",
    "device": "cpu",
    ...
  }
}
```

### CPU C++ Characteristics
- Function signature: `void <name>_<hash>_cpu_kernel_forward(...)`
- Arguments: `wp::launch_bounds_t dim`, `size_t task_index`, `wp_args_* _wp_args`
- Array types: `wp::array_t<T>`
- Vector types: `wp::vec_t<N, T>`
- Thread ID: `builtin_tid1d()`
- Operations: C++ standard library + warp runtime

### Example CPU Forward Kernel
```cpp
void vec_qahftr_e0a9cc7c_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_vec_qahftr_e0a9cc7c *_wp_args)
{
    // argument vars
    wp::array_t<wp::vec_t<4, wp::float32>> var_a = _wp_args->a;
    wp::array_t<wp::vec_t<4, wp::float32>> var_b = _wp_args->b;
    wp::array_t<wp::float32> var_out = _wp_args->out;
    
    // primal vars
    wp::int32 var_0;
    wp::vec_t<4, wp::float32>* var_1;
    wp::vec_t<4, wp::float32>* var_2;
    wp::float32 var_3;
    wp::vec_t<4, wp::float32> var_4;
    wp::vec_t<4, wp::float32> var_5;
    
    // forward
    var_0 = builtin_tid1d();
    var_1 = wp::address(var_a, var_0);
    var_2 = wp::address(var_b, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::dot(var_4, var_5);
    wp::array_store(var_out, var_0, var_3);
}
```

## Verified Functionality

✅ All 10 kernel types compile and generate CPU IR
✅ Pipeline successfully generates samples
✅ JSON output format is consistent
✅ Device parameter exists in ir_extractor (ready for CUDA)

## Sample Statistics (10 CPU samples generated)
- arithmetic: 1
- atomic: 1  
- combined: 2
- math: 1
- matrix: 2
- scalar_param: 1
- vector: 2

Location: `/workspace/data/cpu_samples/`
