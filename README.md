# Warp JIT Code Synthesis - cursor/instructions-wrapup-completion-9a32

## Progress Summary
- **Milestone reached**: M2 (IR Extraction Mechanism)
- **Key deliverables**:
  - `ir_extractor.py` - Extracts C++ source code (IR) from Warp kernels
  - `test_extractor.py` - 5 test cases covering arithmetic, loops, conditionals, arrays, and builtins
  - `notes/ir_format.md` - Documentation of the generated IR structure
  - `notes/warp_basics.md` - Documentation of Warp's compilation pipeline

## What Works
- **IR Extraction**: The `get_kernel_ir()` function successfully extracts C++ source code from any `@wp.kernel` decorated function
- **CPU Device Support**: Full support for generating CPU-targeted C++ code
- **Forward & Backward Passes**: Extracts both forward and adjoint (backward) code for automatic differentiation
- **Diverse Kernel Types**: Tested with arithmetic operations, loops (for/range), conditionals (if/else), array access, and Warp builtins (sin, abs, etc.)

## Requirements

```bash
pip install warp-lang
pip install pytest  # for running tests
```

## Quick Start

```bash
# Run the IR extractor demo
python3 code/extraction/ir_extractor.py

# Run all tests
python3 code/extraction/test_extractor.py

# Run examples
python3 code/examples/check_install.py
python3 code/examples/check_codegen.py
```

### Example Usage

```python
import warp as wp
from code.extraction.ir_extractor import get_kernel_ir

wp.init()

@wp.kernel
def my_kernel(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0

ir = get_kernel_ir(my_kernel, device="cpu")
print(ir)
```

## File Structure

```
jit/
├── code/
│   ├── extraction/
│   │   ├── ir_extractor.py     # Core IR extraction function
│   │   ├── test_extractor.py   # Test cases for extraction
│   │   └── debug_loop.py       # Debug script for loop IR
│   └── examples/
│       ├── check_install.py    # Verify warp installation
│       ├── check_codegen.py    # Demonstrate full codegen output
│       └── example_mesh.py     # Complex mesh example (requires USD)
├── notes/
│   ├── ir_format.md            # Documentation of IR structure
│   └── warp_basics.md          # Warp compilation pipeline notes
├── tasks/
│   ├── m1_tasks.md             # Milestone 1 task list (completed)
│   └── m2_tasks.md             # Milestone 2 task list (completed)
├── STATE.md                    # Project state tracker
├── WRAPUP_STATE.md             # Wrapup session state
├── instructions.md             # Original project instructions
└── instructions_wrapup.md      # Wrapup phase instructions
```

## Generated IR Format

The extracted IR is C++ source code with the following structure:

```cpp
// Argument struct
struct wp_args_kernel_name_HASH {
    wp::array_t<wp::float32> x;
    wp::int32 n;
};

// Forward pass
void kernel_name_HASH_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_name_HASH *_wp_args)
{
    // Argument unpacking
    wp::array_t<wp::float32> var_x = _wp_args->x;
    
    // Primal variables (SSA form)
    wp::int32 var_0;
    wp::float32 var_1;
    
    // Forward logic with source line comments
    var_0 = builtin_tid1d();  // tid = wp.tid()
    // ... operations ...
}

// Backward pass (for autodiff)
void kernel_name_HASH_cpu_kernel_backward(...) { ... }
```

### Key IR Characteristics
- **Variable naming**: Arguments preserve names (`var_x`), locals are numbered (`var_0`, `var_1`)
- **Operations**: Use `wp::` namespace functions (`wp::add`, `wp::load`, `wp::sin`)
- **Loops**: Python `for` becomes `goto`-based loops with `start_for_N`/`end_for_N` labels
- **Comments**: Source line references like `// c[tid] = a[tid] + b[tid] <L 10>`

## Known Issues / TODOs
- GPU/CUDA support exists in `ir_extractor.py` but not tested (no GPU available)
- `example_mesh.py` requires the USD (pxr) library which is not installed by default
- No data generation pipeline yet (M3+ scope)
- No synthesis/generation of new kernels yet (M4+ scope)
