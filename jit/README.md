# Warp JIT Code Synthesis - cursor/instructions-wrapup-completion-0888

## Progress Summary
- **Milestone reached**: M5 (Complete)
- **Key deliverables**:
  - IR extraction utility for Warp kernels
  - Kernel generator with randomized expressions
  - End-to-end synthesis pipeline
  - 10,000+ Python-IR training pairs
  - FEM Poisson solver example with convergence tests

## What Works
- **IR Extraction** (`code/extraction/ir_extractor.py`): Extracts Python source and C++-like IR from Warp kernels. Handles math ops, control flow, loops, structs, and function calls.
- **Kernel Generator** (`code/synthesis/generator.py`): Generates random Warp kernels with arithmetic expressions using `+`, `-`, `*`, `/` and functions like `wp.sin`, `wp.cos`, `wp.exp`, `wp.abs`.
- **Synthesis Pipeline** (`code/synthesis/pipeline.py`): Orchestrates generation of Python-IR pairs, dynamically importing and extracting IR from generated kernels.
- **FEM Example** (`code/examples/poisson_solver.py`): Working Poisson equation solver using Warp's FEM module, with verified 2nd-order convergence.

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Test IR extraction
python3 jit/code/extraction/ir_extractor.py

# Generate a random kernel
python3 jit/code/synthesis/generator.py

# Generate a Python-IR pair sample
python3 jit/code/synthesis/pipeline.py

# Run extraction tests (5 tests)
python3 jit/code/extraction/test_extractor.py

# Run Poisson solver convergence test
python3 jit/code/examples/test_poisson.py
```

## File Structure

```
jit/
├── code/
│   ├── extraction/           # IR extraction utilities
│   │   ├── ir_extractor.py   # Core extraction function
│   │   ├── test_extractor.py # Unit tests (math, control flow, loops, structs, func calls)
│   │   └── debug_extraction.py
│   ├── synthesis/            # Code generation
│   │   ├── generator.py      # Random kernel generator
│   │   ├── pipeline.py       # End-to-end pair generation
│   │   └── batch_generator.py
│   └── examples/             # Warp examples
│       ├── poisson_solver.py # FEM Poisson solver
│       ├── test_poisson.py   # Convergence test
│       └── example_*.py      # Basic Warp examples
├── data/
│   └── samples/              # Generated Python-IR pairs (10k+ JSON files)
├── notes/
│   ├── ir_format.md          # Documentation of Warp IR format
│   ├── warp_basics.md        # Warp fundamentals
│   └── data_stats.md         # Dataset statistics
├── tasks/                    # Milestone task definitions
├── STATE.md                  # Project state tracker
├── WRAPUP_STATE.md           # Wrapup progress tracker
└── instructions.md           # Original project instructions
```

## Generated Data Format

Each sample in `data/samples/` is a JSON file:

```json
{
  "id": "kernel_abc12345",
  "name": "kernel_abc12345",
  "python_source": "@wp.kernel\ndef kernel_abc12345(data: wp.array(dtype=float)):\n    tid = wp.tid()\n    v0 = data[tid]\n    ...",
  "ir": [
    "// Block 0",
    "var_0 = builtin_tid1d();",
    "var_1 = wp::address(var_data, var_0);",
    "var_2 = wp::load(var_1);",
    "..."
  ],
  "args": {
    "data": "array(dtype=float32)"
  }
}
```

### IR Characteristics
- **SSA Form**: Variables named `var_N`, assigned once
- **Builtins**: `wp::add`, `wp::mul`, `wp::sin`, `wp::cos`, `wp::exp`, `wp::abs`, etc.
- **Memory**: `wp::address()`, `wp::load()`, `wp::array_store()`
- **Thread ID**: `builtin_tid1d()`

## Known Issues / TODOs
- **Large dataset in git**: 10,102 samples exceed recommended ≤100 for git. Consider moving to external storage or using `.gitignore`.
- **CPU-only tested**: No GPU available in current environment. CUDA output not validated.
- **Limited kernel variety**: Generator only produces simple arithmetic kernels with fixed signature `(data: wp.array(dtype=float))`.
- **No control flow in generator**: If/else and loops not yet supported in random generation (though extractor handles them).
- **Temp file cleanup**: `synthesis/temp/` contains 10k+ cached `.pyc` files from generation.

## Next Steps (from STATE.md)
- Train an LLM on the generated data
- Expand generator to support control flow (if/else, loops) and complex types (structs)
- Implement more complex FEM examples (elasticity, fluid dynamics)
