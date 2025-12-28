# Warp JIT Code Synthesis - instructions-wrapup-completion-2b4e

## Progress Summary
- Milestone reached: M3
- Key deliverables:
  - IR extraction mechanism (`ir_extractor.py`)
  - 5 fixture kernels for testing
  - warp.fem Poisson solver with validation tests

## What Works
- **IR Extraction**: `extract_ir()` returns Warp's generated CPU C++ source for any kernel
- **Fixture Kernels**: 5 varied kernels (arithmetic, conditionals, structs, atomics, trig functions)
- **Poisson Solver**: FEM solver for -Δu = f on [0,1]² with full Dirichlet BCs
- **All Tests Pass**: 7 tests (5 IR extraction + 2 Poisson solver)

## Requirements
```bash
pip install warp-lang==1.10.1
pip install pytest  # for running tests
```

## Quick Start
```bash
# Run IR extraction tests
python3 -m pytest jit/code/extraction/test_ir_extractor.py -v

# Run Poisson solver tests
python3 -m pytest jit/code/examples/test_poisson.py -v

# Run Poisson solver directly
python3 jit/code/examples/poisson_solver.py
```

## File Structure
```
jit/
├── code/
│   ├── extraction/          # IR extraction utilities
│   │   ├── ir_extractor.py  # extract_ir() function
│   │   ├── fixture_kernels.py # 5 test kernels
│   │   └── test_ir_extractor.py # pytest tests
│   └── examples/            # FEM examples
│       ├── poisson_solver.py # -Δu = f solver
│       └── test_poisson.py   # validation tests
├── notes/                   # Technical documentation
│   ├── ir_format.md         # IR format documentation
│   └── warp_basics.md       # Warp JIT basics
├── tasks/                   # Milestone task lists
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   └── m3_tasks.md
└── requirements.txt         # Dependencies
```

## Generated Data Format
```json
{
  "kernel_name": "add_constant",
  "python_source": "@wp.kernel\ndef add_constant(a: wp.array(dtype=wp.float32), c: float):\n    i = wp.tid()\n    a[i] = a[i] + c",
  "ir_code": "// Warp-generated C++ source...",
  "device": "cpu"
}
```

## Known Issues / TODOs
- M4 (synthesis pipeline) and M5 (scale up) not yet implemented
- GPU/CUDA support not yet added to ir_extractor (CPU-only currently)
- Uses Warp internal APIs (pin version for reproducibility)
