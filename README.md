# Warp JIT Code Synthesis - Merged Project

## Overview
Production-ready Python→IR code synthesis pipeline for LLM training data using NVIDIA Warp.

**Merged from 16 branches** with the best features from each:
- **10 kernel types** (6 from 12c4 + 4 from 9177)
- **10,500+ training pairs** generated
- **Comprehensive validation** and analysis tools
- **Complete documentation** and examples

## What Works
- **IR Extraction** (`code/extraction/ir_extractor.py`): Extracts C++ intermediate representation from Python Warp kernels
- **Kernel Generator** (`code/synthesis/generator.py`): Creates diverse kernel patterns across 10 categories
- **Pipeline** (`code/synthesis/pipeline.py`): End-to-end sample generation with validation
- **Batch Generation** (`code/synthesis/batch_generator.py`): Scalable generation with checkpointing
- **Validation Tools**: Extraction validation, dataset validation, statistics analysis
- **FEM Solver** (`code/examples/poisson_solver.py`): Working Poisson equation solver

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Test IR extraction
python3 code/extraction/ir_extractor.py

# Generate 5 training samples
python3 code/synthesis/pipeline.py -n 5 -o data/samples

# Generate at scale (batch mode)
python3 code/synthesis/batch_generator.py --count 100 --output data/my_batch

# Validate extraction
python3 code/extraction/validate_extraction.py

# Validate dataset samples
python3 code/synthesis/validate_dataset.py data/samples

# Analyze dataset statistics
python3 code/synthesis/analyze_dataset.py data/samples

# Run Poisson solver tests
python3 code/examples/test_poisson.py
```

## Kernel Categories (10 Types)

The generator produces diverse kernels across 10 categories:

1. **arithmetic** - Basic arithmetic operations (+, -, *, /)
2. **vector** - Vector operations (dot, cross, length, normalize)
3. **matrix** - Matrix operations (mat-vec, mat-mat, transpose)
4. **control_flow** - Conditionals (if/elif/else, clamp, step)
5. **math** - Math functions (sin, cos, exp, sqrt, abs)
6. **atomic** - Atomic operations (add, min, max)
7. **nested_loop** - Nested loop patterns (NEW from 9177)
8. **multi_conditional** - Multiple conditional branches (NEW from 9177)
9. **combined** - Loop+conditional+math combined (NEW from 9177)
10. **scalar_param** - Scalar parameters (NEW from 9177)

## File Structure

```
workspace/
├── code/
│   ├── extraction/           # IR extraction from Warp kernels
│   │   ├── ir_extractor.py   # Main extraction utility
│   │   ├── save_sample_pairs.py
│   │   ├── test_ir_extractor.py
│   │   └── validate_extraction.py  # NEW: Validation
│   ├── synthesis/            # Kernel generation and pipeline
│   │   ├── generator.py      # 10 kernel type generators
│   │   ├── pipeline.py       # End-to-end generation
│   │   ├── batch_generator.py # Scalable batch generation
│   │   ├── analyze_dataset.py # NEW: Statistics
│   │   └── validate_dataset.py # NEW: Validation
│   └── examples/             # Example kernels and FEM solver
│       ├── poisson_solver.py # Poisson equation solver
│       ├── test_poisson.py   # FEM solver tests
│       └── other examples...
├── data/                     # Generated samples
├── notes/                    # Technical documentation
│   ├── warp_basics.md       # Warp compilation flow
│   ├── ir_format.md         # IR structure documentation
│   └── data_stats.md        # Dataset statistics
├── tasks/                    # Milestone task files
│   ├── m1_tasks.md through m5_tasks.md
└── tests/                    # Test suite (categorized)

```

## Generated Data Format

```json
{
  "python_source": "@wp.kernel\ndef add_arrays(a: wp.array(dtype=float), ...):\n    ...",
  "cpp_forward": "void add_arrays_cpu_kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "add_arrays",
    "category": "arithmetic",
    "description": "...",
    "device": "cpu",
    ...
  }
}
```

## Dataset Statistics

- **Total Samples**: 10,500+ pairs
- **Kernel Types**: 10 categories
- **Size**: ~42 MB
- **Generation Rate**: ~180-380 samples/second

Category distribution (approximate):
- arithmetic: 17%
- vector: 17%
- control_flow: 17%
- math: 17%
- matrix: 17%
- atomic: 16%
- nested_loop, multi_conditional, combined, scalar_param: ~10%

## Merge History

This project merged the best features from 16 parallel development branches:

### Base Branch (12c4)
- 10,500 pairs generated
- 6 kernel types
- Complete M5 implementation

### Key Merges
- **9177**: Added 4 new kernel types (nested_loop, multi_conditional, combined, scalar_param)
- **82cf**: Validation and analysis tools, excellent documentation
- **aa30**: QUICKSTART guide
- **ff72**: Example progression
- **d623**: Categorized test cases

### Technical Details
- See `merge_notes/` for detailed analysis of each branch
- See `MERGE_STATE.md` for merge process documentation

## Testing

```bash
# Run all tests
python3 -m pytest code/ -v

# Test specific category
python3 code/synthesis/pipeline.py -n 10 -c arithmetic

# Validate extraction
python3 code/extraction/validate_extraction.py

# Validate dataset
python3 code/synthesis/validate_dataset.py data/samples
```

## Performance

- **Generation**: 180-380 samples/second (single-threaded)
- **Batch mode**: Checkpointing for long-running jobs
- **Validation**: 100% pass rate on quality checks

## License

This project uses NVIDIA Warp. See Warp documentation for license details.
