# Milestone 1 Tasks: Environment Setup & JAX Basics

## Task 1: Install JAX and Verify Setup
- [ ] Step 1.1: Install JAX with `pip install jax jaxlib numpy`
- [ ] Step 1.2: Verify installation by importing jax and checking version
- [ ] Step 1.3: Check available backends (CPU/GPU/TPU)
- **Done when**: Can import jax, run a simple jitted function, and print version info

## Task 2: Basic JIT Compilation Examples
- [ ] Step 2.1: Create `code/examples/01_simple_jit.py` with basic @jit example
- [ ] Step 2.2: Create `code/examples/02_array_ops.py` with array operations
- [ ] Step 2.3: Create `code/examples/03_control_flow.py` with lax.cond/scan
- **Done when**: All 3 examples run successfully twice with identical output

## Task 3: IR Extraction Exploration
- [ ] Step 3.1: Use `make_jaxpr()` to extract Jaxpr from a simple function
- [ ] Step 3.2: Use `xla_computation()` to extract HLO from a simple function
- [ ] Step 3.3: Compare both IR formats and save examples
- **Done when**: Can programmatically extract both Jaxpr and HLO, saved to files

## Task 4: Document JAX Compilation Flow
- [ ] Step 4.1: Trace through JAX JIT compilation pipeline
- [ ] Step 4.2: Document Jaxpr → HLO → XLA optimization flow
- [ ] Step 4.3: Create `notes/jax_basics.md` (max 50 lines)
- **Done when**: Document covers: @jit decorator, IR formats, extraction methods

## Success Criteria for M1
- [ ] JAX installed and working
- [ ] 3+ working examples demonstrating different JAX features
- [ ] Can extract both Jaxpr and HLO IR programmatically
- [ ] `notes/jax_basics.md` exists and is accurate
- [ ] All code runs twice with identical results
