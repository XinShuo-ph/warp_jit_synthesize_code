# Milestone 1 Tasks: Environment Setup & JAX Basics

## Task 1: Install and verify JAX
- [x] Step 1.1: Install jax and jaxlib via pip
- [x] Step 1.2: Verify installation by running simple JIT example
- **Done when**: `jax.jit` decorated function runs and returns correct output

## Task 2: Create basic JIT examples
- [x] Step 2.1: Create vector/matrix operations examples
- [x] Step 2.2: Create neural network layer examples
- [x] Step 2.3: Verify all examples run correctly
- **Done when**: `code/examples/basic_jit.py` runs without errors

## Task 3: Explore IR extraction mechanisms
- [x] Step 3.1: Explore `jax.make_jaxpr` for JAXPR extraction
- [x] Step 3.2: Explore `jax.jit().lower().as_text()` for StableHLO
- [x] Step 3.3: Explore `compiler_ir(dialect='hlo').as_hlo_text()` for XLA HLO
- **Done when**: Can extract all three IR formats programmatically

## Task 4: Create IR extractor module
- [x] Step 4.1: Create `code/extraction/ir_extractor.py`
- [x] Step 4.2: Implement `extract_jaxpr`, `extract_stablehlo`, `extract_xla_hlo`
- [x] Step 4.3: Implement `extract_all_ir` that returns all formats
- [x] Step 4.4: Verify extractor works on multiple test functions
- **Done when**: `ir_extractor.py` runs demo without errors

## Task 5: Document findings
- [x] Step 5.1: Create `notes/jax_basics.md` with compilation flow
- **Done when**: Notes file exists and is under 50 lines
