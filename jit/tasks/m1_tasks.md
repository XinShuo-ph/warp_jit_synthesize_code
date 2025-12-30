# Milestone 1 Tasks: Environment Setup & JAX Basics

## Task 1: Install and verify JAX
- [x] Step 1.1: Install jax and jaxlib packages
- [x] Step 1.2: Verify installation with version check
- [x] Step 1.3: Run 3 basic jit examples
- **Done when**: All 3 examples produce correct output

## Task 2: Explore IR extraction APIs
- [x] Step 2.1: Test `jax.make_jaxpr()` for JAXPR extraction
- [x] Step 2.2: Test `.lower().as_text()` for XLA HLO extraction
- [x] Step 2.3: Create example scripts in `code/examples/`
- **Done when**: Can extract both JAXPR and XLA HLO from any jitted function

## Task 3: Document findings
- [x] Step 3.1: Create `notes/jax_basics.md` with compilation flow
- **Done when**: Notes file exists with <50 lines documenting key APIs
