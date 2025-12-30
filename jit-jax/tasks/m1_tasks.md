# Milestone 1 Tasks: Environment Setup & JAX Basics

## Task 1: Verify JAX Installation
- [x] Step 1.1: Install JAX with `pip install jax jaxlib`
- [x] Step 1.2: Verify version and available devices
- [x] Step 1.3: Run simple jax.jit function
- **Done when**: JAX version prints, devices list, and jitted function returns correct result

## Task 2: Run 3+ JAX Examples
- [x] Step 2.1: Create basic arithmetic example with jax.jit
- [x] Step 2.2: Create matrix operations example
- [x] Step 2.3: Create gradient computation example with jax.grad
- [x] Step 2.4: Run each example twice to verify consistency
- **Done when**: All 3 examples run successfully twice each

## Task 3: Understand JIT Compilation Flow
- [x] Step 3.1: Explore jax.make_jaxpr for JAXPR representation
- [x] Step 3.2: Explore jax.jit().lower() for HLO IR
- [x] Step 3.3: Compare JAXPR vs HLO output formats
- **Done when**: Can extract both JAXPR and HLO from same function

## Task 4: Document Findings
- [x] Step 4.1: Create notes/jax_basics.md with compilation flow
- [x] Step 4.2: Document where IR is extracted from (max 50 lines)
- **Done when**: notes/jax_basics.md exists with key findings
