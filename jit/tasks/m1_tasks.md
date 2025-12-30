# Milestone 1 Tasks

## Task 1: Install JAX
- [ ] Step 1.1: Install JAX (CPU version for now)
- [ ] Step 1.2: Verify installation with basic import
- **Done when**: `import jax; jax.__version__` works

## Task 2: Run Basic JAX Examples
- [ ] Step 2.1: Create examples showing @jax.jit compilation
- [ ] Step 2.2: Create examples with jax.grad (automatic differentiation)
- [ ] Step 2.3: Create examples with jax.vmap (vectorization)
- **Done when**: 3+ examples run successfully

## Task 3: Understand JAX IR Extraction
- [ ] Step 3.1: Use jax.xla_computation to extract HLO
- [ ] Step 3.2: Test with jax.jit and observe compilation artifacts
- [ ] Step 3.3: Explore StableHLO output options
- **Done when**: Can programmatically extract IR from JAX functions

## Task 4: Document Findings
- [ ] Step 4.1: Write notes/jax_basics.md with compilation flow
- [ ] Step 4.2: Document where IR lives and how to access it
- **Done when**: Documentation complete (max 50 lines)
