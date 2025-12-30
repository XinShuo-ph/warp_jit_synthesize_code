# Milestone 1: JAX Environment Setup & Basics

## Task 1: Environment Setup & Verification
- [x] Check if `jax` and `jaxlib` are installed.
- [x] Install if missing.
- [x] Verify installation with a simple script (print device).
- **Done when**: JAX can run a simple operation on the available device (CPU or GPU).

## Task 2: JAX Compilation & IR (HLO) Exploration
- [x] Create basic `jax.jit` example.
- [x] specific action: Extract HLO (High Level Optimizer) IR from the jitted function.
- [x] specific action: Extract StableHLO if possible (as it's more portable).
- **Done when**: We have a script that takes a python function, jits it, and prints/saves the HLO/IR.

## Task 3: Reproduce "Kernel" Logic
- [x] Implement a simple element-wise operation (like Warp kernels) in JAX.
- [x] Implement a loop-based operation (scan/fori_loop).
- **Done when**: We have equivalent examples to basic Warp kernels running in JAX.
