# Milestone 1 Tasks: Environment Setup & JAX Basics

## Task 1: Environment Setup & Verification
- [x] Step 1.1: Verify JAX installation (check version and backend: CPU/GPU).
- [x] Step 1.2: Create a simple "Hello World" JAX script to ensure ops are working.
- **Done when**: `code/examples/00_hello_jax.py` runs and prints JAX version + device info.

## Task 2: Basic Examples
- [x] Step 2.1: Create `code/examples/01_jit_grad_vmap.py` demonstrating `jit`, `grad`, and `vmap`.
- [x] Step 2.2: Run the example and verify outputs are correct.
- **Done when**: Scripts run successfully and demonstrate core JAX transforms.

## Task 3: Understand Compilation Flow
- [x] Step 3.1: Create `code/examples/02_compilation_stages.py` using `make_jaxpr` and `jit(...).lower().compile()`.
- [x] Step 3.2: Inspect the output of different lowering stages (Jaxpr, HLO).
- [x] Step 3.3: Write `notes/jax_basics.md` summarizing the flow.
- **Done when**: We can programmaticallly access the IR (HLO/Jaxpr) printed in the example.
