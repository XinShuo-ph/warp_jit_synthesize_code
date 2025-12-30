# Objective
Adapt the current production code to use **JAX on CUDA** (GPU backend).
Currently no GPU device is available for agent to use. Provide concise code and command for me to test on GPU device by myself.

# Milestones

1. Reproduce the current production code using **JAX CPU** backend. Study the `cursor/agent-work-merge-process-....` branches and pick one best branch to use as base.

2. Adapt the code to use **JAX CUDA** backend. Do this iteratively, iterate over:
- All generated program types (the “kernel types” equivalent)
- Both forward and backward pass (grad)
- Batch generation pipeline
- Validation tools
- Test suite for you to run later on an actual GPU device
Since there are ~10 kernel types that I can see currently, I expect ~ 20 iterations to complete this milestone. Focus on one iteration, one task at a time.

## GPU test checklist (for user to run)
- Install CUDA-enabled JAX per official instructions for your CUDA version (example pattern):
  - `pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- Verify device:
  - `python -c "import jax; print(jax.devices())"`
- Run a minimal compile + grad:
  - `python -c "import jax, jax.numpy as jnp; f=lambda x: jnp.sin(x).sum(); jf=jax.jit(f); x=jnp.ones((1024,), jnp.float32); print(jf(x)); print(jax.grad(f)(x).shape)"`

