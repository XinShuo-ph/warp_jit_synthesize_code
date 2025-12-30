# Current State
- **Milestone**: M1 (JAX Edition)
- **Task**: Environment Setup & JAX Basics
- **Status**: ready_for_next

## Next Action
1. Install JAX package: `pip install jax jaxlib`
2. Create `jit/` directory structure:
   - `jit/code/examples/`
   - `jit/code/extraction/`
   - `jit/code/synthesis/`
   - `jit/data/`
   - `jit/notes/`
   - `jit/tasks/`
3. Create `jit/tasks/m1_tasks.md` with detailed breakdown
4. Run basic JAX examples to understand:
   - `jax.jit()` compilation
   - `jax.make_jaxpr()` for Jaxpr IR
   - `jax.xla_computation()` for HLO IR

## Blockers
None

## Session Log
- 2025-12-30: Created instructions_jax.md - adapted original Warp instructions for JAX. Ready to begin M1.

## Notes
- Original instructions were for Nvidia Warp package
- Now adapting to use Google JAX instead
- JAX advantages:
  - More widely used in ML/research
  - Multiple IR formats (Jaxpr, HLO, StableHLO)
  - Rich transformation ecosystem (grad, vmap, pmap)
  - Better documentation and examples
  - Functional programming paradigm
