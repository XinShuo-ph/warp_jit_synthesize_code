# Branch ff72 Analysis

## Quick Stats
- **Milestone**: M5 complete (per `jit/README.md`)
- **Data generated**: 371+ pairs (split across `jit/data/samples/` and `jit/data/generated/`)
- **Kernel types**: 7 (arithmetic, math, loop, conditional, vector, matrix, combined)

## Notable Differences
- **Generator + pipeline are relatively compact** and emphasize a clear mapping from Python patterns → extracted C++ IR.
- **IR extraction via warp cache** in `jit/code/synthesis/pipeline.py` (similar spirit to `9177`), by locating the `.cpp`/`.cu` output under `~/.cache/warp/...`.
- **Nice kernel-type taxonomy** that includes `matrix` (mat33 @ vec3) and “combined”.

## Recommended for Merge
- [ ] `jit/code/synthesis/generator.py` - consider mining/merging kernel templates (esp. `matrix`, `combined`) if not already covered by the final base.
- [ ] `jit/code/synthesis/pipeline.py` - cache-based IR extraction approach is a viable alternative to direct codegen.
- [ ] `jit/code/extraction/test_ir_extractor.py` - useful set of 5 IR tests by category.

## Skip / Handle Carefully
- **Large datasets**: `jit/data/generated/` can grow large; keep ≤100 samples in final.
- **Bytecode artifacts**: `__pycache__/` and `*.pyc`.

