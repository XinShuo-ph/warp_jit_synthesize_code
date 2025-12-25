# Warp basics (M1)

- Kernels are Python functions decorated with `@wp.kernel`.
- A kernel is launched with `wp.launch(kernel, dim=..., inputs=[...], device=...)`.
- `wp.init()` initializes available backends (CUDA if driver present, otherwise CPU).

## What happens on first launch
- First launch triggers JIT compilation of a module containing the kernel.
- Warp reports module load time and whether it was **compiled** or **cached**.
- Compiled artifacts are stored in the kernel cache (shown on init), e.g.:
  - `~/.cache/warp/1.10.1`

## What happens on subsequent launches
- If the kernel/module signature is unchanged, Warp loads from cache (fast “cached” load).

## Observations in this environment
- No NVIDIA driver was present, so Warp runs in **CPU-only mode**.
- CPU kernels still JIT compile and populate the cache, then hit cached loads on reruns.

## Where to look next (for IR/extraction work)
- Warp compilation/translation lives in the Warp package internals; starting points:
  - `warp/context.py` (kernel compilation/launch plumbing)
  - `warp/codegen.py` (code generation pipeline)

