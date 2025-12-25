# IR format (M2)

In this repo, **IR** = the generated kernel module source code produced by Warp’s codegen:

- CPU: C++ source from `warp._src.context.ModuleBuilder(...).codegen("cpu")`
- CUDA: CUDA C++ source from `...codegen("cuda")`

## Extractor output (`IRExtractionResult`)
- `device`: resolved Warp device string (e.g. `"cpu"`)
- `codegen_device`: `"cpu"` or `"cuda"` (controls codegen target)
- `kernel_key`: Warp kernel key (Python-qualified name)
- `mangled_name`: Warp’s mangled kernel name base
- `module_name`: Warp module name containing the kernel
- `module_hash`: content hash for the module (hex)
- `source`: full generated module source (string)

## Pair samples
`jit/data/samples/m2_pairs.jsonl` stores one JSON object per kernel with:
- `python`: `inspect.getsource(kernel.func)`
- `ir`: `IRExtractionResult.source`

