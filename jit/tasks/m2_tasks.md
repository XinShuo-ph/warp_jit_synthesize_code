# Milestone 2 Tasks (IR Extraction Mechanism)

## Task 1: Identify extractable IR artifact
- [x] Step 1.1: Confirm Warp writes generated sources to kernel cache (`.cpp` for CPU / `.cu` for CUDA)
- [x] Step 1.2: Confirm programmatic path: `warp._src.context.ModuleBuilder(...).codegen(device)`
- **Done when**: We can generate a source string for a simple `@wp.kernel(module="unique")` without reading cache files.

## Task 2: Implement IR extractor utility
- [x] Step 2.1: Add `jit/code/extraction/ir_extractor.py` with `extract_ir(kernel, device=...) -> dict`
- [x] Step 2.2: Return includes `device`, `kernel_key`, `module_name`, `module_hash`, `source` (string)
- **Done when**: Extractor works on CPU for at least one kernel and returns non-empty source containing the kernel entry symbol.

## Task 3: Create 5+ kernel→IR test pairs
- [x] Step 3.1: Add `jit/code/extraction/m2_generate_pairs.py` defining 5+ small unique-module kernels
- [x] Step 3.2: Save `jit/data/samples/m2_pairs.jsonl` with `{name, python, ir, meta}` per line
- [x] Step 3.3: Run generator twice and confirm identical file hash/size
- **Done when**: `m2_pairs.jsonl` has >=5 lines and reruns are identical.

## Task 4: Document IR shape
- [x] Step 4.1: Write `jit/notes/ir_format.md` (<= 30 lines) describing exactly what “IR” is here and key fields
- **Done when**: Notes file exists and stays within the line limit.

