# Milestone 2 Tasks

## Task 1: Decide the IR representation to extract
- [x] Step 1.1: Identify candidate representations on CPU (e.g., generated C++ source via `ModuleBuilder.codegen("cpu")`, cache artifacts under `~/.cache/warp/...`)
- [x] Step 1.2: Pick one representation that is stable + string-serializable (document tradeoffs in `jit/notes/ir_format.md`)
- **Done when**: A single IR representation is chosen and documented (<= 30 lines) in `jit/notes/ir_format.md`.

## Task 2: Implement programmatic extractor
- [x] Step 2.1: Create `jit/code/extraction/ir_extractor.py`
- [x] Step 2.2: Implement `extract_ir(kernel: wp.Kernel, device: str = "cpu") -> str`
- [x] Step 2.3: Add a tiny kernel fixture and verify extractor returns non-empty deterministic output across 2 runs
- **Done when**: `extract_ir(...)` returns a non-empty string for a trivial kernel on CPU in two consecutive runs.

## Task 3: Build 5+ Python→IR paired test cases
- [x] Step 3.1: Add `jit/code/extraction/test_ir_extractor.py` with 5 kernels covering: scalar ops, control flow, structs, atomics, simple math
- [x] Step 3.2: For each kernel, assert extracted IR contains the mangled kernel symbol and is stable (hash equality) across two extractions
- **Done when**: 5 tests pass twice consecutively.

## Task 4: Minimal IR format documentation
- [x] Step 4.1: Write `jit/notes/ir_format.md` (<= 30 lines): what the extracted IR is, how to reproduce, key invariants for dataset generation
- **Done when**: `jit/notes/ir_format.md` exists and is <= 30 lines.

