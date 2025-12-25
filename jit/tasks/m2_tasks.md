# Milestone 2 Tasks

## Task 1: Define “IR” artifact(s) we will extract
- [x] Step 1.1: Decide the default IR format for extraction on CPU-only environments (generated `.cpp` from kernel cache).
- [x] Step 1.2: Decide optional formats for CUDA-capable environments (`.cu`, `.ptx`, `.cubin` if present).
- [x] Step 1.3: Write `notes/ir_format.md` (≤ 30 lines) describing what is extracted and where it comes from.
- **Done when**: `notes/ir_format.md` exists and matches the extractor’s behavior.

## Task 2: Implement kernel→IR extraction utility
- [x] Step 2.1: Create `code/extraction/ir_extractor.py` with a single public function `extract_ir(kernel, device="cpu", prefer=("cpp","cu","ptx")) -> str`.
- [x] Step 2.2: Ensure the extractor compiles/loads the module before reading the cache artifacts.
- [x] Step 2.3: Return a helpful error if the cache artifact is missing (include expected paths).
- **Done when**: Calling `extract_ir()` on a compiled kernel returns non-empty IR text.

## Task 3: Provide 5+ Python kernel → IR test cases
- [x] Step 3.1: Add `code/extraction/test_ir_extractor.py` with ≥5 small kernels (different ops) and a minimal harness.
- [x] Step 3.2: Each test compiles the kernel, calls `extract_ir()`, and asserts IR is non-empty.
- [x] Step 3.3: Run the test file twice from a clean shell; both runs pass.
- **Done when**: 5+ kernels pass the extraction assertions twice.

