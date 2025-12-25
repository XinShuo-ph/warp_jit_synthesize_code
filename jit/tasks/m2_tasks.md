# Milestone 2 Tasks

## Task 1: Identify IR artifact(s) and extraction hook
- [x] Step 1.1: Locate installed Warp sources (`import warp; warp.__file__`) and the kernel cache directory
- [x] Step 1.2: Find where codegen emits intermediate representations (search in Warp for "ir", "ptx", "llvm", "adj", "codegen", "module")
- [x] Step 1.3: Decide target IR to extract first (Warp IR vs generated C++/LLVM/PTX) and document the choice
- **Done when**: We can point to an exact Warp function/class that produces the chosen IR and can be called or intercepted programmatically.

## Task 2: Implement `ir_extractor.py`
- [x] Step 2.1: Create `jit/code/extraction/ir_extractor.py` with `extract_ir(kernel, device="cpu") -> str|dict`
- [x] Step 2.2: Ensure it compiles the kernel if needed and returns deterministic IR text/JSON
- [x] Step 2.3: Add minimal error handling for unsupported devices / missing IR
- **Done when**: Calling `extract_ir()` on a simple kernel returns a non-empty IR artifact twice with identical output.

## Task 3: Add 5+ Python→IR test cases
- [x] Step 3.1: Create `jit/code/extraction/test_cases.py` that defines 5+ small kernels (arith, control-flow, atomics, vec/mat, loops)
- [x] Step 3.2: For each kernel, extract IR and save a `(python_source, ir_text)` pair in-memory
- [x] Step 3.3: Verify determinism across two runs (same IR output)
- **Done when**: Running the script produces 5+ non-empty pairs twice with matching IR hashes.

## Task 4: Document IR format (max 30 lines)
- [x] Step 4.1: Write `jit/notes/ir_format.md` (<=30 lines) describing the extracted IR structure and key fields
- **Done when**: File exists, <=30 lines, and matches the artifact returned by `extract_ir()`.

