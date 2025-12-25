# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: Implement IR Extractor
- [x] Step 1.1: Create `code/extraction/ir_extractor.py`.
- [x] Step 1.2: Implement `get_kernel_ir(kernel, device="cpu") -> str` function.
    - Ensure it handles module building (`ModuleBuilder`) if not already built.
    - Use `codegen_kernel` to get the source.
- [x] Step 1.3: Verify it works on the `simple_kernel` from M1.
- **Done when**: `ir_extractor.py` exists and can extract source code from a fresh kernel without errors.

## Task 2: Create Test Cases
- [x] Step 2.1: Create `code/extraction/test_extractor.py`.
- [x] Step 2.2: Define 5 diverse kernels:
    1.  `arithmetic_kernel`: Basic math.
    2.  `loop_kernel`: For/while loops.
    3.  `conditional_kernel`: If/else logic.
    4.  `array_kernel`: Array reads/writes.
    5.  `builtin_kernel`: Using `wp.sin`, `wp.min`, etc.
- [x] Step 2.3: Run tests and ensure non-empty IR is returned for all.
- **Done when**: `pytest code/extraction/test_extractor.py` passes (or running it as script passes).

## Task 3: Document IR Format
- [x] Step 3.1: Analyze the output of the 5 kernels.
- [x] Step 3.2: Create `notes/ir_format.md` describing:
    - Argument struct structure.
    - Forward function signature and body.
    - Backward function signature (if enabled).
    - Variable naming conventions in IR.
- **Done when**: `notes/ir_format.md` is committed.
