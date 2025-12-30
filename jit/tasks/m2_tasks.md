# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: Build IR extractor module
- [x] Step 1.1: Create `ir_extractor.py` with JAXPR extraction
- [x] Step 1.2: Add XLA HLO extraction
- [x] Step 1.3: Add source code extraction (inspect module)
- [x] Step 1.4: Create unified extraction function returning all IR types
- **Done when**: `extract_all()` returns dict with source, jaxpr, hlo

## Task 2: Create test cases
- [x] Step 2.1: Create 5+ diverse kernel types (arithmetic, trig, reduction, etc.)
- [x] Step 2.2: Test extraction on all kernels
- [x] Step 2.3: Verify output format is consistent
- **Done when**: All 5+ test cases produce valid Python→IR pairs

## Task 3: Document IR format
- [x] Step 3.1: Create `notes/ir_format.md` describing JAXPR structure
- **Done when**: Notes file exists with <30 lines
