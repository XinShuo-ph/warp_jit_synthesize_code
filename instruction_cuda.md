# CUDA Backend Adaptation

## Objective
Adapt the current CPU-based Python→IR synthesis pipeline to generate CUDA backend code (.cu files instead of .cpp). Since no GPU is available in this environment, produce code and test scripts that can be validated on a GPU device later.

---

## File Structure (create as needed)

```
cuda/
├── instruction_cuda.md     # This file (read-only reference)
├── CUDA_STATE.md           # CRITICAL: Current progress, next action, blockers
├── tasks/                  # Task lists for each phase
│   ├── p1_tasks.md
│   ├── p2_tasks.md
│   └── ...
├── code/                   # CUDA-adapted implementation
│   ├── extraction/         # IR extractor (CUDA-enabled)
│   ├── synthesis/          # Kernel generator and pipeline
│   └── examples/           # Example CUDA kernels
├── data/                   # Generated CUDA training data samples
├── tests/                  # Test suite for GPU validation
│   ├── test_extraction.py
│   ├── test_kernels.py
│   └── run_gpu_tests.py   # Script to run on actual GPU
└── notes/                  # Technical findings (keep minimal)
    └── cuda_vs_cpu.md      # Key differences documented
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` first
2. Read the current phase's task file (e.g., `tasks/p1_tasks.md`)
3. Resume from the documented next action

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current phase and task
   - Exact next action (be specific: file, function, line if applicable)
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states)
3. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA Backend State
- **Phase**: P1/P2/P3/P4/P5
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next. Be specific enough that a new agent can execute immediately.]

## Blockers (if any)
[What's preventing progress, what was tried]

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Phase Overview

| Phase | Goal | Deliverables |
|-------|------|--------------|
| P1 | Setup base code | Working CPU pipeline from best branch |
| P2 | CUDA extraction | IR extractor with `device="cuda"` |
| P3 | Kernel adaptation | All kernel types generating CUDA code |
| P4 | Batch pipeline | Full CUDA data generation pipeline |
| P5 | GPU test suite | Validation tools for actual GPU testing |

---

## Phases

### P1: Setup & Reproduce CPU Pipeline
**Goal**: Get a working CPU pipeline as the starting point

**Tasks**:
1. Identify best source branch from merge process branches:
   ```bash
   git ls-tree -r --name-only origin/cursor/agent-work-merge-process-0038 | head -30
   ```
2. Copy code structure to `cuda/` directory:
   ```bash
   mkdir -p cuda/code/extraction cuda/code/synthesis cuda/code/examples
   mkdir -p cuda/data cuda/tests cuda/notes cuda/tasks
   ```
3. Pull key files from best branch:
   - `ir_extractor.py`
   - `generator.py`
   - `pipeline.py`
   - `batch_generator.py`
4. Verify CPU pipeline works:
   ```bash
   pip install warp-lang
   python cuda/code/synthesis/pipeline.py --count 3
   ```
5. Document baseline in `notes/baseline.md`

**Done when**: CPU pipeline generates at least 3 valid Python→C++ pairs

---

### P2: CUDA IR Extraction
**Goal**: Modify IR extractor to generate CUDA code

**Tasks**:
1. Study CPU vs CUDA code generation in warp:
   - `device="cpu"` → generates `.cpp` code
   - `device="cuda"` → generates `.cu` code
2. Modify `ir_extractor.py`:
   - Add `device` parameter (default "cuda")
   - Handle CUDA-specific code generation
   - Extract both forward and backward CUDA kernels
3. Create `notes/cuda_vs_cpu.md` documenting key differences
4. Test extraction without GPU (code generation works, execution won't):
   ```python
   # Should generate .cu code even without GPU
   result = extract_ir(kernel, device="cuda")
   assert "cuda" in result["metadata"]["device"]
   ```

**Done when**: `extract_ir(kernel, device="cuda")` returns valid CUDA code structure

---

### P3: Kernel Type Adaptation
**Goal**: Ensure all kernel types work with CUDA backend

**Kernel Types** (from generator.py):
1. `arithmetic` - basic ops (+, -, *, /)
2. `vector` - vec2/vec3/vec4 operations
3. `matrix` - mat22/mat33/mat44 operations  
4. `control_flow` - if/else, for loops
5. `math` - unary functions (sin, cos, exp, etc.)
6. `atomic` - atomic_add, atomic_min, atomic_max

**Per-Kernel Tasks**:
For each kernel type:
1. Generate kernel with `generate_kernel(category)`
2. Extract CUDA IR with `extract_ir(kernel, device="cuda")`
3. Verify forward code is valid CUDA
4. Verify backward code is valid CUDA (if applicable)
5. Add test case to `tests/test_kernels.py`

**Done when**: All 6 kernel types produce valid CUDA forward/backward code

---

### P4: Batch Generation Pipeline
**Goal**: Full pipeline for CUDA training data generation

**Tasks**:
1. Update `pipeline.py`:
   - Add `--device cuda` flag
   - Default to CUDA output
   - Update output format for CUDA metadata
2. Update `batch_generator.py`:
   - Support CUDA device parameter
   - Generate CUDA code in parallel
3. Create `cuda/data/samples/` with 50-100 CUDA pairs
4. Update data format:
   ```json
   {
     "kernel_name": "arith_abc123",
     "python_source": "@wp.kernel\ndef ...",
     "cuda_forward": "void arith_abc123_cuda_kernel_forward(...) {...}",
     "cuda_backward": "void arith_abc123_cuda_kernel_backward(...) {...}",
     "device": "cuda",
     "category": "arithmetic"
   }
   ```

**Done when**: Pipeline generates 50+ valid CUDA Python→IR pairs

---

### P5: GPU Test Suite
**Goal**: Create tests that can run on actual GPU device

**Deliverables**:

1. `tests/test_extraction.py`:
   ```python
   # Can run without GPU - tests code generation only
   def test_cuda_extraction():
       ...
   ```

2. `tests/test_kernels.py`:
   ```python
   # Tests each kernel type generates valid CUDA
   def test_arithmetic_cuda():
       ...
   def test_vector_cuda():
       ...
   # etc.
   ```

3. `tests/run_gpu_tests.py`:
   ```python
   # REQUIRES GPU - for user to run later
   # Tests actual CUDA kernel execution
   def test_cuda_execution():
       wp.init()  # Will fail without GPU
       ...
   ```

4. `GPU_TESTING.md`:
   ```markdown
   # GPU Testing Guide
   
   ## Requirements
   - NVIDIA GPU with CUDA support
   - CUDA toolkit installed
   - warp-lang package
   
   ## Quick Test
   ```bash
   pip install warp-lang
   python tests/run_gpu_tests.py
   ```
   
   ## Expected Output
   ...
   ```

**Done when**: Test suite is ready for GPU validation

---

## Task Breakdown Rules

When starting a phase, create `tasks/pX_tasks.md` with:
```markdown
# Phase X Tasks

## Task 1: [name]
- [ ] Step 1.1: [specific action]
- [ ] Step 1.2: [specific action]
- **Done when**: [concrete success criterion]

## Task 2: [name]
...
```

Rules:
- Each step should be completable in <5k tokens
- "Done when" must be testable (not subjective)
- Mark completed steps with [x]

---

## Validation Protocol

Before marking any task complete:
1. Run the code/test (CPU-side generation works without GPU)
2. Verify CUDA code is generated (not executed)
3. No uncommitted debug code or prints
4. Code runs from clean state (no hidden dependencies)

**Note**: Actual GPU execution will be tested by user later.

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| P1 | ~30k | Setup, copy code, verify baseline |
| P2 | ~40k | CUDA extraction implementation |
| P3 | ~60k | All 6 kernel types (~10k each) |
| P4 | ~30k | Pipeline updates, batch generation |
| P5 | ~30k | Test suite creation |

If blocked for >20k tokens on same issue:
1. Document the blocker in CUDA_STATE.md
2. Move to next task or phase
3. Mark blocker for later resolution

---

## Key Resources

- Warp documentation: https://nvidia.github.io/warp/
- CUDA code generation: `warp/_src/codegen.py`
- Device handling: `warp/_src/context.py`
- Key insight: `builder.codegen("cuda")` generates `.cu` code

---

## Anti-Patterns (Avoid These)

- ❌ Trying to execute CUDA kernels without GPU
- ❌ Writing summaries, READMEs, or reports (except GPU_TESTING.md)
- ❌ Over-commenting code
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Re-exploring already-documented findings
- ❌ Major refactoring of working code

---

## Success Criteria

Project is complete when:
1. ✅ CUDA IR extractor works for all kernel types
2. ✅ Forward and backward passes generate CUDA code
3. ✅ Batch pipeline generates 50+ CUDA pairs
4. ✅ Test suite ready for GPU validation
5. ✅ `GPU_TESTING.md` has clear instructions for user
