# CUDA Backend Development

## Objective
Adapt the current CPU-based production code to use CUDA backend for GPU-accelerated training data generation.

---

## File Structure (create as needed)

```
jit/
├── instructions_cuda.md     # This file (read-only reference)
├── CUDA_STATE.md            # CRITICAL: Current progress, next action, blockers
├── tasks/
│   ├── cuda_m1_tasks.md     # Milestone 1 tasks
│   ├── cuda_m2_tasks.md     # Milestone 2 tasks
│   └── ...
├── code/
│   ├── extraction/          # IR extraction (CPU + CUDA)
│   ├── synthesis/           # Data generation pipeline
│   ├── examples/            # Example kernels
│   └── cuda/                # CUDA-specific utilities
├── data/
│   ├── cpu_samples/         # CPU IR samples (reference)
│   └── cuda_samples/        # CUDA IR samples (new)
├── tests/                   # Test suite for GPU validation
└── notes/
    └── cuda_notes.md        # Technical findings
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` first
2. Read the current milestone's task file (e.g., `tasks/cuda_m1_tasks.md`)
3. Resume from the documented next action

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current milestone and task
   - Exact next action (be specific: file, function, line if applicable)
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states)
3. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA Development State
- **Milestone**: CM1/CM2/CM3/CM4
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next
- **Base Branch**: [which production branch was used as base]

## Next Action
[Exactly what to do next. Be specific enough that a new agent can execute immediately.]

## Blockers (if any)
[What's preventing progress, what was tried]

## Current Kernel Progress
| Kernel Type | Forward | Backward | Status |
|-------------|---------|----------|--------|
| arithmetic  | [ ]     | [ ]      | pending |
| math        | [ ]     | [ ]      | pending |
| loop        | [ ]     | [ ]      | pending |
| conditional | [ ]     | [ ]      | pending |
| vector      | [ ]     | [ ]      | pending |
| matrix      | [ ]     | [ ]      | pending |
| combined    | [ ]     | [ ]      | pending |

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Milestones

### CM1: Base Code Selection & Reproduction
**Goal**: Identify best production branch, reproduce CPU pipeline
**Deliverables**:
- Analysis of `cursor/agent-work-merge-process-*` branches
- Selected base branch documented in `CUDA_STATE.md`
- CPU pipeline working in current branch
- `notes/cpu_baseline.md`: Documented CPU IR output format (max 50 lines)

**Done when**: 
- CPU pipeline generates 10+ sample pairs successfully
- Base code structure understood and documented

### CM2: CUDA IR Extraction
**Goal**: Adapt IR extraction to capture CUDA/PTX code
**Deliverables**:
- `code/extraction/ir_extractor.py`: Updated with `device="cuda"` parameter
- `code/extraction/cuda_utils.py`: CUDA-specific extraction utilities
- Comparison script: CPU IR vs CUDA IR differences
- 10+ test pairs showing Python→CUDA IR

**Done when**:
- CUDA IR extraction works for all 7 kernel types
- Side-by-side comparison shows clear CPU vs GPU differences
- `notes/cuda_ir_format.md`: Documents CUDA IR structure

### CM3: Iterative Kernel Adaptation
**Goal**: Adapt each kernel type for CUDA, both forward and backward
**Deliverables**:
- Updated kernel generators in `code/synthesis/generator.py`
- CUDA-specific patterns for each kernel type
- Forward and backward pass implementations
- Test validation for each kernel type

**Subtasks** (iterate one at a time):
1. Arithmetic kernels (forward + backward)
2. Math kernels (forward + backward)
3. Loop kernels (forward + backward)
4. Conditional kernels (forward + backward)
5. Vector kernels (forward + backward)
6. Matrix kernels (forward + backward)
7. Combined kernels (forward + backward)

**Done when**: 
- All 14 subtasks (7 kernel types × 2 passes) complete
- Each produces valid CUDA IR
- Tests pass (can be run by user on GPU)

### CM4: Batch Generation & Validation
**Goal**: Scale up CUDA data generation, create test suite
**Deliverables**:
- `code/synthesis/cuda_pipeline.py`: CUDA batch generation pipeline
- `tests/test_cuda_kernels.py`: Comprehensive test suite for GPU validation
- `tests/run_on_gpu.sh`: Script for user to run tests on actual GPU
- 100+ CUDA sample pairs in `data/cuda_samples/`
- `notes/cuda_validation.md`: Testing instructions and expected outputs

**Done when**:
- Pipeline generates CUDA IR without errors
- Test suite is complete and documented
- Clear instructions provided for GPU testing
- Sample data committed (≤100 pairs for git)

### CM5: CUDA Code Production Pipeline
**Goal**: Generate standalone CUDA C++ code that can be compiled without Warp
**Deliverables**:
- `code/cuda_production/code_generator.py`: Converts Python kernels to pure CUDA C++
- `code/cuda_production/cuda_template.py`: CUDA code templates and utilities
- `code/cuda_production/compile_cuda.py`: Compilation validation (nvcc wrapper)
- Generated `.cu` files that compile with nvcc
- PTX assembly output for analysis
- 50+ Python→CUDA code pairs in `data/cuda_production/`
- `notes/cuda_production.md`: CUDA code generation details

**Subtasks**:
1. Analyze Warp CUDA IR structure and extract patterns
2. Create CUDA code templates (kernel signature, memory ops, thread indexing)
3. Build Python→CUDA translator for each kernel type
4. Generate standalone .cu files with host code
5. Create compilation pipeline (code → .cu → PTX)
6. Validate generated code structure
7. Document CUDA code generation process

**Done when**:
- All 10 kernel types generate compilable CUDA code
- Generated .cu files have proper CUDA syntax
- PTX assembly can be produced (without GPU)
- 50+ production-ready code samples
- Documentation explains code generation process

---

## Task Breakdown Rules

When starting a milestone, create `tasks/cuda_mX_tasks.md` with:
```markdown
# CUDA Milestone X Tasks

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
- Focus on ONE kernel type at a time in CM3

---

## Validation Protocol

Before marking any task complete:
1. Run the code/test twice (if no GPU: validate code structure)
2. Results must be consistent
3. No uncommitted debug code or prints
4. Code is documented for user to test on GPU

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Orientation | ~5k | Read CUDA_STATE.md, task file, understand context |
| Planning | ~10k | Analyze base branches, select best one |
| Execution | ~150k | Implement, adapt kernels iteratively |
| Handoff | ~10k | Update CUDA_STATE.md, clean up, verify state |

If blocked for >20k tokens on same issue:
1. Document the blocker in CUDA_STATE.md
2. Move to next kernel type or milestone
3. Mark blocker for later resolution

---

## Key Resources

### Production Branches to Evaluate
- `cursor/agent-work-merge-process-0038`
- `cursor/agent-work-merge-process-0499`
- `cursor/agent-work-merge-process-4dce`
- `cursor/agent-work-merge-process-6964`
- `cursor/agent-work-merge-process-96fd`
- `cursor/agent-work-merge-process-ad19`
- `cursor/agent-work-merge-process-bc08`

### Warp Resources
- Warp repo: https://github.com/NVIDIA/warp.git
- Key files to study:
  - `warp/codegen.py` (IR generation)
  - `warp/context.py` (kernel compilation, device selection)
  - `warp/types.py` (type system)
- Device parameter: `wp.init()` and `wp.launch(device="cuda")`

### CUDA Specific
- PTX (Parallel Thread Execution) assembly format
- CUDA C++ code generation vs CPU C++ code
- Thread/block organization patterns

---

## Anti-Patterns (Avoid These)

- ❌ Testing on GPU without clear instructions (no GPU available to agent)
- ❌ Working on all kernel types simultaneously
- ❌ Starting new kernel before previous is validated
- ❌ Writing lengthy analysis documents
- ❌ Leaving code in broken state at session end
- ❌ Generating large datasets during development (≤100 samples)
- ❌ Re-implementing CPU code from scratch (adapt existing)

---

## Success Criteria

CUDA backend is complete when:
1. All 7 kernel types generate CUDA IR (forward + backward)
2. CUDA pipeline runs without errors
3. Test suite is documented with clear GPU execution instructions
4. Sample CUDA IR data (50-100 pairs) committed
5. Side-by-side comparison shows CPU vs CUDA differences
6. User can run tests on GPU device following provided instructions

---

## Notes

- **No GPU available to agent**: Focus on code structure, provide test scripts
- **Concise code**: Prioritize clarity and testability over optimization
- **Iterative approach**: ONE kernel type at a time, ONE task at a time
- **User validation**: All tests designed for user to run on actual GPU hardware
