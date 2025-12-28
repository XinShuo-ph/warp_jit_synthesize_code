# CUDA Backend Development

## Objective
Adapt the current CPU-based Warp JIT code synthesis pipeline to generate CUDA backend code. Since no GPU device is available in this environment, produce validated code and test commands for manual GPU testing.

---

## File Structure (create as needed)

```
cuda/
├── instructions_cuda.md        # This file (read-only reference)
├── CUDA_STATE.md               # CRITICAL: Current progress, next action, blockers
├── tasks/                      # Task lists for each milestone
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   └── ...
├── code/                       # CUDA-adapted implementation
│   ├── examples/               # CUDA kernel examples
│   ├── extraction/             # CUDA IR extraction utilities
│   └── synthesis/              # CUDA data synthesis pipeline
├── data/                       # Generated CUDA training data samples
├── tests/                      # Test suite for GPU validation
└── notes/                      # Technical findings (keep minimal)
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` first
2. Read the current milestone's task file (e.g., `tasks/m1_tasks.md`)
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
- **Milestone**: M1/M2/M3
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

## Milestones

### M1: Baseline Setup & Analysis
**Goal**: Reproduce CPU pipeline and analyze CUDA requirements
**Deliverables**:
- Working copy of best CPU branch (from `branch_progresses.md`)
- `notes/cuda_analysis.md`: CPU vs CUDA IR differences, CUDA code generation mechanism
- `notes/kernel_inventory.md`: List all kernel types used in CPU pipeline
- Test run of CPU pipeline to establish baseline

### M2: CUDA IR Extraction
**Goal**: Adapt IR extractor to generate CUDA code
**Deliverables**:
- `code/extraction/cuda_ir_extractor.py`: Modified IR extractor with `device="cuda"` support
- `code/extraction/test_cuda_extraction.py`: Unit tests for CUDA extraction
- 10+ test cases showing Python kernel → CUDA IR pairs
- `notes/cuda_ir_format.md`: CUDA IR structure documentation (max 30 lines)

### M3: CUDA Synthesis Pipeline
**Goal**: Adapt synthesis pipeline for CUDA backend
**Deliverables**:
- `code/synthesis/cuda_generator.py`: Generator for CUDA-specific kernel patterns
- `code/synthesis/cuda_pipeline.py`: End-to-end CUDA pipeline
- `code/synthesis/cuda_batch_generator.py`: Batch generation for CUDA
- Pipeline generates 100+ valid Python→CUDA IR pairs (on CPU, for later GPU testing)

### M4: GPU Test Suite
**Goal**: Create comprehensive test suite for GPU validation
**Deliverables**:
- `tests/test_cuda_kernels.py`: Test all kernel types on GPU
- `tests/test_cuda_pipeline.py`: End-to-end pipeline tests
- `tests/run_gpu_tests.sh`: Bash script with clear instructions for GPU testing
- `tests/README.md`: How to run tests on GPU, expected outputs

### M5: Validation & Documentation
**Goal**: Final validation and user documentation
**Deliverables**:
- `README.md`: Quick start guide for CUDA pipeline
- `CUDA_TESTING_GUIDE.md`: Detailed instructions for GPU testing
- `notes/cuda_vs_cpu.md`: Performance expectations, differences
- Sample CUDA IR data (50-100 pairs) in `data/samples/`

---

## Task Breakdown Rules

When starting a milestone, create `tasks/mX_tasks.md` with:
```markdown
# Milestone X Tasks

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
- Focus on ONE kernel type at a time when adapting to CUDA

---

## Validation Protocol

Before marking any task complete:
1. Run the code/test twice (on CPU for IR generation)
2. Results must match both times
3. No uncommitted debug code or prints
4. Code runs from clean state (no hidden dependencies)
5. Generated CUDA IR looks syntactically valid (manual inspection)

---

## CUDA-Specific Considerations

### Device Parameter
All Warp compilation functions support `device` parameter:
- `device="cpu"` → generates `.cpp` files
- `device="cuda"` → generates `.cu` files

### Kernel Patterns to Cover
Based on branch analysis, adapt these kernel types:
1. Arithmetic operations (add, mul, etc.)
2. Math functions (sin, cos, exp, etc.)
3. Loop patterns (for loops, reductions)
4. Conditional patterns (if/else)
5. Vector operations (wp.vec3, wp.vec2)
6. Matrix operations (wp.mat33, wp.mat22)
7. Combined patterns (multi-operation)
8. Atomic operations (CUDA-specific)
9. Shared memory patterns (CUDA-specific)
10. Thread synchronization (CUDA-specific)

### CUDA-Specific Patterns (New)
Add these patterns not in CPU version:
- Thread indexing (`wp.tid()`)
- Block/grid patterns
- Shared memory usage
- Warp-level primitives
- Atomic operations

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Orientation | ~5k | Read CUDA_STATE.md, task file, branch selection |
| Planning | ~10k | Analyze CPU code, plan CUDA adaptations |
| Execution | ~150k | Implement, test, iterate (one kernel type at a time) |
| Handoff | ~10k | Update CUDA_STATE.md, clean up, verify state |

If blocked for >20k tokens on same issue:
1. Document the blocker in CUDA_STATE.md
2. Move to next kernel type or milestone
3. Mark blocker for later resolution

---

## Key Resources

- Warp documentation: https://nvidia.github.io/warp/
- Warp CUDA backend: `warp/native/cuda/`
- CPU branches: See `branch_progresses.md` (use branch 12c4 as primary base)
- Warp compilation: `warp/context.py`, `warp/codegen.py`

---

## Anti-Patterns (Avoid These)

- ❌ Trying to run CUDA code on CPU-only environment
- ❌ Writing summaries, READMEs, or reports before code works
- ❌ Adapting all kernel types at once (do ONE at a time)
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Generating large datasets during development
- ❌ Re-implementing CPU pipeline from scratch (adapt existing best branch)

---

## Success Criteria

Project is complete when:
1. CUDA IR extraction works for all kernel types (validated on CPU)
2. Pipeline generates valid Python→CUDA IR pairs
3. Comprehensive test suite ready for GPU execution
4. Clear documentation for manual GPU testing
5. Sample CUDA IR data committed (50-100 pairs)
6. All code changes documented in CUDA_STATE.md
