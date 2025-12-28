# CUDA Backend Development

## Objective
Adapt the current production code to use the CUDA backend. Since no GPU is available in the agent environment, provide concise code and commands for the user to test on a GPU device.

---

## File Structure

```
jit/
├── instruction_cuda.md      # This file
├── CUDA_STATE.md            # Progress tracker
├── code/                    # Implementation code (adapted for CUDA)
├── cuda_tests/              # GPU-specific validation scripts
└── notes/                   # Technical findings and GPU analysis
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md`.
2. Resume from the documented next action.

### On Session End
1. Update `CUDA_STATE.md` with:
   - Current milestone and task.
   - Exact next action.
   - Key findings.
2. Commit working code.
3. Stop—do not start new tasks.

### CUDA_STATE.md Template
```markdown
# CUDA State
- **Milestone**: M1/M2/M3/M4
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Specific next step]

## Key Findings
- [finding]

## Session Log
- [date]: [summary]
```

---

## Milestones

### M1: Setup & Analysis
**Goal**: Establish a working CPU baseline and understand CUDA requirements.
**Tasks**:
1. Identify and checkout the best "production" base branch (e.g., from `cursor/agent-work-merge-process-...` or `following-instructions-md-12c4`).
2. Verify CPU pipeline works (`pipeline.py`, `ir_extractor.py`).
3. Analyze `warp` documentation for CUDA vs CPU code generation differences.
4. Create `notes/cuda_plan.md` detailing the adaptation strategy.

### M2: Infrastructure Adaptation
**Goal**: Enable CUDA device support in the extraction and synthesis pipeline.
**Tasks**:
1. Modify `ir_extractor.py` to accept `device="cuda"`.
2. Update `pipeline.py` to support GPU targeting.
3. Create `cuda_tests/smoke_test.py` for user to verify basic GPU connectivity.

### M3: Kernel Iteration
**Goal**: Adapt all kernel types for CUDA.
**Tasks**:
Iterate through kernel types (arithmetic, math, loop, conditional, vector, matrix, combined):
1. Adapt generator/extractor for the specific kernel type.
2. Generate a "dry run" (compile without running if possible, or assume success if API matches).
3. Create a specific validation script `cuda_tests/test_[type].py` for the user.

### M4: Final Polish & Packaging
**Goal**: Ensure a smooth user experience for GPU data generation.
**Tasks**:
1. Consolidate tests into a suite.
2. Document usage in `README_CUDA.md`.
3. Verify all code paths (except actual execution) on agent machine.

---

## Anti-Patterns
- ❌ Assuming GPU availability (always guard with `if device == 'cuda'` or similar).
- ❌ Hardcoding device strings without configuration.
- ❌ Over-complicating the "dry run" (trust `warp` types if possible).

## Validation
Since the agent cannot run CUDA:
1. Write scripts that *check* for CUDA availability and fail gracefully or skip.
2. Provide exact commands for the user to run.
3. Rely on `warp`'s static analysis or compilation checks if available without a GPU.
