# CUDA Backend Development

## Objective
Adapt the current production code to use a CUDA backend.
Currently, no GPU device is available for the agent to use. The goal is to provide concise code and commands that allow the user to test on a GPU device.

---

## File Structure (create as needed)

```
cuda/
├── instructions.md          # This file (read-only reference)
├── STATE.md                 # CRITICAL: Current progress, next action, blockers
├── tasks/                   # Task lists for each milestone
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   └── ...
├── code/                    # All implementation code
│   ├── base/                # Reproduced CPU baseline
│   ├── backend/             # CUDA backend infrastructure
│   ├── kernels/             # CUDA kernels (forward/backward)
│   └── validation/          # Validation tools and test suite
└── notes/                   # Technical findings
```

---

## State Management Protocol

### On Session Start
1. Read `cuda/STATE.md` first
2. Read the current milestone's task file (e.g., `cuda/tasks/m1_tasks.md`)
3. Resume from the documented next action

### On Session End (or ~20k tokens remaining)
1. Update `cuda/STATE.md` with:
   - Current milestone and task
   - Exact next action (be specific: file, function, line if applicable)
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states)
3. Stop—do not start new tasks

### STATE.md Template
```markdown
# Current State
- **Milestone**: M1/M2/M3/M4/M5
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

### M1: Baseline Reproduction
**Goal**: Reproduce the current production code using the CPU backend. Study the `cursor/agent-work-merge-process-bc08` branch and use it as the base.
**Deliverables**:
- `cuda/code/base/`: Working CPU implementation reproduced from the base branch.
- `cuda/notes/baseline_analysis.md`: Analysis of kernel types, forward/backward passes, and pipeline structure.

### M2: CUDA Infrastructure Setup
**Goal**: Adapt the code to support a CUDA backend structure.
**Deliverables**:
- `cuda/code/backend/`: CUDA context, device management, and compilation utilities.
- `cuda/code/common/`: Shared types and utilities between CPU and CUDA.
- `cuda/tasks/kernel_inventory.md`: List of all kernels to be ported.

### M3: Kernel Adaptation (Forward Pass)
**Goal**: Port forward pass kernels to CUDA.
**Deliverables**:
- `cuda/code/kernels/forward/`: CUDA implementations for all identified kernels (forward).
- `cuda/validation/test_forward.py`: Tests comparing CPU vs CUDA (forward).

### M4: Kernel Adaptation (Backward Pass)
**Goal**: Port backward pass kernels to CUDA.
**Deliverables**:
- `cuda/code/kernels/backward/`: CUDA implementations for all identified kernels (backward).
- `cuda/validation/test_backward.py`: Tests comparing CPU vs CUDA (backward).

### M5: Pipeline & Validation
**Goal**: Complete batch generation pipeline and test suite.
**Deliverables**:
- `cuda/code/pipeline.py`: Full batch generation pipeline using CUDA.
- `cuda/validation/test_suite.py`: Comprehensive test suite for GPU execution.
- `run_cuda_tests.sh`: Script for the user to run tests on a GPU device.

---

## Task Breakdown Rules

When starting a milestone, create `cuda/tasks/mX_tasks.md` with:
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

---

## Validation Protocol

Before marking any task complete:
1. Run the code/test twice
2. Results must match both times
3. No uncommitted debug code or prints
4. Code runs from clean state (no hidden dependencies)

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Orientation | ~5k | Read STATE.md, task file, understand context |
| Planning | ~10k | Break down next task, explore relevant code |
| Execution | ~150k | Implement, test, iterate |
| Handoff | ~10k | Update STATE.md, clean up, verify state |

If blocked for >20k tokens on same issue:
1. Document the blocker in STATE.md
2. Move to next task or milestone
3. Mark blocker for later resolution

---

## Key Resources
- `cursor/agent-work-merge-process-bc08` branch (Source of Truth for CPU version)
- PyTorch CUDA semantics / Numba / CuPy (depending on implementation choice)

---

## Anti-Patterns (Avoid These)
- ❌ Writing summaries, READMEs, or reports
- ❌ Over-commenting code
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Reading entire large files (use targeted searches)
- ❌ Re-exploring already-documented findings
