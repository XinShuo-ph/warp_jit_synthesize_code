# CUDA Backend Development

## Objective
Adapt the current production code to use the CUDA backend. Since no GPU device is available in this environment, the goal is to provide concise, correct code and commands that the user can verify on a GPU-enabled machine.

---

## File Structure
```
cuda/
├── instruction_cuda.md      # This file (read-only reference)
├── STATE.md                 # CRITICAL: Current progress, next action, blockers
├── tasks/                   # Task lists for each milestone
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   ├── m3_tasks.md
│   └── m4_tasks.md
├── code/                    # Implementation code (adapted for CUDA)
│   ├── common/
│   ├── kernels/
│   ├── pipeline/
│   └── tests/
└── notes/                   # Technical findings and verification instructions
```

---

## State Management Protocol

### On Session Start
1. Read `STATE.md` first.
2. Read the current milestone's task file (e.g., `tasks/m1_tasks.md`).
3. Resume from the documented next action.

### On Session End (or ~20k tokens remaining)
1. Update `STATE.md` with:
   - Current milestone and task
   - Exact next action (be specific: file, function, line)
   - Any blockers or failed attempts
2. Commit working code.
3. Stop—do not start new tasks.

### STATE.md Template
```markdown
# Current State
- **Milestone**: M1/M2/M3
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next. Be specific.]

## Blockers (if any)
[What's preventing progress, what was tried]

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Milestones

### M1: Baseline Reproduction (CPU)
**Goal**: Establish a working baseline using the CPU backend from the best existing branch.
**Deliverables**:
- Selected base branch from `cursor/agent-work-merge-process-*`.
- `code/`: Working copy of the production code running on CPU.
- `tasks/m1_reproduction.md`: Log of steps to reproduce.
- Verification that CPU backend works (even if slow).

### M2: CUDA Adaptation (Iterative)
**Goal**: systematic migration of kernels and pipeline to CUDA backend.
**Approach**: Iterate through kernel types one by one.
**Deliverables**:
- `code/kernels/`: Adapted CUDA kernels.
- `code/pipeline/`: Pipeline adapted for CUDA device allocation.
- **Iterations**:
  - Arithmetic & Math kernels
  - Loop & Conditional kernels
  - Vector & Matrix kernels
  - Forward & Backward pass logic
  - Batch generation pipeline

### M3: Validation & Handoff
**Goal**: Create tools for the user to verify functionality on a real GPU.
**Deliverables**:
- `code/tests/`: Test suite designed for GPU execution.
- `instructions_gpu_verification.md`: Concise commands for the user to run.
- `notes/cuda_differences.md`: Summary of changes made for CUDA support.

### M4: Offline CUDA IR Generation
**Goal**: Enable generation of CUDA intermediate representation (CUDA C++) without a physical GPU.
**Approach**: Investigate Warp's codegen internals and bypass runtime driver checks.
**Deliverables**:
- `code/synthesis/pipeline_offline.py`: Pipeline that generates CUDA source without GPU.
- `data/offline_cuda/`: Sample generated CUDA files (at least 3 valid `.cu` contents).
- `notes/offline_generation.md`: Documentation of the technique used.

---

## Task Breakdown Rules

When starting a milestone, create `tasks/mX_tasks.md` with:
```markdown
# Milestone X Tasks

## Task 1: [name]
- [ ] Step 1.1: [specific action]
- [ ] Step 1.2: [specific action]
- **Done when**: [concrete success criterion]
```

---

## Validation Protocol

Since **NO GPU** is available:
1. **Static Analysis**: Verify code correctness by reading standard CUDA/Warp documentation.
2. **CPU Fallback**: Ensure code still runs on CPU if possible, or gracefully errors if CUDA is strictly required.
3. **Mocking/Dry-run**: If possible, verify compilation paths without execution.
4. **User Instructions**: Provide exact commands for the user to run validation on their GPU machine.

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Setup | ~5k | Initialize file structure, pick base branch |
| M1 | ~20k | Reproduce CPU baseline |
| M2 | ~100k | Iterative CUDA adaptation |
| M3 | ~20k | Finalizing validation tools |
| M4 | ~40k | Offline IR generation |

---

## Anti-Patterns

- ❌ Assuming GPU availability (always check/handle gracefully).
- ❌ Committing broken code that prevents CPU execution (unless unavoidable).
- ❌ Over-optimizing for performance without profiling (focus on correctness first).
- ❌ Ignoring existing code patterns in the base branch.
