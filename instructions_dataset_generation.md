# Dataset Generation and Reporting Instructions

## Objective
Produce a 200MB CPU code dataset and a 200MB CUDA code dataset. Write a simple report in markdown format for the chief scientist to review, introducing JIT, IR, NVIDIA Warp, and the current dataset.

---

## File Structure (create as needed)

```
dataset_generation/
├── instructions.md          # This file (read-only reference)
├── STATE.md                 # CRITICAL: Current progress, next action, blockers
├── tasks/                   # Task lists for each milestone
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   └── m3_tasks.md
├── code/                    # Implementation code selected/adapted from branches
│   ├── cpu/                 # CPU generation code
│   └── cuda/                # CUDA generation code
├── data/                    # Generated datasets (ensure .gitignore handles large files)
│   ├── cpu/
│   └── cuda/
└── report/                  # Final report
    └── report.md
```

---

## State Management Protocol

### On Session Start
1. Read `STATE.md` first
2. Read the current milestone's task file (e.g., `tasks/m1_tasks.md`)
3. Resume from the documented next action

### On Session End
1. Update `STATE.md` with:
   - Current milestone and task
   - Exact next action
   - Blockers/Issues
2. Commit working code
3. Push data to remote (as requested in phases)

---

## Milestones

### M1: CPU Code Production
**Goal**: Produce 200MB of CPU code dataset.
**Workflow**:
1. Study `cursor/agent-work-merge-*` branches.
2. Reproduce code from these branches.
3. Pick the best implementation for production.
4. Gradually produce 200 MB of data.
5. Push to remote.

### M2: CUDA Code Production
**Goal**: Produce 200MB of CUDA code dataset.
**Workflow**:
1. Study `cursor/cuda...` branches.
2. Reproduce code from these branches.
3. Pick the best implementation for production.
4. Gradually produce 200 MB of data.
5. Push to remote.

### M3: Report Generation
**Goal**: Create a summary report.
**Content**:
- Introduction to JIT, IR, NVIDIA Warp.
- Overview of the current dataset (CPU and CUDA).
- Markdown format for Chief Scientist review.

---

## Task Breakdown Rules

When starting a milestone, create `tasks/mX_tasks.md` with specific, testable steps.
- Each step should be actionable.
- Define "Done when" criteria.
- Mark completed steps.

---

## Validation Protocol

- Ensure generated data meets size requirements (200MB each).
- Verify code reproducibility before large-scale generation.
