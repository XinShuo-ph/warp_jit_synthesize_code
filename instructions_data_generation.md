# Data Generation and Reporting

## Objective
Produce 200MB of CPU code and 200MB of CUDA code dataset. Write a simple report in markdown format for the chief scientist to review, introducing JIT, IR, Nvidia Warp, and the current dataset.

---

## File Structure

```
data_generation/
├── instructions.md          # This file
├── STATE.md                 # CRITICAL: Current progress, next action, blockers
├── tasks/                   # Task lists for each milestone
├── code/                    # Implementation code
│   ├── cpu/                 # CPU code generation
│   ├── cuda/                # CUDA code generation
├── data/                    # Generated datasets
│   ├── cpu/
│   ├── cuda/
├── report/                  # Final report
```

---

## State Management Protocol

### On Session Start
1. Read `STATE.md` first
2. Read the current milestone's task file
3. Resume from the documented next action

### On Session End
1. Update `STATE.md` with current status, next actions, and blockers
2. Commit working code
3. Stop—do not start new tasks

---

## Milestones

### M1: CPU Code Production
**Goal**: Produce 200MB of CPU code dataset.
**Process**:
1. Study `cursor/agent-work-merge-` branches.
2. Reproduce each code and evaluate.
3. Pick the best implementation for production.
4. Gradually produce 200 MB of data.
5. Push to remote.

### M2: CUDA Code Production
**Goal**: Produce 200MB of CUDA code dataset.
**Process**:
1. Study `cursor/cuda...` branches.
2. Reproduce each code and evaluate.
3. Pick the best implementation for production.
4. Gradually produce 200 MB of data.
5. Push to remote.

### M3: Report
**Goal**: Write a review report for the chief scientist.
**Content**:
- Introduction to JIT (Just-In-Time compilation)
- Introduction to IR (Intermediate Representation)
- Introduction to Nvidia Warp
- Overview of the current dataset (CPU & CUDA)

---

## Task Breakdown Rules

When starting a milestone, create `tasks/mX_tasks.md`.
Each step should be completable in <5k tokens.
"Done when" must be testable.

---

## Validation Protocol

1. Verify data generation volume (200MB targets).
2. Verify code quality and reproducibility.
3. Ensure report covers all required topics.

---

## Anti-Patterns

- ❌ Starting generation without validating the generator first.
- ❌ Overwriting existing useful data without backup.
- ❌ Leaving the report until the very end without taking notes.
