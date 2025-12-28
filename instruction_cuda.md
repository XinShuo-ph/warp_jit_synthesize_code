# CUDA Backend Development

## Objective
Adapt the current production code to use the CUDA backend. Since no GPU device is available in this environment, provide concise code and commands that allow for external testing on a GPU device.

---

## File Structure

```
jit/
├── instruction_cuda.md      # This file (read-only reference)
├── CUDA_STATE.md            # CRITICAL: Current progress, next action, blockers
├── tasks/                   # Task lists for each milestone
│   ├── cuda_m1_tasks.md
│   ├── cuda_m2_tasks.md
│   └── ...
├── code/                    # Implementation code (adapted for CUDA)
│   ├── examples/            # Reproduced/new examples
│   ├── extraction/          # IR extraction utilities
│   └── synthesis/           # Data synthesis pipeline
├── data/                    # Generated training data samples
└── notes/                   # Technical findings and GPU analysis
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` first.
2. Read the current milestone's task file (e.g., `tasks/cuda_m1_tasks.md`).
3. Resume from the documented next action.

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current milestone and task
   - Exact next action (be specific: file, function, line if applicable)
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states).
3. Stop—do not start new tasks.

### CUDA_STATE.md Template
```markdown
# Current State
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

### M1: Reproduce & Analyze CPU Baseline
**Goal**: Reproduce the current production code using the CPU backend and select the best base branch.
**Deliverables**:
- working `code/` directory populated from the best existing branch (e.g., `12c4`, `9177`).
- `notes/cpu_baseline.md`: Confirmation that CPU generation works and analysis of where CUDA adaptation is needed.
- `tasks/cuda_m1_tasks.md`: Task list for this milestone.

### M2: CUDA Adaptation (Iterative)
**Goal**: Adapt the code to use the CUDA backend.
**Scope**:
- All kernel types (arithmetic, math, loop, conditional, vector, matrix, combined)
- Both forward and backward pass (if applicable)
- Batch generation pipeline
**Deliverables**:
- `code/extraction/ir_extractor_cuda.py`: Extractor capable of handling CUDA kernels.
- `code/synthesis/generator_cuda.py`: Generator producing CUDA-compatible Python code.
- `notes/cuda_adaptation.md`: Documentation of changes required for CUDA.

### M3: Validation & Test Suite
**Goal**: Create tools to validate the CUDA implementation on an actual GPU.
**Deliverables**:
- `tests/test_cuda_execution.py`: Test suite to run on a GPU machine.
- `README_GPU.md`: Instructions for running the validation suite on a GPU.
- Verified generation of `.cu` or CUDA IR code (even if execution fails locally).

---

## Task Breakdown Rules

When starting a milestone, create `tasks/cuda_mX_tasks.md` with:
```markdown
# Milestone X Tasks

## Task 1: [name]
- [ ] Step 1.1: [specific action]
- [ ] Step 1.2: [specific action]
- **Done when**: [concrete success criterion]

## Task 2: [name]
...
```

---

## Validation Protocol

Since local GPU execution is not possible:
1. **Static Analysis**: Verify that generated code contains expected CUDA constructs (e.g., `wp.launch(device="cuda")`).
2. **Mock Testing**: If possible, mock `warp`'s CUDA context to verify logic flow without execution.
3. **Artifact Inspection**: Check generated IR/code for CUDA-specific headers or syntax.
4. **Reproducibility**: Ensure generation scripts run deterministically.

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Orientation | ~5k | Read STATE.md, task file, understand context |
| Planning | ~10k | Break down next task, explore relevant code |
| Execution | ~150k | Implement, test, iterate |
| Handoff | ~10k | Update STATE.md, clean up, verify state |

---

## Anti-Patterns (Avoid These)

- ❌ Writing summaries, READMEs, or reports (unless requested)
- ❌ Over-commenting code
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Assuming GPU is available (always check/handle `device="cpu"` fallback for local runs)
