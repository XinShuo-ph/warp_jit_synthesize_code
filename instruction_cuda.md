# CUDA Backend Development

## Objective
Adapt the current production code to support a **CUDA** backend in addition to **CPU**.

Constraints:
- This environment may not have a GPU. All changes must be safe to run on CPU-only machines.
- Provide **concise commands** for the user to validate on a real GPU machine later.

---

## Recommended Base Branch

Use `origin/cursor/agent-work-merge-process-bc08` as the base reference for “production-ready CPU pipeline”, unless a later branch is demonstrably better.

Why:
- It contains a complete `code/` pipeline, tests, validation tools, and existing GPU notes.

---

## Target File Structure (create as needed)

```
.
├── instruction_cuda.md        # This file (read-only reference)
├── CUDA_STATE.md              # CRITICAL: CUDA work progress + exact next action
├── code/                      # Production code (CPU + CUDA paths)
│   ├── extraction/            # IR extraction + tests
│   ├── synthesis/             # Generators + pipeline + batch generation
│   └── examples/              # Example kernels / demos
├── data/                      # Small committed samples (≤100 files)
└── docs/                      # Short notes (≤100 lines per doc)
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` (create it if missing using the template below)
2. Resume from **Next Action**

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current phase/task
   - Exact next action (file + function + command)
   - Any blockers (e.g., “no GPU available to validate X”)
2. Leave the code in a runnable state on CPU
3. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA State
- **Phase**: P0/P1/P2/P3
- **Task**: [current task name]
- **Status**: in_progress | blocked | ready_for_next | completed

## Next Action
[Exact next step: file(s) to edit + command(s) to run]

## Blockers (if any)
- [blocker]: [what was tried / what’s needed]

## Session Log
- [date/session]: [1-3 bullets]
```

---

## Phases

### P0: Establish CPU Baseline (choose base + reproduce)
**Goal**: Ensure the production pipeline works on CPU in this branch before any CUDA work.

**Tasks**:
- Pull baseline code from the recommended base branch (or the best alternative after quick comparison).
- Install runtime dependencies.
- Run the pipeline on CPU and run the unit tests.

**Done when**:
- `python -m pytest -q` completes (or the repo has no tests, documented in `CUDA_STATE.md`)
- `python code/synthesis/pipeline.py --count 3 --device cpu` runs successfully

---

### P1: Device Plumbing (CPU default, CUDA optional)
**Goal**: Introduce a consistent `--device {cpu|cuda}` option across pipeline, extraction, and tests.

**Requirements**:
- Default remains **CPU**.
- If `--device cuda` is requested without CUDA availability, fail fast with a clear error message.
- Add a small helper for device resolution (single source of truth).

**Done when**:
- `--device` is accepted and wired end-to-end (pipeline → generator/extractor)
- CPU path still passes tests and sample generation

---

### P2: CUDA Functional Parity (iterate by subsystem)
**Goal**: Achieve the same behavior on CUDA as CPU for:
- Kernel types (all generators)
- Extraction (forward)
- Validation tools
- Test suite

**Iteration rule**:
- One subsystem change per iteration (small PR-sized steps)
- Each iteration adds/updates tests that can be executed on a GPU machine

**Done when**:
- GPU validation commands (below) pass on a CUDA machine for at least:
  - 1 extraction test
  - 1 pipeline run generating multiple kernel types

---

### P3: GPU Test Pack + Handoff
**Goal**: Provide a clear, minimal set of commands the user can run on a GPU machine.

**Done when**:
- `docs/gpu_test_plan.md` exists with copy/paste commands
- Tests include `@pytest.mark.cuda` (or equivalent) so GPU checks are easy to select

---

## Validation Protocol

### CPU (must pass here)
Run locally:
```bash
python -m pytest -q
python code/synthesis/pipeline.py --count 3 --device cpu
```

### GPU (user runs later)
On a CUDA machine:
```bash
python -m pip install -U warp-lang pytest
python -m pytest -q -m cuda
python code/synthesis/pipeline.py --count 10 --device cuda
```

---

## Anti-Patterns (Avoid)
- ❌ Making CUDA the default device
- ❌ Adding “silent fallback to CPU” when `--device cuda` is explicitly requested
- ❌ Large dataset generation committed to git (keep ≤100 samples)
- ❌ Broad refactors unrelated to CUDA enablement
