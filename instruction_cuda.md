# CUDA Backend Development (Warp JIT Synthesis)

## Objective
Adapt the current **CPU-only** Warp JIT code-synthesis pipeline to support a **CUDA backend** end-to-end:
- Extract generated code/IR for **`device=cpu` and `device=cuda`**
- Ensure the synthesis pipeline (all kernel types) runs on both devices
- Add a test suite that runs in CPU-only environments (this agent) and enables CUDA tests when a GPU is available (your machine)

**Constraint**: This environment has no GPU. All CUDA paths must be implemented with graceful runtime detection and test skipping, plus clear commands for you to run on a GPU box.

---

## File Structure (create as needed)

```
jit/
├── instructions_cuda.md       # This file (copy/move here if preferred; root file is source-of-truth)
├── CUDA_STATE.md              # CRITICAL: Current progress, next action, blockers
├── code/
│   ├── extraction/            # IR + generated code extraction utilities
│   ├── synthesis/             # kernel generators + pipeline + batch generator
│   └── examples/              # runnable examples / smoke tests
├── tests/                     # pytest suite (CPU tests always; CUDA tests skip if unavailable)
└── notes/
    └── cuda_backend.md        # minimal notes: API quirks, CUDA-vs-CPU differences
```

---

## State Management Protocol

### On Session Start
1. Read `jit/CUDA_STATE.md` (create if missing)
2. Resume from the documented **Next Action**

### On Session End (or ~20k tokens remaining)
1. Update `jit/CUDA_STATE.md` with:
   - Current milestone + task
   - Exact next action (file/function/command)
   - Blockers / failed attempts
   - Any CUDA-specific findings that affect next steps
2. Leave the repo in a runnable state (CPU path must work)

### CUDA_STATE.md Template
```markdown
# CUDA State
- **Milestone**: M0/M1/M2/M3/M4/M5
- **Task**: [task id + short name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next; include commands/files.]

## Blockers (if any)
[What’s blocking, what was tried.]

## Session Log
- [yyyy-mm-dd]: [1-3 bullets of progress]
```

---

## Milestones

### M0: Select & Import Production Baseline
**Goal**: Choose the best existing “production” branch and bring its `jit/` code into this branch.

**Default base**: `origin/cursor/following-instructions-md-12c4` (most complete pipeline + tests).

**Deliverables**:
- `jit/` exists locally with code/tests/docs
- No huge datasets checked into this branch (keep git light)

### M1: CPU Baseline Reproduction (must pass here)
**Goal**: Run the existing pipeline on CPU and ensure tests pass in this CPU-only environment.

**Deliverables**:
- `python -m pytest` passes (or a documented minimal subset)
- `python jit/code/synthesis/pipeline.py --count 3` succeeds on CPU

### M2: Device Abstraction + CUDA Extraction
**Goal**: Add `device` plumbing (cpu/cuda) and implement CUDA code extraction.

**Deliverables**:
- `ir_extractor.py` can compile/extract for `device="cpu"` and `device="cuda"`
- Output metadata includes device + generated code artifact type (e.g., `.cpp` vs `.cu`)

### M3: CUDA-Enabled Synthesis Pipeline (all kernel types)
**Goal**: Make generator + pipeline + batch generator work on CUDA (when available).

**Deliverables**:
- CLI supports `--device cpu|cuda`
- All kernel types compile on selected device (CUDA tests can skip here)

### M4: Validation + Test Suite
**Goal**: CPU tests always run; CUDA tests run when a GPU exists.

**Deliverables**:
- `pytest` has CUDA-marked tests with runtime skip if CUDA unavailable
- A single command you can run on GPU to validate CUDA end-to-end

### M5: Documentation & GPU Runbook
**Goal**: Make it easy to reproduce results on a GPU box.

**Deliverables**:
- `jit/README.md` updated with CPU/CUDA usage
- `jit/notes/cuda_backend.md` (≤80 lines) with pitfalls and how to debug failures

---

## Validation Protocol

Before marking any milestone complete:
1. Run CPU pipeline + CPU tests at least once from a clean state
2. Ensure CUDA paths are guarded by runtime detection (no hard failure on CPU-only boxes)
3. No debug prints / temporary hacks left behind

---

## GPU Test Commands (for you)

On a GPU machine (after `pip install warp-lang`):

```bash
# Quick CUDA capability check
python -c "import warp as wp; wp.init(); print('devices:', wp.get_devices())"

# Run CUDA extraction/unit tests
python -m pytest -q

# Run CUDA pipeline smoke test (small)
python jit/code/synthesis/pipeline.py --count 5 --device cuda --output /tmp/jit_cuda_smoke
```

---

## Anti-Patterns (Avoid)
- ❌ Committing large generated datasets (keep samples small; prefer `/tmp` for bulk)
- ❌ Making CUDA required to import or run CPU paths
- ❌ Changing unrelated APIs/structure without clear need
- ❌ Leaving tests flaky or dependent on local state


