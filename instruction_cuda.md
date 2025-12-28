## Objective
Adapt the current Warp Python→IR synthesis codebase to support a **CUDA backend** in addition to CPU.

**Constraint**: This environment may not have a GPU. All CUDA work must be implemented so it can be tested later on a GPU machine via clear commands and a GPU-only test suite.

---

## File Structure (create as needed)

```
jit/
├── instruction_cuda.md        # This file (read-only reference)
├── CUDA_STATE.md              # CRITICAL: CUDA progress tracker
├── code/
│   ├── extraction/            # IR extraction utilities (CPU+CUDA)
│   ├── synthesis/             # generator/pipeline/batch for CPU+CUDA
│   ├── validation/            # dataset + extraction validators
│   └── examples/              # minimal kernels used by tests
├── tests/
│   ├── test_cpu_smoke.py       # always runnable in this env
│   └── test_cuda_smoke.py      # skips unless CUDA is available
└── notes/
    └── cuda_findings.md        # concise findings + known issues (≤100 lines)
```

---

## State Management Protocol

### On Session Start
1. Read `jit/CUDA_STATE.md` (create if missing).
2. Resume from **Next Action** exactly.

### On Session End (or ~20k tokens remaining)
1. Update `jit/CUDA_STATE.md` with:
   - Current phase and iteration
   - Exact next action (file + function + what to change/run)
   - Blockers (if any)
2. Leave the workspace in a runnable state (CPU tests must pass).

### CUDA_STATE.md Template
```markdown
# CUDA State
- **Phase**: P0/P1/P2/P3
- **Iteration**: I0/I1/... (one kernel family or one subsystem at a time)
- **Status**: in_progress | blocked | ready_for_next | completed

## Next Action
[Exact next step with commands and/or file edits]

## Blockers (if any)
- [blocker]: [what was tried]

## Session Log
- [date/session]: [what changed, what was verified]
```

---

## Phases

### P0: Select Baseline + Reproduce CPU
**Goal**: Pull a known-good “production” baseline into `jit/` and verify CPU pipeline works.

**Rules**:
- Prefer the most complete branch as baseline (typically the largest dataset / full pipeline).
- Keep CPU behavior unchanged while adding CUDA support (no breaking changes).

**Done when**:
- `python -m pytest jit/tests/test_cpu_smoke.py -q` passes
- `python jit/code/synthesis/pipeline.py --device cpu --count 5 --output jit/data/test_cpu` succeeds

### P1: Introduce CUDA Plumbing (No GPU Required)
**Goal**: Make device selection a first-class option across extractor + pipeline + batch tools.

**Requirements**:
- Every entrypoint that compiles/launches kernels accepts `--device {cpu,cuda}` (default: cpu).
- Extractor supports CUDA compilation mode and returns CUDA code/IR when available.
- CUDA tests **must auto-skip** if CUDA is unavailable, but run on GPU machines.

**Done when**:
- CPU tests still pass
- GPU machine command (documented below) runs and produces `.cu`/CUDA output

### P2: CUDA Coverage Iterations (Kernel Families + Forward/Backward)
**Goal**: Iteratively ensure every kernel family is CUDA-compatible, including gradient path where applicable.

**Iteration workflow (repeat)**:
1. Pick **one kernel family** (e.g. arithmetic, control_flow, vector, reduction, atomic…).
2. Add/adjust **one minimal example kernel** in `jit/code/examples/`.
3. Ensure:
   - Forward compilation/launch works on CUDA
   - Backward/adjoint path works if the kernel participates in gradients (Warp `Tape`)
4. Add/extend CUDA test covering that family in `jit/tests/test_cuda_smoke.py`.
5. Ensure CPU suite remains green.

**Done when**:
- All generator kernel families can compile for CUDA (on a GPU machine)
- CUDA test suite covers forward + at least one backward case

### P3: Batch Generation + Validation on CUDA
**Goal**: Ensure batch dataset generation and validators work for CUDA outputs.

**Done when**:
- `jit/code/synthesis/batch_generator.py --device cuda ...` works on GPU machine
- Validators accept CPU+CUDA artifacts and report clear errors

---

## GPU Machine Commands (for you to run)

```bash
# 1) Install deps
python -m pip install -U pip
python -m pip install -U warp-lang pytest

# 2) CPU sanity (should always pass)
python -m pytest jit/tests/test_cpu_smoke.py -q

# 3) CUDA sanity (will run only if CUDA is available)
python -m pytest jit/tests/test_cuda_smoke.py -q

# 4) Generate a small CUDA dataset
python jit/code/synthesis/pipeline.py --device cuda --count 10 --output jit/data/test_cuda
```

---

## Validation Protocol

Before marking any phase/iteration complete:
1. Run CPU smoke tests twice.
2. Ensure CLI help (`--help`) documents the device flag everywhere relevant.
3. No debug prints, no hard-coded device assumptions.

---

## Anti-Patterns (Avoid)

- ❌ Writing CUDA-only code paths that break CPU
- ❌ Assuming a GPU is present (CUDA tests must skip cleanly)
- ❌ Large refactors while “adding CUDA”
- ❌ Generating large datasets in git (keep committed samples ≤100)

---

## Success Criteria

CUDA backend work is complete when:
1. CPU pipeline continues to work unchanged
2. CUDA pipeline works on a GPU machine via the documented commands
3. `jit/tests/test_cuda_smoke.py` provides meaningful forward+backward coverage
4. Batch generation and validation tools support `--device cuda`


