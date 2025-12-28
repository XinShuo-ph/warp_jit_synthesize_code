# JIT Code Synthesis - CUDA Backend Development

## Objective
Adapt the existing Warp-based Python→IR synthesis pipeline to reliably produce **CUDA** generated code/IR (in addition to CPU), while keeping CPU behavior stable.

**Constraint**: This environment may not have a CUDA GPU available. Implementation must:
- Work on CPU-only machines (no hard CUDA dependency at runtime)
- Provide clear GPU test commands for you to run on a CUDA machine
- Add a small CUDA test suite that **skips** cleanly when CUDA is unavailable

---

## File Structure (create as needed)

```
jit/
├── instruction_cuda.md        # This file (read-only reference)
├── CUDA_STATE.md              # CRITICAL: CUDA progress tracker + next action
├── code/                      # Implementation code (must stay importable)
│   ├── extraction/            # IR extraction utilities (device-aware)
│   ├── synthesis/             # Generator/pipeline/batch generation (device-aware)
│   └── examples/              # Smoke tests & runnable demos (cpu/cuda)
├── tests/                     # pytest suite (cpu + cuda-skipped tests)
├── data/                      # Generated sample pairs (keep ≤100 in git)
└── notes/                     # Minimal findings (keep concise)
```

---

## State Management Protocol

### On Session Start
1. Read `jit/CUDA_STATE.md` (create if missing; template below)
2. Run the current CPU smoke command from the “Validation Protocol”
3. Resume from the documented **Next Action**

### On Session End (or ~20k tokens remaining)
1. Update `jit/CUDA_STATE.md` with:
   - Current phase + iteration (kernel type + forward/backward)
   - Exact next action (file + function + what to change)
   - Any blockers (e.g., Warp API mismatch, codegen differences)
2. Ensure repo is left in a runnable state on CPU-only machines
3. Stop—do not start a new iteration

### CUDA_STATE.md Template
```markdown
# CUDA State
- **Phase**: P0/P1/P2/P3
- **Iteration**: [kernel_type] / forward|backward
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next, with file paths and commands]

## Blockers (if any)
- [blocker]: [what was tried]

## Session Log
- [date/session]: [what changed, what was validated]
```

---

## Phases

### P0: Select Baseline + CPU Repro
**Goal**: Start from the best “production” baseline and confirm CPU pipeline still works.

**Baseline rule**: Prefer the top-ranked production branch from `branch_progresses.md` (typically `origin/cursor/following-instructions-md-12c4`) unless it is broken.

**Done when**:
- `python3 jit/code/synthesis/pipeline.py -n 3 --device cpu` succeeds
- `pytest -q` (if present) succeeds on CPU-only

---

### P1: Device Plumbing (CPU + CUDA)
**Goal**: Thread a `device` / `wp.Device` selection through:
- `jit/code/extraction/ir_extractor.py`
- `jit/code/synthesis/generator.py`
- `jit/code/synthesis/pipeline.py`
- `jit/code/synthesis/batch_generator.py`
- any validation utilities / examples

**Rules**:
- Default must remain CPU-safe: `device="cpu"` unless user explicitly requests CUDA
- CUDA paths must be guarded (skip/raise friendly error if CUDA unavailable)

**Done when**:
- CPU pipeline still works
- CUDA smoke script exists and prints a clear “CUDA not available -> skipped” message on CPU machines

---

### P1.5: CUDA Codegen-Only (No GPU Required)
**Goal**: Produce CUDA generated code/IR on machines **without** a CUDA driver/GPU by running Warp’s *codegen* for `device="cuda"` without launching kernels.

**Key idea**: Warp can emit CUDA generated code (e.g., `.cu`-style output) even when `wp.is_cuda_available()` is false. This milestone does **not** validate runtime execution on GPU—only that CUDA code generation succeeds.

**Requirements**:
- Add a `--codegen-only` path that allows `--device cuda` even when CUDA is unavailable
- Mark outputs with metadata indicating `codegen_only` and whether CUDA was available at generation time
- Add a pytest that asserts CUDA codegen works on CPU-only machines

**Done when**:
- `python3 jit/code/synthesis/cuda_codegen_pipeline.py -n 3` succeeds on a CPU-only machine
- `python3 -m pytest -q` passes (CUDA-runtime tests may still skip)

---

### P2: CUDA Iterations (kernel type × forward/backward)
**Goal**: Ensure each kernel type produces valid CUDA generated code/IR, for both forward and backward (if applicable).

**Iteration template (do one at a time)**:
1. Pick **one kernel type** (e.g. arithmetic, loops, branching, vec/mat, atomic, etc.)
2. Ensure **forward** works on CUDA (smoke test + extraction)
3. Ensure **backward** works on CUDA (if the kernel participates in autodiff; otherwise document “N/A”)
4. Add/extend one pytest test that is marked/skipped when CUDA isn’t available
5. Re-run CPU pipeline to ensure nothing regressed

**Done when**:
- For the chosen kernel type: CUDA smoke passes on your GPU machine
- Tests are present and skip cleanly without CUDA

---

### P3: Batch Generation + Validation on CUDA
**Goal**: Make the batch generator and validators work with CUDA device selection.

**Done when**:
- `python3 jit/code/synthesis/batch_generator.py -n 20 --device cuda ...` works on your GPU machine
- Output format includes device metadata and is consistent between cpu/cuda

---

## Validation Protocol

### CPU (must always work here)
```bash
python3 jit/code/synthesis/pipeline.py -n 3 --device cpu -o jit/data/samples_cpu
```

### CUDA (run on your GPU machine)
```bash
# Smoke + small pipeline run
python3 jit/code/examples/smoke_cuda.py
python3 jit/code/synthesis/pipeline.py -n 3 --device cuda -o jit/data/samples_cuda

# Tests (CUDA tests should be skipped if CUDA missing)
python3 -m pytest -q
```

---

## Anti-Patterns (Avoid)
- ❌ Making CUDA the default device
- ❌ Crashing import-time when CUDA isn’t available
- ❌ Generating/committing large datasets (keep ≤100 samples in git)
- ❌ Adding heavy dependencies beyond `warp-lang` unless strictly necessary

