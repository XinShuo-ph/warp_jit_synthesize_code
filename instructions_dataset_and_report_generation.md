# Dataset + Report Generation

## Objective
Produce **~200MB CPU code dataset** and **~200MB CUDA code dataset** for training, then write a short scientist-facing Markdown report introducing **JIT**, **IR**, **NVIDIA Warp**, and summarizing the **current dataset** produced in this run.

This run should **reuse the best existing production pipelines** found in:
- CPU: `origin/cursor/agent-work-merge-process-*`
- CUDA: `origin/cursor/cuda-backend-development-*`

---

## File Structure

```
.
├── instructions_dataset_and_report_generation.md   # This file (read-only reference)
├── DATASET_STATE.md                               # CRITICAL: progress + next action
├── tools/
│   └── generate_dataset_to_size.py                # Size-targeted generator (CPU/CUDA)
├── artifacts/                                     # NOT committed (see .gitignore)
│   └── datasets/
│       ├── cpu/
│       │   ├── dataset.jsonl
│       │   └── stats.json
│       └── cuda/
│           ├── dataset.jsonl
│           └── stats.json
└── REPORT.md                                      # Chief scientist report
```

---

## State Management Protocol

### On Session Start
1. Read `DATASET_STATE.md`
2. Resume from **Next Action**

### On Session End
1. Update `DATASET_STATE.md` with:
   - What was generated (bytes + sample counts)
   - Exact next action (command + file paths)
   - Any blockers (missing deps, CUDA toolkit, etc.)
2. Do not leave partially-written instructions or broken scripts

### `DATASET_STATE.md` Template
```markdown
# Dataset State
- **Phase**: P1/P2/P3
- **Status**: in_progress | blocked | completed

## Targets
- CPU dataset bytes: 200MB
- CUDA dataset bytes: 200MB

## Next Action
[Exact command(s) to run next]

## Progress
- CPU: [bytes], [records], [path]
- CUDA: [bytes], [records], [path]

## Blockers (if any)
- ...

## Session Log
- [timestamp]: [what changed]
```

---

## Phases

### P1: CPU Dataset Production
**Goal**: Generate ~200MB of CPU code records (Warp-generated C/C++ code extracted from JIT codegen).

**Workflow**:
1. Study `origin/cursor/agent-work-merge-process-*` branches and pick the best production base.
2. Reproduce the chosen pipeline locally.
3. Generate dataset until `artifacts/datasets/cpu/dataset.jsonl` reaches ≥200MB.
4. Save summary stats to `artifacts/datasets/cpu/stats.json`.

**Done when**:
- `artifacts/datasets/cpu/dataset.jsonl` size ≥ 200MB
- `artifacts/datasets/cpu/stats.json` exists and matches the dataset

### P2: CUDA Dataset Production
**Goal**: Generate ~200MB of CUDA code records (Warp-generated CUDA C code extracted from JIT codegen).

**Workflow**:
1. Study `origin/cursor/cuda-backend-development-*` branches and pick the best production base.
2. Reproduce the chosen pipeline locally (CPU-only codegen is acceptable; execution may require GPU).
3. Generate dataset until `artifacts/datasets/cuda/dataset.jsonl` reaches ≥200MB.
4. Save summary stats to `artifacts/datasets/cuda/stats.json`.

**Done when**:
- `artifacts/datasets/cuda/dataset.jsonl` size ≥ 200MB
- `artifacts/datasets/cuda/stats.json` exists and matches the dataset

### P3: Report
**Goal**: Write a short Markdown report for the chief scientist.

**Must include**:
- Brief explanation of JIT and why it is relevant for code generation datasets
- What IR means in this project (Warp-generated C++/CUDA source as compiler IR proxy)
- A concise intro to NVIDIA Warp (what it is, why used)
- Current dataset summary (CPU/CUDA sizes, record counts, schema overview)

**Deliverable**: `REPORT.md`

---

## Validation Checklist

Before marking complete:
1. `python3 tools/generate_dataset_to_size.py --help` works
2. CPU dataset generation completes and stats file matches produced bytes
3. CUDA dataset generation completes (codegen-only is acceptable if no GPU)
4. `REPORT.md` reflects actual generated numbers

