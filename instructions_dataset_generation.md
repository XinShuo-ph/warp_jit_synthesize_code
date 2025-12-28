# Training Data Production (CPU + CUDA Code) + Chief Scientist Report

## Objective
Produce:
- **~200MB CPU code dataset** (Warp-generated CPU C++ kernel code)
- **~200MB CUDA code dataset** (Warp-generated CUDA C++ kernel code, generated via codegen without requiring a GPU)
- **A short markdown report** for the chief scientist introducing **JIT**, **IR**, **NVIDIA Warp**, and summarizing the produced datasets.

This work is derived from studying prior work branches:
- **CPU production candidates**: `cursor/agent-work-merge-*` / `cursor/agent-work-merge-process-*`
- **CUDA production candidates**: `cursor/cuda-backend-development-*`

Selected production bases:
- **CPU codegen extraction approach**: `cursor/agent-work-merge-process-bc08` (robust `warp._src.context.ModuleBuilder.codegen()` extraction)
- **Kernel diversity + CUDA workstream**: `cursor/cuda-backend-development-eb03` (10 kernel categories + CUDA-related tooling)

---

## File Structure (create as needed)

```
.
├── instructions_dataset_generation.md   # This file (read-only reference)
├── DATASET_STATE.md                     # CRITICAL: progress + next action
├── requirements.txt                     # Runtime dependencies
├── code/
│   ├── synthesis/                       # Kernel generators (10 categories)
│   ├── extraction/                      # IR/code extraction utilities
│   └── production/                      # Size-target dataset production scripts (this task)
├── data/
│   ├── production_cpu/                  # CPU dataset outputs (≈200MB)
│   ├── production_cuda/                 # CUDA dataset outputs (≈200MB)
│   └── manifests/                       # Dataset manifests + stats
└── reports/
    └── chief_scientist_report.md        # Requested markdown report
```

---

## State Management Protocol

### On Session Start
1. Read `DATASET_STATE.md`
2. Resume from the documented **Next Action**

### On Session End
1. Update `DATASET_STATE.md` with:
   - What was produced (paths + sizes)
   - Exact next action (command to run next)
   - Any blockers (dependency/tooling/runtime)
2. Leave the repo in a runnable state (scripts should work)

### DATASET_STATE.md Template
```markdown
# Dataset Production State
- **Phase**: P1_CPU | P2_CUDA | P3_REPORT
- **Status**: in_progress | blocked | completed

## Next Action
[Exact command(s) to run next]

## Outputs
- CPU dataset: [path], [size], [records/modules], [notes]
- CUDA dataset: [path], [size], [records/modules], [notes]
- Report: [path]

## Blockers (if any)
- [blocker]

## Session Log
- [date/session]: [high-signal summary]
```

---

## Workflow

### Phase 1 (P1_CPU): CPU Code Dataset (~200MB)
**Goal**: Produce ≈200MB of Warp-generated CPU C++ kernel code.

**Steps**:
1. Ensure dependencies installed: `python3 -m pip install -r requirements.txt`
2. Smoke test codegen on CPU (generate a tiny dataset)
3. Run size-target production until the dataset reaches ≈200MB
4. Write a manifest with category distribution, record counts, and sizes

**Done when**:
- `data/production_cpu/cpu_code_dataset.jsonl` is **≥ 200 MiB**
- `data/manifests/cpu_manifest.json` exists and matches the dataset

---

### Phase 2 (P2_CUDA): CUDA Code Dataset (~200MB)
**Goal**: Produce ≈200MB of Warp-generated CUDA C++ kernel code.

**Notes**:
- This pipeline uses Warp **codegen** and does **not** require a GPU driver.

**Steps**:
1. Smoke test CUDA codegen (generate a tiny dataset)
2. Run size-target production until the dataset reaches ≈200MB
3. Write a manifest with category distribution, record counts, and sizes

**Done when**:
- `data/production_cuda/cuda_code_dataset.jsonl` is **≥ 200 MiB**
- `data/manifests/cuda_manifest.json` exists and matches the dataset

---

### Phase 3 (P3_REPORT): Chief Scientist Report
**Goal**: Create a concise markdown report covering:
- JIT (what/why, where it fits)
- IR (what/why, typical forms)
- NVIDIA Warp (why it matters here, how kernels compile/codegen)
- Current datasets (schema, sizes, counts, category distribution, sample snippet references)

**Done when**:
- `reports/chief_scientist_report.md` exists and is readable in 2–5 minutes

---

## Anti-Patterns (Avoid)
- ❌ Depending on a real GPU to generate the CUDA dataset (codegen must work CPU-only)
- ❌ Generating datasets without manifests/stats
- ❌ Changing branches / doing interactive git operations
- ❌ Long-running background processes

