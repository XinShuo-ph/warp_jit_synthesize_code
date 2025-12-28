# CPU + CUDA Code Dataset Generation (Warp JIT)

## Objective
Produce **~200MB of CPU-generated code** and **~200MB of CUDA-generated code** as LLM training data, using NVIDIA Warp’s Python→(generated C++/CUDA) codegen path. Then write a concise Markdown report for the chief scientist introducing **JIT**, **IR**, **NVIDIA Warp**, and summarizing the **current dataset** produced in this branch.

This work intentionally treats Warp’s generated C++ (`device="cpu"`) and generated CUDA (`device="cuda"`) as the “CPU code” and “CUDA code” datasets.

---

## File Structure (create as needed)

```
jit/
├── code/
│   ├── extraction/
│   │   └── ir_extractor.py            # Extract generated code from Warp kernels
│   └── synthesis/
│       ├── generator.py               # Programmatic kernel generator
│       └── produce_codegen_dataset.py # Produce 200MB+ CPU & CUDA code datasets
├── data/
│   ├── generated/
│   │   ├── cpu_code.jsonl             # ~200MB CPU code dataset (ignored by git)
│   │   ├── cuda_code.jsonl            # ~200MB CUDA code dataset (ignored by git)
│   │   └── dataset_stats.json         # Sizes, counts, category distribution
│   └── samples/                       # Small samples for quick validation (≤100)
└── REPORT.md                          # Chief scientist report
```

---

## State Management Protocol

### On Session Start
1. Identify your branch: `git branch --show-current`
2. Read this file (`instructions_dataset_generation.md`)
3. If present, read `DATASET_STATE.md` (create/update as needed)
4. Resume from the documented next action

### On Session End
1. Update `DATASET_STATE.md` with:
   - What was generated (paths + sizes)
   - What remains
   - Exact next action (command + args)
2. Keep code in a runnable state
3. (If operating locally) push to remote after verification

### DATASET_STATE.md Template
```markdown
# Dataset Generation State
- **Phase**: P1/P2/P3
- **Status**: in_progress | blocked | completed

## Next Action
[Exact next command to run + where outputs should appear]

## Outputs
- [path]: [size] [notes]

## Session Log
- [date/session]: [what changed]
```

---

## Workflow

### Phase 1: CPU code production (Warp `device="cpu"`)
**Goal**: Choose a production baseline and generate **~200MB** of Warp-generated CPU code.

**Steps**
1. **Study branches (no checkout required)**:
   - `origin/cursor/agent-work-merge-process-*` (merged candidates)
   - `origin/cursor/following-instructions-md-12c4` (Tier-1 reference baseline)
2. **Pick one best baseline for production**:
   - Prefer branches that already have: kernel generator variety, stable pipeline, and code extraction utilities.
3. **Generate CPU dataset**:
   - Run `python jit/code/synthesis/produce_codegen_dataset.py --device cpu --target-mb 200 ...`
   - Output: `jit/data/generated/cpu_code.jsonl`
4. **Validate**:
   - Check file size and that JSONL lines parse
   - Generate `jit/data/generated/dataset_stats.json`

**Deliverables**
- `jit/data/generated/cpu_code.jsonl` (≥200MB)
- `jit/data/generated/dataset_stats.json` includes CPU size + record count + category breakdown

---

### Phase 2: CUDA code production (Warp `device="cuda"`)
**Goal**: Generate **~200MB** of Warp-generated CUDA code.

**Steps**
1. Use the same kernel generator and extraction path as Phase 1
2. Generate CUDA dataset:
   - Run `python jit/code/synthesis/produce_codegen_dataset.py --device cuda --target-mb 200 ...`
   - Output: `jit/data/generated/cuda_code.jsonl`
3. Validate as in Phase 1

**Deliverables**
- `jit/data/generated/cuda_code.jsonl` (≥200MB)
- `jit/data/generated/dataset_stats.json` includes CUDA size + record count + category breakdown

---

### Phase 3: Report
**Goal**: Write a short report for the chief scientist.

**Deliverables**
- `jit/REPORT.md` must include:
  - What JIT is (high-level)
  - What “IR” means in this project context (Warp-generated code as an intermediate artifact)
  - What NVIDIA Warp is and why it’s relevant
  - Current dataset summary (paths, sizes, counts, categories)
  - How to reproduce dataset generation (commands)

---

## Validation Checklist
- `warp-lang` imports successfully
- CPU codegen works end-to-end on this machine
- CUDA codegen works end-to-end (code generation; not necessarily GPU execution)
- Both datasets meet the size targets (≥200MB each)
- `jit/REPORT.md` references the produced sizes and locations

