# Dataset and Report Generation

## Objective
Produce 200MB of CPU code dataset and 200MB of CUDA code dataset from existing branch implementations. Write a concise technical report in markdown format for the chief scientist, introducing JIT, IR, NVIDIA Warp, and the current dataset.

---

## File Structure (create as needed)

```
jit/
├── instructions_dataset_production.md  # This file (read-only)
├── PRODUCTION_STATE.md                  # Progress tracker
├── code/                                # Production code (from best branch)
├── data/
│   ├── cpu/                             # CPU dataset (~200MB)
│   └── cuda/                            # CUDA dataset (~200MB)
└── REPORT.md                            # Chief scientist report
```

---

## State Management Protocol

### On Session Start
1. Read `PRODUCTION_STATE.md` (create if missing)
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `PRODUCTION_STATE.md` with exact next action
2. Commit working changes
3. Push to remote
4. Stop—do not start new work

### PRODUCTION_STATE.md Template
```markdown
# Production State
- **Phase**: P1/P2/P3
- **Task**: [current task]
- **Status**: in_progress | blocked | completed
- **CPU Data Size**: X MB / 200 MB
- **CUDA Data Size**: X MB / 200 MB

## Next Action
[Specific next step]

## Session Log
- [session]: [what was done]
```

---

## Phases

### P1: CPU Code Production
**Goal**: Generate 200MB of CPU code dataset

**Tasks**:
1. Study `cursor/agent-work-merge-*` branches to understand available implementations
2. Reproduce each code implementation, test that it works
3. Pick the best branch for production (based on Tier 1 from `branch_progresses.md`)
4. Copy production code to `jit/code/`
5. Configure pipeline for CPU backend (`device="cpu"`)
6. Generate data incrementally, monitoring size
7. Target: ~200MB of Python→IR pairs in `jit/data/cpu/`
8. Push to remote periodically

**Done when**: `jit/data/cpu/` contains ~200MB of valid CPU code pairs

### P2: CUDA Code Production
**Goal**: Generate 200MB of CUDA code dataset

**Tasks**:
1. Study `cursor/cuda*` branches to understand CUDA implementations
2. Reproduce each code implementation, test that it works
3. Pick the best branch for CUDA production
4. Adapt pipeline for CUDA backend (`device="cuda"`)
5. Generate data incrementally, monitoring size
6. Target: ~200MB of Python→IR pairs in `jit/data/cuda/`
7. Push to remote periodically

**Done when**: `jit/data/cuda/` contains ~200MB of valid CUDA code pairs

### P3: Chief Scientist Report
**Goal**: Write technical report introducing the technology and dataset

**REPORT.md must include**:
```markdown
# JIT Code Synthesis Dataset - Technical Report

## Executive Summary
[2-3 sentences on what was built]

## 1. Introduction to JIT Compilation
- What is JIT (Just-In-Time) compilation
- Why it matters for ML/scientific computing

## 2. Intermediate Representation (IR)
- What is IR and its role in compilation
- IR as a bridge between high-level code and machine code

## 3. NVIDIA Warp
- Overview of the warp package
- How warp compiles Python to CPU/GPU code
- Key features: kernels, types, codegen

## 4. Dataset Overview
### 4.1 CPU Dataset
- Size: X MB
- Number of pairs: N
- Kernel types covered
- Sample format

### 4.2 CUDA Dataset
- Size: X MB  
- Number of pairs: N
- Kernel types covered
- Sample format

## 5. Data Format
[JSON structure with example]

## 6. Usage Notes
- How to load the data
- Recommended preprocessing for LLM training
```

**Done when**: `REPORT.md` is complete and accurate

---

## Branch Reference

### CPU Branches (cursor/agent-work-merge-*)
Based on `branch_progresses.md`:
- **Tier 1** (Production Ready): 12c4, 9177, 8631
- Use 12c4 as primary (10,727 pairs, full pipeline)

### CUDA Branches (cursor/cuda*)
- List and evaluate available CUDA branches
- Look for branches with `device="cuda"` support

---

## Data Size Estimation

Approximate sizes per 1000 pairs:
- Small kernels (~500 bytes each): ~1MB per 1000
- Medium kernels (~2KB each): ~4MB per 1000
- Large kernels (~5KB each): ~10MB per 1000

Target 200MB likely requires:
- ~40,000-200,000 pairs depending on kernel complexity
- Use batch generation with progress monitoring

---

## Key Commands

```bash
# Check data size
du -sh jit/data/cpu/
du -sh jit/data/cuda/

# Count pairs
find jit/data/cpu/ -name "*.json" | wc -l
find jit/data/cuda/ -name "*.json" | wc -l

# Generate batch
python jit/code/synthesis/batch_generator.py --count 10000 --output jit/data/cpu/

# Push incrementally
git add jit/data/
git commit -m "Add X pairs (Y MB total)"
git push origin HEAD
```

---

## Anti-Patterns (Avoid)

- ❌ Committing all data at once (use incremental pushes)
- ❌ Generating without monitoring size
- ❌ Writing report before data generation complete
- ❌ Skipping validation of generated data
- ❌ Over-engineering the report (keep it concise)

---

## Success Criteria

Production is complete when:
1. `jit/data/cpu/` contains ~200MB of valid Python→CPU IR pairs
2. `jit/data/cuda/` contains ~200MB of valid Python→CUDA IR pairs
3. `REPORT.md` is complete with all sections
4. All data validated (random sample check)
5. Everything committed and pushed
