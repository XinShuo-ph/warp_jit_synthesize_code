# Dataset Generation and Report for Chief Scientist

## Objective
Produce 200MB of CPU code training data and 200MB of CUDA code training data. Write a comprehensive markdown report for the chief scientist introducing JIT, IR, NVIDIA Warp, and the generated dataset.

---

## File Structure (create as needed)

```
jit/
├── instructions_dataset_report.md   # This file (read-only)
├── DATASET_STATE.md                  # Progress tracker
├── code/                             # Production code
│   ├── extraction/                   # IR extraction
│   └── synthesis/                    # Data generation pipeline
├── data/
│   ├── cpu/                          # CPU training data (~200MB)
│   └── cuda/                         # CUDA training data (~200MB)
└── REPORT.md                         # Report for chief scientist
```

---

## State Management Protocol

### On Session Start
1. Read `DATASET_STATE.md` (create if missing)
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `DATASET_STATE.md` with exact next action
2. Commit working changes
3. Push to remote
4. Stop—do not start new tasks

### DATASET_STATE.md Template
```markdown
# Dataset Generation State
- **Phase**: P1/P2/P3
- **Task**: [current task]
- **Status**: in_progress | blocked | completed

## Data Progress
- CPU data: [X] MB / 200 MB
- CUDA data: [X] MB / 200 MB

## Next Action
[Specific next step]

## Session Log
- [session]: [what was done]
```

---

## Phases

### P1: CPU Code Data Production
**Goal**: Generate 200MB of CPU-backend Python→IR paired training data

**Tasks**:
1. **Study CPU branches**: Explore `cursor/agent-work-merge-process-*` branches (7 branches)
   - List files and structure from each branch
   - Identify which has the most complete/production-ready pipeline
   - Test each pipeline with small batch generation

2. **Select best branch**: Pick the most complete and reliable pipeline based on:
   - Code completeness (generator, pipeline, batch_generator)
   - Kernel variety (number of kernel types)
   - Code quality and error handling
   
3. **Reproduce and set up**: Copy production code to working directory
   - Set up `code/extraction/` and `code/synthesis/`
   - Verify pipeline runs correctly
   - Run small test generation (10 samples)

4. **Scale up production**: Generate 200MB of CPU training data
   - Estimate: ~200MB ≈ 50,000-100,000 JSON pairs (at ~2-4KB each)
   - Generate in batches, verify periodically
   - Save to `data/cpu/`

5. **Push to remote**: Commit and push CPU dataset

**Done when**: `data/cpu/` contains ≥200MB of valid Python→IR pairs

---

### P2: CUDA Code Data Production
**Goal**: Generate 200MB of CUDA-backend Python→IR paired training data

**Tasks**:
1. **Study CUDA branches**: Explore `cursor/cuda-backend-development-*` branches (16 branches)
   - List files and structure from each branch
   - Identify which has the most complete CUDA pipeline
   - Check for GPU-specific kernel types and patterns

2. **Select best branch**: Pick the most complete CUDA pipeline based on:
   - CUDA-specific code generation
   - Kernel variety for GPU
   - Compatibility with production pipeline

3. **Reproduce and set up**: Merge CUDA code into working directory
   - Adapt CPU pipeline to use CUDA backend
   - Verify pipeline generates CUDA IR (`.cu` format)
   - Run small test generation (10 samples)

4. **Scale up production**: Generate 200MB of CUDA training data
   - Generate in batches, verify periodically
   - Save to `data/cuda/`

5. **Push to remote**: Commit and push CUDA dataset

**Done when**: `data/cuda/` contains ≥200MB of valid Python→CUDA IR pairs

---

### P3: Report for Chief Scientist
**Goal**: Write comprehensive markdown report on JIT, IR, NVIDIA Warp, and the dataset

**REPORT.md Structure**:
```markdown
# JIT Code Synthesis Training Data Report

## Executive Summary
[2-3 paragraph overview of the project and deliverables]

## 1. Introduction to JIT Compilation
### 1.1 What is JIT (Just-In-Time) Compilation?
### 1.2 Benefits of JIT for Scientific Computing
### 1.3 JIT in Python Ecosystem

## 2. Intermediate Representation (IR)
### 2.1 What is IR?
### 2.2 Role of IR in Code Generation
### 2.3 CPU vs GPU IR Differences

## 3. NVIDIA Warp
### 3.1 Overview of Warp
### 3.2 Warp's JIT Compilation Pipeline
### 3.3 Supported Backends (CPU/CUDA)
### 3.4 Kernel Types and Primitives

## 4. Dataset Overview
### 4.1 Data Generation Methodology
### 4.2 CPU Dataset Statistics
- Total size: X MB
- Number of samples: N
- Kernel type distribution
- Sample format

### 4.3 CUDA Dataset Statistics
- Total size: X MB
- Number of samples: N
- Kernel type distribution
- Sample format

### 4.4 Data Quality and Validation

## 5. Sample Data Examples
### 5.1 CPU Sample (Python → C++ IR)
### 5.2 CUDA Sample (Python → CUDA IR)

## 6. Potential Applications
### 6.1 LLM Training for Code Generation
### 6.2 Code Translation Tasks
### 6.3 Future Directions

## Appendix
### A. File Structure
### B. Reproduction Instructions
### C. References
```

**Done when**: `REPORT.md` is complete and covers all sections

---

## Branch Reference

### CPU Code Branches (agent-work-merge-process-*)
| Branch | Suffix |
|--------|--------|
| agent-work-merge-process-0038 | 0038 |
| agent-work-merge-process-0499 | 0499 |
| agent-work-merge-process-4dce | 4dce |
| agent-work-merge-process-6964 | 6964 |
| agent-work-merge-process-96fd | 96fd |
| agent-work-merge-process-ad19 | ad19 |
| agent-work-merge-process-bc08 | bc08 |

### CUDA Code Branches (cuda-backend-development-*)
| Branch | Suffix |
|--------|--------|
| cuda-backend-development-02a0 | 02a0 |
| cuda-backend-development-20ae | 20ae |
| cuda-backend-development-3072 | 3072 |
| cuda-backend-development-3fe7 | 3fe7 |
| cuda-backend-development-435e | 435e |
| cuda-backend-development-5cc5 | 5cc5 |
| cuda-backend-development-5f21 | 5f21 |
| cuda-backend-development-927f | 927f |
| cuda-backend-development-9686 | 9686 |
| cuda-backend-development-ced6 | ced6 |
| cuda-backend-development-db73 | db73 |
| cuda-backend-development-dff4 | dff4 |
| cuda-backend-development-e364 | e364 |
| cuda-backend-development-eb03 | eb03 |
| cuda-backend-development-f57c | f57c |
| cuda-backend-development-fb54 | fb54 |

---

## Git Commands Reference

```bash
# View files in branch without checkout
git ls-tree --name-only -r origin/cursor/agent-work-merge-process-{SUFFIX}

# Copy file from branch
git show origin/cursor/agent-work-merge-process-{SUFFIX}:path/to/file > local/path

# Check data size
du -sh data/cpu/ data/cuda/

# Count files and estimate size
find data/cpu/ -name "*.json" | wc -l
find data/cpu/ -name "*.json" -exec ls -l {} + | awk '{sum += $5} END {print sum/1024/1024 " MB"}'
```

---

## Token Budget

| Phase | Budget | Activities |
|-------|--------|------------|
| P1 (CPU) | ~100k | Study branches, reproduce, generate 200MB |
| P2 (CUDA) | ~100k | Study branches, reproduce, generate 200MB |
| P3 (Report) | ~30k | Write comprehensive report |

---

## Anti-Patterns (Avoid)

- ❌ Generating data without testing pipeline first
- ❌ Committing broken or incomplete code
- ❌ Skipping branch analysis (always evaluate before choosing)
- ❌ Writing report without actual data statistics
- ❌ Leaving session without updating DATASET_STATE.md
