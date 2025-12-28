# JIT Code Synthesis - Dataset Generation & Scientific Report

## Objective
Generate large-scale training datasets for LLM training:
- **200MB CPU code** (Python→IR pairs using CPU backend)
- **200MB CUDA code** (Python→IR pairs using CUDA backend)
- **Scientific Report** for chief scientist covering JIT, IR, NVIDIA Warp, and dataset overview

---

## File Structure

```
jit/
├── instructions_dataset_report.md  # This file (read-only)
├── DATASET_STATE.md                # Current progress tracker
├── code/                           # Production code
│   ├── extraction/                 # IR extraction utilities
│   └── synthesis/                  # Data synthesis pipeline
├── data/
│   ├── cpu/                        # CPU backend dataset (target: 200MB)
│   └── cuda/                       # CUDA backend dataset (target: 200MB)
├── report/
│   └── scientific_report.md        # Report for chief scientist
└── notes/                          # Technical findings
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
- **CPU Data Size**: X MB (target: 200MB)
- **CUDA Data Size**: X MB (target: 200MB)

## Next Action
[Specific next step]

## Session Log
- [session]: [what was done]
```

---

## Phases

### P1: CPU Code Production (Target: 200MB)

**Goal**: Produce 200MB of Python→IR paired data using CPU backend

**Tasks**:

#### P1.1: Study & Select Best Base
1. Examine `cursor/agent-work-merge-process-*` branches:
   - `agent-work-merge-process-0038`
   - `agent-work-merge-process-0499`
   - `agent-work-merge-process-4dce`
   - `agent-work-merge-process-6964`
   - `agent-work-merge-process-96fd`
   - `agent-work-merge-process-ad19`
   - `agent-work-merge-process-bc08`
2. Also examine `cursor/following-instructions-md-*` Tier 1 branches:
   - `12c4` (10,727 pairs)
   - `9177` (10,320 pairs)
   - `8631` (10,101 pairs)
3. Evaluate each on:
   - Pipeline completeness
   - Kernel variety (7+ types preferred)
   - Code quality and reliability
   - Data output size per batch
4. Select best branch for production

#### P1.2: Reproduce & Validate
1. Copy production code from selected branch
2. Install dependencies: `pip install warp-lang`
3. Run pipeline with small batch to verify
4. Document any fixes needed

#### P1.3: Scale Production
1. Calculate samples needed for 200MB
   - Estimate: ~1-2KB per JSON sample → ~100k-200k samples
2. Run batch generation iteratively:
   ```bash
   python code/synthesis/batch_generator.py --output data/cpu/ --count 10000
   ```
3. Monitor disk usage: `du -sh data/cpu/`
4. Repeat until 200MB achieved
5. Push to remote periodically (every ~50MB)

**Done when**: `data/cpu/` contains ≥200MB of valid Python→IR pairs

---

### P2: CUDA Code Production (Target: 200MB)

**Goal**: Produce 200MB of Python→IR paired data using CUDA backend

**Tasks**:

#### P2.1: Analyze CUDA Adaptation Needs
1. Study `instruction_cuda.md` for guidance
2. Review ir_extractor.py for device parameter support
3. Identify kernel types that work on CUDA
4. Note: May need to generate code for GPU testing by human

#### P2.2: Adapt Pipeline
1. Modify ir_extractor.py to support `device="cuda"`
2. Update generator.py for CUDA-specific patterns if needed
3. Create test script for GPU validation (for human to run)

#### P2.3: Scale CUDA Production
1. Generate CUDA IR code (may produce `.cu` format annotations)
2. If no GPU available: generate code that CAN produce CUDA output
3. Target 200MB in `data/cuda/`
4. Push to remote periodically

**Done when**: `data/cuda/` contains ≥200MB of valid CUDA-targeted data

---

### P3: Scientific Report

**Goal**: Write comprehensive report for chief scientist

**Report Structure** (`report/scientific_report.md`):

```markdown
# JIT Code Synthesis for LLM Training Data
## Scientific Report

### 1. Introduction
- Project goals and motivation
- Training data for code LLMs

### 2. Just-In-Time (JIT) Compilation
- What is JIT compilation
- Benefits for numerical computing
- Python JIT landscape (Numba, JAX, Warp)

### 3. Intermediate Representation (IR)
- What is IR in compiler design
- Role of IR in code synthesis
- Why Python→IR pairs are valuable for LLMs

### 4. NVIDIA Warp
- Overview of Warp framework
- Kernel programming model
- IR extraction mechanism
- CPU vs CUDA code generation

### 5. Dataset Overview
- CPU dataset statistics
  - Total size, sample count
  - Kernel type distribution
  - Example samples
- CUDA dataset statistics
  - Total size, sample count
  - Kernel type distribution
  - Example samples

### 6. Data Format
- JSON structure
- Field descriptions
- Sample validation

### 7. Potential Applications
- LLM training for code generation
- IR understanding and optimization
- Cross-device code translation

### 8. Appendix
- Generation pipeline architecture
- Kernel type catalog
- Validation methodology
```

**Done when**: Report is complete, accurate, and suitable for chief scientist review

---

## Branch Reference

### CPU Code Sources (agent-work-merge-process-*)
| Branch | Suffix | Notes |
|--------|--------|-------|
| agent-work-merge-process-0038 | 0038 | Check for pipeline |
| agent-work-merge-process-0499 | 0499 | Check for pipeline |
| agent-work-merge-process-4dce | 4dce | Check for pipeline |
| agent-work-merge-process-6964 | 6964 | Check for pipeline |
| agent-work-merge-process-96fd | 96fd | Check for pipeline |
| agent-work-merge-process-ad19 | ad19 | Check for pipeline |
| agent-work-merge-process-bc08 | bc08 | Check for pipeline |

### Also Consider (following-instructions-md-*)
| Branch | Data Count | Features |
|--------|------------|----------|
| 12c4 | 10,727 | Full pipeline, batch generator |
| 9177 | 10,320 | Complete project |
| 8631 | 10,101 | Synthesis pipeline |

---

## Key Commands

```bash
# Check branch files
git ls-tree --name-only -r origin/cursor/BRANCH_NAME | head -30

# View file from branch
git show origin/cursor/BRANCH_NAME:path/to/file

# Copy file from branch
git show origin/cursor/BRANCH_NAME:path/to/file > local/file

# Check data size
du -sh data/cpu/ data/cuda/

# Count samples
find data/cpu -name "*.json" | wc -l

# Run batch generator
python code/synthesis/batch_generator.py --output data/cpu/ --count 10000
```

---

## Size Calculation Reference

Target: 200MB per dataset

Typical sample sizes:
- Simple kernel: ~500 bytes
- Complex kernel: ~2KB
- Average: ~1KB

Estimated samples needed:
- 200MB ÷ 1KB = ~200,000 samples
- Safety margin: target 250,000 samples

---

## Anti-Patterns (Avoid)

- ❌ Starting new phase before current phase complete
- ❌ Committing broken code
- ❌ Generating data without validation
- ❌ Skipping size checks before push
- ❌ Writing report before data generation complete

---

## Success Criteria

Project is complete when:
1. `data/cpu/` contains ≥200MB of valid Python→IR pairs
2. `data/cuda/` contains ≥200MB of valid CUDA-targeted data  
3. `report/scientific_report.md` is comprehensive and accurate
4. All data pushed to remote
5. Report reviewed for completeness
