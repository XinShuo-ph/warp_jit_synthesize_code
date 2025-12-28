# Training Data Production Pipeline

## Objective
Produce 400MB total training data (200MB CPU code, 200MB CUDA code) and write a technical report for the chief scientist covering JIT, IR, NVIDIA Warp, and the dataset.

---

## File Structure

```
/workspace/
├── instructions_production.md   # This file (read-only)
├── PRODUCTION_STATE.md          # Current progress tracker
├── code/                        # Production code (copied from best branch)
│   ├── extraction/              # IR extraction utilities
│   ├── synthesis/               # Data synthesis pipeline
│   └── examples/                # Example kernels
├── data/
│   ├── cpu/                     # CPU training data (200MB target)
│   └── cuda/                    # CUDA training data (200MB target)
└── report/
    └── chief_scientist_report.md  # Technical report
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
4. Stop—do not start new tasks

### PRODUCTION_STATE.md Template
```markdown
# Production State
- **Phase**: P1/P2/P3
- **Task**: [current task]
- **Status**: in_progress | blocked | completed
- **CPU Data Generated**: X MB / 200 MB
- **CUDA Data Generated**: X MB / 200 MB

## Next Action
[Specific next step]

## Session Log
- [session]: [what was done]
```

---

## Phases

### Phase 1: CPU Code Production
**Goal**: Generate 200MB of CPU-backend Python→IR paired training data

**Tasks**:
1. Study `cursor/agent-work-merge-*` branches:
   - List all available merge-related branches
   - Identify best production-ready branch (look at Tier 1: 12c4, 9177, 8631)
   - Reproduce pipeline from best branch locally
   - Validate the pipeline works
2. Select best branch based on:
   - Data pair count
   - Code quality
   - Pipeline reliability
3. Set up production environment:
   - Install dependencies (`pip install warp-lang`)
   - Copy production code to workspace
4. Gradual data production:
   - Start with small batch (100 pairs) to verify
   - Scale up incrementally (1000, 10000, etc.)
   - Monitor output size until reaching ~200MB
   - Use batch_generator.py for parallel generation
5. Push data to remote regularly

**Done when**: 200MB of CPU training data generated and pushed

---

### Phase 2: CUDA Code Production
**Goal**: Generate 200MB of CUDA-backend Python→IR paired training data

**Tasks**:
1. Study `cursor/cuda*` branches:
   - List all available CUDA-related branches
   - Identify best CUDA production branch
   - Reproduce pipeline from best branch locally
   - Validate the pipeline works with CUDA device parameter
2. Select best branch based on:
   - CUDA support completeness
   - Kernel type coverage
   - Pipeline reliability
3. Adapt production code:
   - Ensure device="cuda" parameter is supported
   - Verify CUDA IR output format (.cu files)
4. Gradual data production:
   - Start with small batch (100 pairs) to verify
   - Scale up incrementally
   - Monitor output size until reaching ~200MB
5. Push data to remote regularly

**Done when**: 200MB of CUDA training data generated and pushed

---

### Phase 3: Technical Report
**Goal**: Write a concise technical report for the chief scientist

**Report Structure** (`report/chief_scientist_report.md`):
```markdown
# JIT Code Synthesis Training Data Report

## Executive Summary
[1-2 paragraphs: what we built, key numbers]

## 1. JIT (Just-In-Time) Compilation
[What is JIT, why it matters for ML training data]

## 2. Intermediate Representation (IR)
[What is IR, types of IR (LLVM, PTX, etc.), why paired data is valuable]

## 3. NVIDIA Warp
[What is Warp, why we chose it, how it generates IR]

## 4. Dataset Overview
### 4.1 CPU Dataset
- Total size: X MB
- Number of pairs: N
- Kernel types: [list]
- Sample format: [JSON structure]

### 4.2 CUDA Dataset
- Total size: X MB  
- Number of pairs: N
- Kernel types: [list]
- Sample format: [JSON structure]

### 4.3 Quality Metrics
[Validation results, coverage statistics]

## 5. Production Pipeline
[Brief description of how data was generated]

## 6. Usage Recommendations
[How to use this data for LLM training]
```

**Done when**: Report reviewed and ready for chief scientist

---

## Key Commands Reference

```bash
# List merge branches
git branch -r | grep "cursor/agent-work-merge"

# List CUDA branches  
git branch -r | grep "cursor/cuda"

# View files from a branch
git show origin/BRANCH_NAME:path/to/file

# Copy code from branch
git checkout origin/BRANCH_NAME -- path/to/dir/

# Install warp
pip install warp-lang

# Run production pipeline
python code/synthesis/batch_generator.py --count 10000 --output data/cpu/

# Check data size
du -sh data/cpu/ data/cuda/

# Push to remote
git add -A
git commit -m "production: [description]"
git push origin HEAD
```

---

## Token Budget

| Phase | Budget | Activities |
|-------|--------|------------|
| P1 | ~100k | Study branches, reproduce, generate 200MB CPU data |
| P2 | ~100k | Study CUDA branches, adapt, generate 200MB CUDA data |
| P3 | ~30k | Write technical report |

---

## Anti-Patterns (Avoid)

- ❌ Generating all data at once (do incrementally)
- ❌ Skipping validation of pipeline before scaling
- ❌ Committing broken code
- ❌ Writing overly long reports (keep concise for chief scientist)
- ❌ Ignoring data quality for quantity

---

## Success Criteria

Phase 1-2 complete when:
1. 200MB CPU data generated and pushed
2. 200MB CUDA data generated and pushed
3. Data validates correctly (proper JSON format, IR extraction works)

Phase 3 complete when:
1. Report covers JIT, IR, Warp, and dataset
2. Report is concise (max 3-4 pages equivalent)
3. Clear for technical leadership to review
