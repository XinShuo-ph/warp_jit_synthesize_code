# Large-Scale Dataset Production & Technical Report

## Objective
Produce production-ready training datasets (200MB CPU code + 200MB CUDA code) and a technical report for chief scientist review covering JIT compilation, intermediate representations, NVIDIA Warp framework, and dataset characteristics.

---

## File Structure

```
/workspace/
├── instructions_wrapup.md      # This file (read-only reference)
├── PRODUCTION_STATE.md          # Current progress tracker
├── production/                  # Production code and data
│   ├── cpu/                     # CPU dataset production
│   │   ├── code/                # Selected production code
│   │   ├── data/                # Generated CPU dataset (200MB target)
│   │   └── production_log.md   # Generation statistics
│   ├── cuda/                    # CUDA dataset production  
│   │   ├── code/                # Selected CUDA production code
│   │   ├── data/                # Generated CUDA dataset (200MB target)
│   │   └── production_log.md   # Generation statistics
│   └── report/                  # Technical documentation
│       └── technical_report.md  # Final report for chief scientist
└── evaluation/                  # Branch evaluation notes
    ├── cpu_branches.md
    └── cuda_branches.md
```

---

## State Management Protocol

### On Session Start
1. Read `PRODUCTION_STATE.md` first
2. Resume from documented next action
3. Check data size progress: `du -sh production/*/data/`

### On Session End (or ~20k tokens remaining)
1. Update `PRODUCTION_STATE.md` with:
   - Current phase (P1/P2/P3)
   - Data generation progress (MB)
   - Next action
   - Key findings
2. Commit all changes
3. Push to remote
4. Stop—do not start new phase

### PRODUCTION_STATE.md Template
```markdown
# Production State
- **Phase**: P1_CPU / P2_CUDA / P3_REPORT
- **CPU Data**: X MB / 200 MB
- **CUDA Data**: Y MB / 200 MB
- **Status**: in_progress | ready_for_next

## Next Action
[Specific next step with commands/files]

## Progress This Session
- [accomplishment 1]
- [accomplishment 2]

## Generation Statistics
- CPU: X pairs, Y MB, Z iterations
- CUDA: X pairs, Y MB, Z iterations

## Session Log
- [session]: [what was done]
```

---

## Phase 1: CPU Dataset Production (Target: 200MB)

### Step 1: Evaluate CPU Branches (~10k tokens)

**Goal**: Identify best production code from `cursor/agent-work-merge-process-*` branches

#### 1a. Survey Branches
```bash
# List all agent-work-merge-process branches
git branch -a | grep agent-work-merge-process

# For each branch, check key files:
# - pipeline.py / batch_generator.py (production code)
# - data stats (how much generated)
# - code quality (tests, validation)
```

#### 1b. Test Top 3 Candidates
For each candidate branch:
```bash
# Create test workspace
mkdir -p /tmp/test_cpu_branch
cd /tmp/test_cpu_branch

# Extract production code
git show origin/cursor/agent-work-merge-process-XXXX:path/to/pipeline.py > pipeline.py
git show origin/cursor/agent-work-merge-process-XXXX:path/to/generator.py > generator.py

# Test generation (small batch)
python pipeline.py --count 10 --output /tmp/test_output

# Measure: 
# - Success rate
# - Data quality (valid Python→IR pairs)
# - Generation speed
# - Code size vs data size ratio
```

#### 1c. Document Findings
Create `evaluation/cpu_branches.md`:
```markdown
# CPU Branch Evaluation

## Tested Branches
| Branch | Pipeline Works | Speed | Quality | Recommendation |
|--------|---------------|-------|---------|----------------|
| XXXX   | Yes/No        | X/sec | High/Med/Low | ⭐ Best / Skip |

## Selected: [branch-XXXX]
**Rationale**: [why this is best for production]

## Key Files
- `path/to/pipeline.py`
- `path/to/generator.py`
- `path/to/batch_generator.py`
```

### Step 2: Setup Production Environment (~5k tokens)

```bash
# Create production structure
mkdir -p production/cpu/{code,data}

# Copy selected production code
git show origin/cursor/agent-work-merge-process-XXXX:path/to/pipeline.py > production/cpu/code/pipeline.py
git show origin/cursor/agent-work-merge-process-XXXX:path/to/generator.py > production/cpu/code/generator.py
git show origin/cursor/agent-work-merge-process-XXXX:path/to/batch_generator.py > production/cpu/code/batch_generator.py

# Copy any dependencies
# (ir_extractor.py, helper utilities, etc.)

# Commit setup
git add production/cpu/code/
git commit -m "P1: Setup CPU production environment from branch XXXX"
git push origin HEAD
```

### Step 3: Generate CPU Dataset (~100k tokens)

**Target**: 200MB of Python→IR paired data

#### 3a. Calculate Generation Parameters
```bash
# Test single sample size
python production/cpu/code/pipeline.py --count 1 --output /tmp/size_test
du -b /tmp/size_test/*.json | awk '{sum+=$1} END {print sum}'
# Example output: ~2048 bytes per sample

# Calculate needed samples
# 200 MB = 209,715,200 bytes
# Samples needed ≈ 209,715,200 / 2048 ≈ 102,400 samples
```

#### 3b. Iterative Batch Generation
```bash
# Generate in batches (commit + push periodically)
cd production/cpu/code

# Batch 1 (10k samples)
python batch_generator.py --count 10000 --output ../data/batch_001
du -sh ../data/
git add ../data/batch_001/
git commit -m "P1: CPU batch 1 - 10k samples (~20MB)"
git push origin HEAD

# Batch 2 (10k samples)
python batch_generator.py --count 10000 --output ../data/batch_002
du -sh ../data/
git add ../data/batch_002/
git commit -m "P1: CPU batch 2 - 10k samples (~40MB total)"
git push origin HEAD

# Continue until 200MB reached...
```

#### 3c. Track Progress
Update `production/cpu/production_log.md` after each batch:
```markdown
# CPU Dataset Production Log

## Target: 200 MB

| Batch | Samples | Size | Cumulative | Status |
|-------|---------|------|------------|--------|
| 001   | 10,000  | 20MB | 20MB       | ✓      |
| 002   | 10,000  | 20MB | 40MB       | ✓      |
| ...   | ...     | ...  | ...        | ...    |

## Generation Settings
- Kernel types: [list]
- Backend: CPU
- Validation: [method]

## Quality Checks
- Valid Python syntax: X%
- Valid IR extraction: Y%
- Compilation success: Z%
```

### Step 4: Validation (~10k tokens)

```bash
# Sample validation
python -c "
import json
import os
from pathlib import Path

data_dir = Path('production/cpu/data')
samples = list(data_dir.rglob('*.json'))

print(f'Total files: {len(samples)}')

# Validate 100 random samples
import random
test_samples = random.sample(samples, min(100, len(samples)))

valid = 0
for s in test_samples:
    data = json.load(open(s))
    if 'python_code' in data and 'ir_code' in data:
        valid += 1

print(f'Valid samples: {valid}/100')
"

# Check total size
du -sh production/cpu/data/
```

---

## Phase 2: CUDA Dataset Production (Target: 200MB)

### Step 1: Evaluate CUDA Branches (~10k tokens)

**Goal**: Identify best CUDA production code from `cursor/agent-work-merge-*` branches

**Note**: Per `instruction_cuda.md`, CUDA branches adapted CPU code to use CUDA backend

#### 1a. Survey CUDA Branches
```bash
# List relevant branches (likely agent-work-merge without "process")
git branch -a | grep 'agent-work-merge-[^p]'

# For each branch, check:
# - Does it have CUDA backend support?
# - Pipeline modified for CUDA?
# - Test suite for CUDA kernels?
```

#### 1b. Test CUDA Code Generation
For each candidate branch:
```bash
# Extract and test
mkdir -p /tmp/test_cuda_branch
cd /tmp/test_cuda_branch

# Get CUDA-enabled code
git show origin/cursor/agent-work-merge-XXXX:path/to/pipeline.py > pipeline.py

# Test generation (will generate code, but won't run on CPU-only machine)
python pipeline.py --backend cuda --count 5 --output /tmp/test_cuda

# Verify:
# - Code contains CUDA-specific syntax (device='cuda', wp.launch with CUDA device)
# - IR extraction adapted for CUDA backend
# - Generated samples have CUDA-specific IR
```

#### 1c. Document Findings
Create `evaluation/cuda_branches.md` with same structure as CPU evaluation

### Step 2: Setup CUDA Production (~5k tokens)

```bash
# Create CUDA production structure
mkdir -p production/cuda/{code,data}

# Copy selected CUDA code
git show origin/cursor/agent-work-merge-XXXX:path/to/pipeline.py > production/cuda/code/pipeline.py
git show origin/cursor/agent-work-merge-XXXX:path/to/generator.py > production/cuda/code/generator.py
git show origin/cursor/agent-work-merge-XXXX:path/to/batch_generator.py > production/cuda/code/batch_generator.py

# Commit
git add production/cuda/code/
git commit -m "P2: Setup CUDA production environment from branch XXXX"
git push origin HEAD
```

### Step 3: Generate CUDA Dataset (~100k tokens)

Same iterative approach as CPU:
- Calculate samples needed for 200MB
- Generate in batches (10k-20k samples per batch)
- Commit + push after each batch
- Track progress in `production/cuda/production_log.md`

```bash
cd production/cuda/code

# Batch generation loop
python batch_generator.py --backend cuda --count 10000 --output ../data/batch_001
du -sh ../data/
git add ../data/batch_001/
git commit -m "P2: CUDA batch 1 - 10k samples (~20MB)"
git push origin HEAD

# Continue until 200MB...
```

### Step 4: Validation (~10k tokens)

Same validation as CPU, but verify CUDA-specific characteristics:
- CUDA device specifications in IR
- CUDA kernel launch parameters
- PTX/SASS code in IR (if available)

---

## Phase 3: Technical Report (~30k tokens)

### Goal
Produce markdown report for chief scientist covering:
1. JIT compilation overview
2. Intermediate representations (IR)
3. NVIDIA Warp framework
4. Dataset characteristics and production process

### Report Structure

Create `production/report/technical_report.md`:

```markdown
# Training Dataset Production Report

**Date**: [date]
**Author**: AI Agent
**For**: Chief Scientist Review

---

## Executive Summary

This report documents the production of 400MB of JIT-compiled code training data:
- 200MB CPU-backend Python→IR pairs
- 200MB CUDA-backend Python→IR pairs

Total: [X] samples across [Y] kernel types.

---

## 1. Just-In-Time (JIT) Compilation

### Overview
[Explain JIT compilation: runtime translation, performance benefits, use cases]

### In NVIDIA Warp Context
[How Warp uses JIT to compile Python DSL → LLVM IR → native code]

### Relevance to LLM Training
[Why Python→IR pairs useful for code LLMs]

---

## 2. Intermediate Representations (IR)

### What is IR?
[Definition, purpose, abstraction level]

### LLVM IR in Warp
[Structure of Warp's LLVM IR output]
- Type system
- Instructions
- Control flow
- Memory model

### Example
[Show concrete Python kernel → IR transformation]

---

## 3. NVIDIA Warp Framework

### Overview
[What is Warp, key features, design goals]

### Architecture
[How Warp compiles kernels: Python DSL → IR → CUDA/CPU]

### Kernel Types Covered
[List kernel types in our dataset]
- Arithmetic operations
- Math functions
- Control flow
- Vector/matrix operations
- Memory operations
- [etc.]

---

## 4. Dataset Characteristics

### CPU Dataset
- **Size**: 200 MB
- **Samples**: [N] pairs
- **Backend**: CPU (LLVM)
- **Kernel Types**: [list]
- **Average Sample Size**: [X] bytes
- **Quality Metrics**:
  - Valid Python: [X]%
  - Valid IR: [Y]%
  - Compilation success: [Z]%

### CUDA Dataset  
- **Size**: 200 MB
- **Samples**: [N] pairs
- **Backend**: CUDA (PTX/SASS)
- **Kernel Types**: [list]
- **Average Sample Size**: [X] bytes
- **Quality Metrics**:
  - Valid Python: [X]%
  - Valid IR: [Y]%
  - CUDA-specific features: [list]

### Dataset Structure
[Explain JSON format, fields, organization]

---

## 5. Production Process

### Branch Evaluation
[How we selected production code]
- CPU: Selected branch [XXXX] for [reasons]
- CUDA: Selected branch [YYYY] for [reasons]

### Generation Pipeline
[Technical details of batch generation]
1. Kernel generation
2. Compilation
3. IR extraction
4. Validation
5. Storage

### Challenges & Solutions
[Any issues encountered and how resolved]

---

## 6. Recommendations

### For LLM Training
[How to use this dataset effectively]

### Future Work
[Potential improvements, additional kernel types, etc.]

---

## Appendices

### A. Sample Python→IR Pair (CPU)
[Complete example]

### B. Sample Python→IR Pair (CUDA)
[Complete example]

### C. Generation Statistics
[Detailed tables of production metrics]

### D. Code References
- CPU Production: `production/cpu/code/`
- CUDA Production: `production/cuda/code/`
- Branch Sources: [list]
```

### Writing Process

1. **Research phase** (~10k tokens):
   - Review Warp documentation
   - Study generated samples
   - Analyze production logs

2. **Writing phase** (~15k tokens):
   - Write each section
   - Include concrete examples from dataset
   - Add statistics from production logs

3. **Review phase** (~5k tokens):
   - Verify technical accuracy
   - Check completeness
   - Format for readability

---

## Final Validation Checklist

Before marking project complete:

```bash
# 1. CPU dataset size
du -sh production/cpu/data/
# Should be ≥ 200MB

# 2. CUDA dataset size  
du -sh production/cuda/data/
# Should be ≥ 200MB

# 3. Dataset integrity
python -c "
from pathlib import Path
import json

for dataset in ['cpu', 'cuda']:
    data_dir = Path(f'production/{dataset}/data')
    samples = list(data_dir.rglob('*.json'))
    
    valid = 0
    for s in samples[:100]:  # Check first 100
        try:
            data = json.load(open(s))
            if 'python_code' in data and 'ir_code' in data:
                valid += 1
        except:
            pass
    
    print(f'{dataset}: {len(samples)} samples, {valid}% valid (of 100 tested)')
"

# 4. Report exists and complete
wc -l production/report/technical_report.md
# Should be substantial (200+ lines)

# 5. All pushed to remote
git status
# Should be clean

git log --oneline -20
# Should show gradual commits throughout production
```

---

## Token Budget

| Phase | Activity | Budget |
|-------|----------|--------|
| P1 | Branch evaluation | ~15k |
| P1 | Setup + validation | ~10k |
| P1 | CPU generation (200MB) | ~100k |
| P2 | Branch evaluation | ~15k |
| P2 | Setup + validation | ~10k |
| P2 | CUDA generation (200MB) | ~100k |
| P3 | Report research | ~10k |
| P3 | Report writing | ~20k |
| Final | Validation | ~5k |

**Estimated total**: 285k tokens (~2-3 sessions with context refresh)

---

## Anti-Patterns (Avoid)

- ❌ Generating all data in single batch (memory issues, no checkpoints)
- ❌ Committing datasets without size validation
- ❌ Skipping branch evaluation (may use suboptimal code)
- ❌ Writing report before data generation complete
- ❌ Including large data samples in git (>100 files)
- ❌ Not pushing to remote periodically (data loss risk)
- ❌ Generating beyond 200MB target (waste resources)

---

## Success Criteria

Project complete when:
1. ✓ CPU dataset: 200MB of valid Python→IR pairs
2. ✓ CUDA dataset: 200MB of valid Python→IR pairs  
3. ✓ Technical report: Complete, accurate, ready for review
4. ✓ All code and data pushed to remote
5. ✓ Production logs document generation process
6. ✓ Validation confirms data integrity

---

## Notes

- **Gradual push strategy**: Commit and push after each batch to avoid data loss
- **Size monitoring**: Check `du -sh` frequently to track progress toward 200MB
- **Quality over quantity**: Valid, compilable samples more important than hitting exact 200MB
- **Report clarity**: Chief scientist may not be Warp expert—explain clearly
- **Branch selection**: Test before committing to a production code source
