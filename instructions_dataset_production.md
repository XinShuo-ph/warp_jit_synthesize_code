# Dataset Production & Technical Report

## Objective
Generate production-scale training datasets (200MB CPU code + 200MB CUDA code) and produce a technical report for the chief scientist covering JIT compilation, IR representations, NVIDIA Warp, and the datasets.

---

## File Structure

```
/workspace/
├── instructions_dataset_production.md  # This file (read-only)
├── PRODUCTION_STATE.md                 # Progress tracker
├── datasets/
│   ├── cpu_code/                       # CPU-based Python→IR pairs
│   │   └── *.json
│   ├── cuda_code/                      # CUDA-based Python→IR pairs
│   │   └── *.json
│   └── statistics/                     # Dataset statistics
├── production_code/
│   ├── cpu_pipeline/                   # CPU production pipeline
│   └── cuda_pipeline/                  # CUDA production pipeline
└── report/
    └── chief_scientist_report.md       # Final technical report
```

---

## State Management Protocol

### On Session Start
1. Read `PRODUCTION_STATE.md` first
2. Resume from documented next action
3. Check current branch: `cursor/dataset-and-report-generation-0622`

### On Session End (or when buffer is low)
1. Update `PRODUCTION_STATE.md` with:
   - Current phase
   - Exact next action
   - Data generation progress (MB generated)
   - Key findings
2. Push to remote branch
3. Stop—do not start new work

### PRODUCTION_STATE.md Template
```markdown
# Production State
- **Phase**: P1/P2/P3
- **Current Branch**: cursor/dataset-and-report-generation-0622
- **Status**: in_progress | ready_for_next

## Progress Metrics
- CPU data generated: X MB / 200 MB
- CUDA data generated: Y MB / 200 MB

## Next Action
[Specific next step with commands/files]

## Key Findings This Session
- [finding 1]
- [finding 2]

## Session Log
- [session]: [what was done]
```

---

## Phase 1: CPU Code Production (Target: 200MB)

**Goal**: Select best CPU pipeline and generate 200MB of Python→IR training pairs

### Step 1: Evaluate CPU Branches (~20k tokens)

Review these branches in order:
1. `cursor/agent-work-merge-process-*` (8 branches) - merged work
2. `cursor/following-instructions-md-12c4` - 10,727 pairs
3. `cursor/following-instructions-md-9177` - 10,320 pairs
4. `cursor/following-instructions-md-8631` - 10,101 pairs

For each:
```bash
# Check pipeline completeness
git show origin/BRANCH:jit/code/synthesis/batch_generator.py > /tmp/test.py
python /tmp/test.py --help 2>&1 | head -20

# Check data quality
git ls-tree -r --name-only origin/BRANCH | grep "data/.*\.json$" | wc -l
```

Select criteria:
- ✓ Batch generator exists and runs
- ✓ Multiple kernel types (arithmetic, math, loop, conditional, vector, matrix)
- ✓ Clean IR extraction
- ✓ Validation passing

### Step 2: Setup Production Pipeline (~10k tokens)

```bash
# Copy best pipeline to production_code/cpu_pipeline/
git show origin/BEST_BRANCH:jit/code/synthesis/batch_generator.py > production_code/cpu_pipeline/batch_generator.py
git show origin/BEST_BRANCH:jit/code/synthesis/generator.py > production_code/cpu_pipeline/generator.py
git show origin/BEST_BRANCH:jit/code/extraction/ir_extractor.py > production_code/cpu_pipeline/ir_extractor.py

# Test on small batch
cd production_code/cpu_pipeline
python batch_generator.py --count 100 --output ../../datasets/cpu_code/test
```

### Step 3: Incremental Production (~100k tokens)

Generate in batches, push incrementally:

```bash
# Generate 10MB batches
for batch in {1..20}; do
    python batch_generator.py --count 5000 --output ../../datasets/cpu_code/batch_$batch
    
    # Check size
    du -sh ../../datasets/cpu_code
    
    # Push every 50MB
    if [ $((batch % 5)) -eq 0 ]; then
        cd /workspace
        git add datasets/cpu_code/
        git commit -m "CPU data: ${batch}0MB generated"
        git push origin HEAD
    fi
done
```

Monitor:
- Average file size: ~2KB per JSON pair
- Target: ~100,000 pairs for 200MB
- Validation: Random sample IR compilation check

### Step 4: Validation (~10k tokens)

```bash
# Statistics
python -c "
import json, os, glob
files = glob.glob('datasets/cpu_code/**/*.json', recursive=True)
total_size = sum(os.path.getsize(f) for f in files)
sample = json.load(open(files[0]))
print(f'Files: {len(files)}')
print(f'Size: {total_size / 1024 / 1024:.2f} MB')
print(f'Keys: {list(sample.keys())}')
" > datasets/statistics/cpu_stats.txt
```

---

## Phase 2: CUDA Code Production (Target: 200MB)

**Goal**: Select best CUDA pipeline and generate 200MB of CUDA Python→IR pairs

### Step 1: Evaluate CUDA Branches (~20k tokens)

Review `cursor/cuda-*` branches:

```bash
# List CUDA branches
git branch -a | grep cuda

# For each CUDA branch
git show origin/BRANCH:jit/code/synthesis/batch_generator.py | grep -i cuda
git show origin/BRANCH:jit/code/synthesis/generator.py | grep -i "wp.set_device\|cuda"
```

Select criteria:
- ✓ CUDA backend enabled (`wp.set_device("cuda")`)
- ✓ GPU kernel types (device functions, parallel launches)
- ✓ Memory operations (global, shared, local)
- ✓ Synchronization primitives

### Step 2: Setup CUDA Pipeline (~10k tokens)

```bash
# Copy best CUDA pipeline
git show origin/BEST_CUDA_BRANCH:jit/code/synthesis/batch_generator.py > production_code/cuda_pipeline/batch_generator.py
git show origin/BEST_CUDA_BRANCH:jit/code/synthesis/generator.py > production_code/cuda_pipeline/generator.py
git show origin/BEST_CUDA_BRANCH:jit/code/extraction/ir_extractor.py > production_code/cuda_pipeline/ir_extractor.py

# Verify CUDA patterns
grep -n "cuda\|gpu\|device" production_code/cuda_pipeline/*.py
```

### Step 3: Incremental Production (~100k tokens)

Generate in batches (same structure as CPU):

```bash
# Generate 10MB batches
for batch in {1..20}; do
    python batch_generator.py --count 5000 --backend cuda --output ../../datasets/cuda_code/batch_$batch
    
    du -sh ../../datasets/cuda_code
    
    if [ $((batch % 5)) -eq 0 ]; then
        cd /workspace
        git add datasets/cuda_code/
        git commit -m "CUDA data: ${batch}0MB generated"
        git push origin HEAD
    fi
done
```

### Step 4: Validation (~10k tokens)

```bash
# Statistics
python -c "
import json, os, glob
files = glob.glob('datasets/cuda_code/**/*.json', recursive=True)
total_size = sum(os.path.getsize(f) for f in files)
print(f'Files: {len(files)}')
print(f'Size: {total_size / 1024 / 1024:.2f} MB')
# Check for CUDA-specific patterns
sample = json.load(open(files[0]))
cuda_patterns = ['__global__', '__device__', '__shared__', 'threadIdx', 'blockIdx']
ir_code = sample.get('ir_code', '')
found = [p for p in cuda_patterns if p in ir_code]
print(f'CUDA patterns found: {found}')
" > datasets/statistics/cuda_stats.txt
```

---

## Phase 3: Technical Report (~30k tokens)

**Goal**: Write comprehensive report for chief scientist

### Report Structure

```markdown
# JIT Code Synthesis for LLM Training: Technical Report

## Executive Summary
[3-5 sentences: what was built, scale, key results]

## 1. Introduction to JIT Compilation
### 1.1 Just-In-Time Compilation Fundamentals
- Compilation at runtime vs ahead-of-time
- Performance benefits and trade-offs
- Use cases in scientific computing

### 1.2 Intermediate Representations (IR)
- Role of IR in compilation pipeline
- LLVM IR as industry standard
- IR optimization passes

## 2. NVIDIA Warp Framework
### 2.1 Overview
- Python-embedded language for GPU computing
- JIT compilation to native code
- Type system and kernel model

### 2.2 Architecture
- Frontend: Python decorators (@wp.kernel)
- Middle: IR generation (LLVM-based)
- Backend: CUDA/CPU code generation

### 2.3 IR Extraction Process
[Document the extraction mechanism from best pipeline]

## 3. Dataset Production
### 3.1 CPU Code Dataset
- Size: 200MB
- Pairs: ~100,000 Python→IR examples
- Kernel types: [list from generator.py]
- Quality metrics: [validation results]

### 3.2 CUDA Code Dataset
- Size: 200MB
- Pairs: ~100,000 CUDA Python→IR examples
- GPU-specific patterns: [list]
- Quality metrics: [validation results]

### 3.3 Generation Pipeline
[Diagram/description of batch_generator.py workflow]

## 4. Dataset Characteristics
### 4.1 Diversity Metrics
- Kernel complexity distribution
- IR instruction variety
- Code pattern coverage

### 4.2 Quality Assurance
- Validation methodology
- Error rates
- Sample quality checks

## 5. Recommendations for LLM Training
### 5.1 Dataset Usage
- Suggested train/validation split
- Tokenization considerations
- Data augmentation opportunities

### 5.2 Future Work
- Additional kernel types
- Multi-GPU patterns
- Advanced optimizations

## Appendices
### A. Sample Data Pairs
[3-5 representative examples]

### B. Pipeline Code References
[Key functions with file:line references]

### C. Generation Statistics
[Full dataset statistics]
```

### Report Generation Steps

1. **Gather data** (~5k tokens):
   ```bash
   # Dataset statistics
   du -sh datasets/cpu_code datasets/cuda_code
   find datasets -name "*.json" | wc -l
   
   # Sample pairs
   head -1 datasets/cpu_code/batch_1/*.json | head -3
   head -1 datasets/cuda_code/batch_1/*.json | head -3
   ```

2. **Write sections incrementally** (~20k tokens):
   - Write 1-2 sections at a time
   - Include code references from production_code/
   - Add actual statistics from datasets/statistics/

3. **Review and refine** (~5k tokens):
   - Check technical accuracy
   - Ensure clarity for chief scientist audience
   - Verify all references resolve

---

## Validation Checklist

Before marking complete:

- [ ] CPU dataset: ≥200MB, validated
- [ ] CUDA dataset: ≥200MB, validated
- [ ] Statistics files generated
- [ ] Report complete with all sections
- [ ] Sample data pairs in report
- [ ] All code pushed to remote
- [ ] PRODUCTION_STATE.md up to date

---

## Token Budget

| Phase | Budget | Activities |
|-------|--------|------------|
| P1: CPU production | ~140k | Branch eval, setup, generation, validation |
| P2: CUDA production | ~140k | Branch eval, setup, generation, validation |
| P3: Report writing | ~30k | Research, write, refine |
| **Total** | ~310k | Estimated 2-3 sessions |

---

## Success Criteria

Project complete when:
1. 200MB CPU dataset in `datasets/cpu_code/`, pushed to remote
2. 200MB CUDA dataset in `datasets/cuda_code/`, pushed to remote
3. Technical report in `report/chief_scientist_report.md`
4. Report covers JIT, IR, NVIDIA Warp, and datasets comprehensively
5. All validation checks pass
6. Production pipelines documented in `production_code/`
