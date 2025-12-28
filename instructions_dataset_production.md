# Dataset Production and Technical Report

## Objective
Generate large-scale training datasets (200MB each of CPU and CUDA code) and produce a technical report for the chief scientist covering JIT compilation, intermediate representations, NVIDIA Warp, and the generated datasets.

---

## File Structure

```
/workspace/
├── instructions_dataset_production.md  # This file (read-only)
├── PRODUCTION_STATE.md                 # Progress tracker
├── production/                         # Working directory
│   ├── cpu_code/                      # CPU dataset output
│   ├── cuda_code/                     # CUDA dataset output
│   ├── scripts/                       # Production scripts
│   └── validation/                    # Validation tools
└── report/                            # Technical report
    ├── REPORT.md                      # Main report
    └── figures/                       # Supporting materials
```

---

## State Management Protocol

### On Session Start
1. Read `PRODUCTION_STATE.md` first
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `PRODUCTION_STATE.md` with:
   - Current phase
   - Exact next action
   - Progress metrics (MB generated)
   - Any blockers
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
- **CUDA Data Generated**: Y MB / 200 MB

## Next Action
[Specific next step with exact commands/files]

## Progress Metrics
- CPU samples generated: N
- CUDA samples generated: M
- Generation rate: X samples/min
- Estimated time remaining: Y hours

## Blockers (if any)
[What's preventing progress]

## Session Log
- [timestamp]: [what was accomplished]
```

---

## Phases

### Phase 1: CPU Code Production (Target: 200MB)

**Goal**: Select best CPU code generation approach and produce 200MB dataset

#### Task 1.1: Survey CPU Code Branches (~10k tokens)
Study all `cursor/agent-work-merge-*` branches to understand approaches:
```bash
# List all agent-work-merge branches
git branch -a | grep agent-work-merge

# For each branch, examine:
git show origin/cursor/agent-work-merge-{SUFFIX}:jit/code/synthesis/pipeline.py 2>/dev/null || echo "No pipeline"
git show origin/cursor/agent-work-merge-{SUFFIX}:jit/code/synthesis/generator.py 2>/dev/null || echo "No generator"
git ls-tree -r --name-only origin/cursor/agent-work-merge-{SUFFIX} | head -20
```

Document in `production/cpu_analysis.md`:
```markdown
# CPU Code Generation Analysis

## Branch Comparison
| Branch | Has Pipeline | Has Generator | Data Exists | Notes |
|--------|--------------|---------------|-------------|-------|
| 1496   | Yes/No       | Yes/No        | Yes/No      | ...   |
| ...    | ...          | ...           | ...         | ...   |

## Recommended Branch
- **Selected**: [branch suffix]
- **Rationale**: [why this is best]

## Key Features
- Generation rate: X samples/min
- Kernel types supported: [list]
- Code quality: [assessment]
```

**Done when**: `production/cpu_analysis.md` exists with clear recommendation

#### Task 1.2: Reproduce Best CPU Code (~20k tokens)
```bash
# Create test environment
mkdir -p /tmp/cpu_test production/scripts

# Copy code from best branch
git show origin/cursor/agent-work-merge-{BEST}:jit/code/synthesis/pipeline.py > production/scripts/cpu_pipeline.py
git show origin/cursor/agent-work-merge-{BEST}:jit/code/synthesis/generator.py > production/scripts/cpu_generator.py
git show origin/cursor/agent-work-merge-{BEST}:jit/code/extraction/ir_extractor.py > production/scripts/cpu_ir_extractor.py

# Install dependencies
pip install warp-lang

# Test run
cd production/scripts
python cpu_pipeline.py --count 10 --output /tmp/cpu_test
```

**Validation**:
- Pipeline runs without errors
- Generates valid JSON files
- IR extraction works correctly
- At least 3 different kernel types generated

**Done when**: Can generate 10+ samples successfully

#### Task 1.3: Production-Scale CPU Generation (~50k tokens)
Create `production/scripts/cpu_production.py`:
```python
"""
Batch CPU code generation script
Target: 200MB of training data
"""
import os
import json
import time
from pathlib import Path

# Configuration
TARGET_SIZE_MB = 200
OUTPUT_DIR = Path("production/cpu_code")
BATCH_SIZE = 1000
CHECKPOINT_INTERVAL = 100

# Main generation loop with progress tracking
# Include size estimation and checkpoint saving
# Add validation checks
```

Run production:
```bash
cd production/scripts
python cpu_production.py

# Monitor progress
watch -n 10 'du -sh ../cpu_code && ls ../cpu_code | wc -l'
```

**Progress checkpoints**:
- [ ] 50MB generated
- [ ] 100MB generated
- [ ] 150MB generated
- [ ] 200MB generated

**Done when**: `production/cpu_code/` contains 200MB of valid data

#### Task 1.4: Push CPU Dataset (~5k tokens)
```bash
# Commit and push (split if needed due to size)
git add production/cpu_code/
git commit -m "Production: CPU dataset 200MB"
git push origin HEAD

# If too large, use git-lfs or split:
# git lfs track "production/cpu_code/*.json"
# Or compress: tar -czf cpu_data.tar.gz production/cpu_code/
```

**Done when**: CPU dataset pushed to remote

---

### Phase 2: CUDA Code Production (Target: 200MB)

**Goal**: Adapt best approach for CUDA and produce 200MB dataset

#### Task 2.1: Survey CUDA Branches (~10k tokens)
Study all CUDA-related branches (or follow-up branches with GPU work):
```bash
# Search for CUDA-related code
for branch in $(git branch -a | grep origin/cursor | awk -F'/' '{print $NF}'); do
  echo "=== $branch ==="
  git show origin/cursor/$branch:jit/code/extraction/ir_extractor.py 2>/dev/null | grep -i "cuda\|gpu\|device" | head -5
done
```

Document in `production/cuda_analysis.md`:
```markdown
# CUDA Code Generation Analysis

## Branch Comparison
| Branch | CUDA Support | Device Param | Tests GPU | Notes |
|--------|--------------|--------------|-----------|-------|
| ...    | Yes/No       | Yes/No       | Yes/No    | ...   |

## Recommended Approach
- **Selected**: [branch or manual adaptation]
- **Rationale**: [why]

## Key Differences from CPU
- IR format changes: [list]
- Kernel modifications needed: [list]
- Generation strategy: [description]
```

**Done when**: `production/cuda_analysis.md` exists with clear plan

#### Task 2.2: Adapt Code for CUDA (~30k tokens)
Based on analysis, either:
- Copy from CUDA branch, or
- Modify CPU code for CUDA

Key modifications:
1. Change device parameter: `device="cuda"`
2. Update IR extraction to capture `.cu` files
3. Add CUDA-specific kernel patterns:
   - Thread indexing (`wp.tid()`)
   - Shared memory patterns
   - Atomic operations
   - Block/grid dimensions

Create `production/scripts/cuda_pipeline.py`:
```python
"""
CUDA code generation pipeline
Differences from CPU:
- device="cuda" in kernel compilation
- Extract .cu files instead of .cpp
- CUDA-specific kernel patterns
"""
```

**Validation**:
```bash
# Test CUDA pipeline (will fail without GPU, that's OK)
python cuda_pipeline.py --count 5 --output /tmp/cuda_test

# Verify output format:
# - Kernels have CUDA syntax
# - IR contains GPU-specific code
# - File extensions correct (.cu)
```

**Done when**: CUDA pipeline generates valid-looking CUDA code (even if not GPU-tested)

#### Task 2.3: Production-Scale CUDA Generation (~50k tokens)
Create `production/scripts/cuda_production.py`:
```python
"""
Batch CUDA code generation script
Target: 200MB of training data
"""
# Similar structure to cpu_production.py
# but with CUDA-specific configuration
```

Run production:
```bash
cd production/scripts
python cuda_production.py

# Monitor progress
watch -n 10 'du -sh ../cuda_code && ls ../cuda_code | wc -l'
```

**Progress checkpoints**:
- [ ] 50MB generated
- [ ] 100MB generated
- [ ] 150MB generated
- [ ] 200MB generated

**Done when**: `production/cuda_code/` contains 200MB of valid CUDA data

#### Task 2.4: Push CUDA Dataset (~5k tokens)
```bash
git add production/cuda_code/
git commit -m "Production: CUDA dataset 200MB"
git push origin HEAD
```

**Done when**: CUDA dataset pushed to remote

---

### Phase 3: Technical Report (~40k tokens)

**Goal**: Write comprehensive markdown report for chief scientist

#### Task 3.1: Report Structure and Content

Create `report/REPORT.md` with following sections:

```markdown
# Technical Report: JIT Code Synthesis for LLM Training

**Prepared for**: Chief Scientist  
**Date**: [date]  
**Author**: Agent  
**Branch**: cursor/dataset-and-report-generation-891a

---

## Executive Summary
[2-3 paragraphs: what was done, key results, significance]

- Generated 200MB CPU code dataset
- Generated 200MB CUDA code dataset
- Datasets suitable for training code generation models

---

## 1. Background: JIT Compilation and Intermediate Representations

### 1.1 Just-In-Time (JIT) Compilation
[Explain JIT concept, 300-400 words]
- What is JIT compilation
- Why it matters for performance
- Examples in modern frameworks (PyTorch, JAX, etc.)
- Advantages over AOT compilation

### 1.2 Intermediate Representations (IR)
[Explain IR concept, 300-400 words]
- Role of IR in compilation pipeline
- Common IR formats (LLVM IR, MLIR, etc.)
- Why IR is valuable for ML training
- Source Code → IR → Machine Code flow

### 1.3 Python to Low-Level Code Translation
[Explain the Python→IR problem, 200-300 words]
- Challenges of translating dynamic Python
- Type inference and specialization
- Benefits for ML model training

---

## 2. NVIDIA Warp Framework

### 2.1 Overview
[Introduce Warp, 300-400 words]
- What is NVIDIA Warp
- Design goals and use cases
- Key features (kernels, types, autodiff)
- Comparison to CUDA/C++ direct programming

### 2.2 Architecture
[Explain Warp internals, 400-500 words]
- Python kernel definition via decorators
- Type system (wp.vec3, wp.mat33, etc.)
- Code generation pipeline
- Runtime compilation and caching

### 2.3 Backend Support
[Explain CPU/CUDA backends, 300-400 words]
- CPU backend: generates C++ code
- CUDA backend: generates .cu files
- Device abstraction and portability
- Performance characteristics

---

## 3. Dataset Generation Methodology

### 3.1 CPU Code Dataset

#### 3.1.1 Branch Selection Process
[Explain how best branch was chosen, 200-300 words]
- Branches evaluated: [list]
- Selection criteria
- Chosen branch: [name]
- Rationale for selection

#### 3.1.2 Generation Pipeline
[Describe CPU pipeline, 400-500 words]
- Kernel generation strategy
- IR extraction mechanism
- Data format and structure
- Quality validation

#### 3.1.3 Kernel Type Coverage
[List kernel types, 200-300 words]
- Arithmetic operations
- Mathematical functions
- Control flow (loops, conditionals)
- Vector/matrix operations
- [other types from actual implementation]

#### 3.1.4 Production Statistics
```
Total Size: 200 MB
Total Samples: [N]
Kernel Types: [M]
Average Sample Size: [X KB]
Generation Time: [Y hours]
Generation Rate: [Z samples/min]
```

### 3.2 CUDA Code Dataset

#### 3.2.1 Adaptation Strategy
[Explain CPU→CUDA adaptation, 300-400 words]
- Key differences from CPU approach
- CUDA-specific patterns added
- Device parameter handling
- IR extraction modifications

#### 3.2.2 CUDA-Specific Patterns
[Describe CUDA patterns, 300-400 words]
- Thread indexing patterns
- Shared memory usage
- Atomic operations
- Block/grid configurations
- Parallel reduction patterns

#### 3.2.3 Production Statistics
```
Total Size: 200 MB
Total Samples: [N]
Kernel Types: [M]
Average Sample Size: [X KB]
Generation Time: [Y hours]
Generation Rate: [Z samples/min]
```

---

## 4. Dataset Structure and Format

### 4.1 Data Format
Each sample is a JSON file:
```json
{
  "kernel_name": "example_kernel",
  "python_source": "...",
  "ir_code": "...",
  "device": "cpu" | "cuda",
  "kernel_type": "arithmetic",
  "timestamp": "...",
  "metadata": { }
}
```

### 4.2 Directory Organization
```
production/
├── cpu_code/
│   ├── arithmetic_0001.json
│   ├── arithmetic_0002.json
│   ├── loop_0001.json
│   └── ...
└── cuda_code/
    ├── arithmetic_0001.json
    ├── thread_idx_0001.json
    └── ...
```

### 4.3 Data Quality Metrics
[Describe validation, 200-300 words]
- Validation checks performed
- Pass rates
- Common issues and handling
- Data integrity verification

---

## 5. Training Data Characteristics

### 5.1 CPU Dataset Analysis
[Analyze CPU data, 300-400 words]
- Code complexity distribution
- IR length statistics
- Token count distributions
- Kernel type balance

### 5.2 CUDA Dataset Analysis
[Analyze CUDA data, 300-400 words]
- Similar metrics as CPU
- CUDA-specific statistics
- Comparison to CPU dataset
- Coverage of GPU patterns

### 5.3 Diversity and Coverage
[Discuss dataset diversity, 200-300 words]
- Range of programming patterns
- Edge cases included
- Scalability patterns
- Real-world applicability

---

## 6. Potential Applications

### 6.1 LLM Training
[Describe use for LLM training, 300-400 words]
- Code generation models
- Python→IR translation
- JIT compiler assistance
- Performance optimization

### 6.2 Program Synthesis
[Describe program synthesis applications, 200-300 words]
- Automated kernel generation
- Performance-aware synthesis
- Domain-specific optimization

### 6.3 Compiler Research
[Describe compiler research uses, 200-300 words]
- IR optimization
- Code pattern recognition
- Cross-backend translation

---

## 7. Limitations and Future Work

### 7.1 Current Limitations
[List limitations, 200-300 words]
- Coverage gaps
- Generated vs real code
- CUDA testing constraints
- Kernel complexity limits

### 7.2 Future Enhancements
[Suggest improvements, 200-300 words]
- Additional kernel patterns
- Multi-file examples
- Performance annotations
- Optimization pass data

### 7.3 Dataset Extensions
[Suggest extensions, 200-300 words]
- Other backends (Metal, ROCm)
- More programming patterns
- Real-world kernel collection
- Error case examples

---

## 8. Conclusion
[Summarize, 200-300 words]
- What was achieved
- Dataset value proposition
- Next steps recommendation

---

## Appendices

### Appendix A: Sample Data Examples
[Include 3-5 example JSON files with Python and IR]

### Appendix B: Generation Scripts
[Brief description of key scripts used]

### Appendix C: Branch Analysis Summary
[Table of all branches evaluated]

### Appendix D: References
- NVIDIA Warp documentation
- Relevant papers on JIT compilation
- IR design references
```

#### Task 3.2: Write Report Content (~35k tokens)
Fill in each section with actual data and analysis:
1. Research JIT/IR concepts if needed
2. Analyze actual generated data for statistics
3. Include real examples from datasets
4. Add validation results
5. Write clear, technical prose suitable for chief scientist

**Quality criteria**:
- Technical accuracy
- Clear explanations
- Concrete data/statistics
- Professional tone
- Well-structured sections

**Done when**: Complete REPORT.md exists with all sections filled

#### Task 3.3: Push Report (~2k tokens)
```bash
git add report/
git commit -m "Production: Technical report for chief scientist"
git push origin HEAD
```

**Done when**: Report pushed to remote

---

## Validation Protocol

### CPU Dataset Validation
```bash
# Check total size
du -sh production/cpu_code

# Count samples
ls production/cpu_code/*.json | wc -l

# Validate JSON format
python production/scripts/validate_cpu.py

# Check kernel type distribution
python production/scripts/analyze_dataset.py --input cpu_code
```

### CUDA Dataset Validation
```bash
# Same checks as CPU
du -sh production/cuda_code
ls production/cuda_code/*.json | wc -l
python production/scripts/validate_cuda.py
python production/scripts/analyze_dataset.py --input cuda_code
```

### Report Validation
- [ ] All sections complete
- [ ] Statistics match actual data
- [ ] Examples are real samples
- [ ] No placeholders remain
- [ ] Markdown renders correctly
- [ ] Technical accuracy verified

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| P1: CPU | ~85k | Survey, reproduce, generate 200MB, push |
| P2: CUDA | ~95k | Survey, adapt, generate 200MB, push |
| P3: Report | ~40k | Write comprehensive technical report |
| **Total** | **~220k** | Full production + report |

Expected: 2-3 sessions depending on generation speeds

---

## Key Commands Reference

```bash
# Check data size
du -sh production/{cpu_code,cuda_code}

# Count samples
find production/ -name "*.json" | wc -l

# Monitor generation
watch -n 30 'du -sh production/cpu_code && ls production/cpu_code | wc -l'

# Validate sample
python -c "import json; print(json.load(open('production/cpu_code/arithmetic_0001.json')))"

# Commit and push
git add production/ report/
git commit -m "Production: [description]"
git push origin HEAD
```

---

## Anti-Patterns (Avoid)

- ❌ Generating more than 200MB per dataset type
- ❌ Starting Phase 2 before Phase 1 complete
- ❌ Writing report before datasets exist
- ❌ Using placeholder statistics in report
- ❌ Committing datasets if >100MB (use git-lfs or compress)
- ❌ Incomplete report sections
- ❌ Not pushing incrementally

---

## Success Criteria

Project complete when:
1. ✅ 200MB CPU dataset generated and pushed
2. ✅ 200MB CUDA dataset generated and pushed
3. ✅ Technical report complete with all sections
4. ✅ All validation checks pass
5. ✅ PRODUCTION_STATE.md shows "completed" status
6. ✅ Report includes real statistics and examples
7. ✅ Code and data pushed to remote branch
