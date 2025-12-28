# Dataset Production & Technical Report

## Objective
Produce large-scale training datasets (200MB each) for CPU and CUDA code, and deliver a technical report for the chief scientist on JIT compilation, intermediate representations, NVIDIA Warp, and the generated datasets.

---

## File Structure

```
/workspace/
├── instructions_dataset_production.md  # This file (read-only)
├── DATASET_STATE.md                    # Current progress tracker
├── cpu_data/                           # CPU code dataset (target: 200MB)
├── cuda_data/                          # CUDA code dataset (target: 200MB)
├── production_code/                    # Selected production code
│   ├── cpu/                            # CPU pipeline code
│   └── cuda/                           # CUDA pipeline code
├── evaluation_notes/                   # Branch evaluation findings
│   ├── cpu_branch_eval.md
│   └── cuda_branch_eval.md
└── TECHNICAL_REPORT.md                 # Final report for chief scientist
```

---

## State Management Protocol

### On Session Start
1. Read `DATASET_STATE.md` first
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `DATASET_STATE.md` with:
   - Current phase and progress
   - Data size generated so far (CPU and CUDA)
   - Exact next action
   - Any blockers
2. Commit all changes with descriptive message
3. Push to remote
4. Stop—do not start new work

### DATASET_STATE.md Template
```markdown
# Dataset Production State
- **Phase**: P1/P2/P3
- **Status**: in_progress | ready_for_next
- **CPU Data Size**: X MB / 200 MB
- **CUDA Data Size**: X MB / 200 MB

## Next Action
[Specific next step with exact commands/files]

## Progress This Session
- [accomplishment 1]
- [accomplishment 2]

## Blockers (if any)
[What's preventing progress]

## Session Log
- [session]: [what was done]
```

---

## Phase 1: CPU Code Production (Target: 200MB)

### Milestone P1.1: Branch Evaluation & Selection (~20k tokens)

**Branches to evaluate** (from `branch_progresses.md`):
- `cursor/agent-work-merge-process-0038`
- `cursor/agent-work-merge-process-0499`
- `cursor/agent-work-merge-process-4dce`
- `cursor/agent-work-merge-process-6964`
- `cursor/agent-work-merge-process-96fd`
- `cursor/agent-work-merge-process-ad19`
- `cursor/agent-work-merge-process-bc08`

**Per-Branch Workflow:**

1. **Quick Assessment**
   ```bash
   # List files in branch
   git ls-tree -r --name-only origin/[BRANCH_NAME]
   
   # Check for production code
   git show origin/[BRANCH_NAME]:jit/code/synthesis/batch_generator.py 2>/dev/null
   git show origin/[BRANCH_NAME]:code/synthesis/batch_generator.py 2>/dev/null
   ```

2. **Test Run**
   ```bash
   # Create temp workspace
   mkdir -p /tmp/test_[BRANCH_SUFFIX]
   cd /tmp/test_[BRANCH_SUFFIX]
   
   # Copy production code and test with small batch
   # Run: python batch_generator.py --count 10 --output test_output
   ```

3. **Evaluation Criteria**
   - Pipeline runs without errors
   - Generates valid Python→IR pairs
   - Can scale to large batches
   - Code quality and maintainability
   - Generation speed (pairs/minute)

4. **Document Findings**
   Create `evaluation_notes/cpu_branch_eval.md`:
   ```markdown
   # CPU Branch Evaluation
   
   ## Branches Tested
   
   ### [BRANCH_NAME]
   - **Pipeline Status**: Works/Broken
   - **Generation Rate**: X pairs/minute
   - **Code Quality**: Good/Fair/Poor
   - **Unique Features**: [list]
   - **Issues**: [list]
   
   ## Selected Branch
   **[BRANCH_NAME]** - [rationale]
   
   ## Selection Criteria Met
   - [ ] Runs without errors
   - [ ] Generates valid data
   - [ ] Can scale to 200MB
   - [ ] Well-structured code
   ```

### Milestone P1.2: Production Setup (~10k tokens)

1. **Copy Selected Code**
   ```bash
   # Copy from selected branch to production_code/cpu/
   git show origin/[SELECTED_BRANCH]:path/to/batch_generator.py > production_code/cpu/batch_generator.py
   git show origin/[SELECTED_BRANCH]:path/to/generator.py > production_code/cpu/generator.py
   git show origin/[SELECTED_BRANCH]:path/to/ir_extractor.py > production_code/cpu/ir_extractor.py
   ```

2. **Validation Test**
   ```bash
   cd production_code/cpu
   python batch_generator.py --count 100 --output /tmp/validation
   # Verify: 100 valid JSON files generated
   ```

3. **Commit Setup**
   ```bash
   git add production_code/cpu/ evaluation_notes/cpu_branch_eval.md
   git commit -m "P1: Setup CPU production code from [BRANCH]"
   git push origin HEAD
   ```

### Milestone P1.3: Incremental Data Generation (~100k+ tokens)

**Strategy**: Generate data in batches, push incrementally

1. **Batch Generation Loop**
   ```bash
   # Calculate: 200MB ÷ avg_file_size = total_pairs needed
   # Example: 200MB ÷ 2KB = ~100,000 pairs
   
   # Generate in batches of 10,000
   for i in {1..10}; do
       python production_code/cpu/batch_generator.py \
           --count 10000 \
           --output cpu_data/batch_$i \
           --start-index $((i * 10000))
       
       # Check size
       du -sh cpu_data/
       
       # Push every 2-3 batches (40-60MB)
       if [ $((i % 3)) -eq 0 ]; then
           git add cpu_data/
           git commit -m "P1: CPU data batch $i ($(du -sh cpu_data/ | cut -f1))"
           git push origin HEAD
       fi
   done
   ```

2. **Monitor Progress**
   - Update `DATASET_STATE.md` every 20-30MB
   - Check data validity periodically
   - Track generation rate and ETA

3. **Exit Criteria**
   - `cpu_data/` size ≥ 200MB
   - All files are valid JSON
   - Data successfully pushed to remote

---

## Phase 2: CUDA Code Production (Target: 200MB)

### Milestone P2.1: Branch Evaluation & Selection (~20k tokens)

**Branches to evaluate**:
- Look for branches with "cuda" in commit messages/files
- Check recent work branches for CUDA adaptations

**Same workflow as P1.1**, but looking for:
- CUDA kernel generation code
- GPU-specific IR extraction
- CUDA backend compatibility

Document in `evaluation_notes/cuda_branch_eval.md`

### Milestone P2.2: Production Setup (~10k tokens)

Same process as P1.2, but for CUDA code:
- Copy to `production_code/cuda/`
- Validate CUDA-specific features
- Ensure generation works (even without GPU runtime)

### Milestone P2.3: Incremental Data Generation (~100k+ tokens)

Same strategy as P1.3:
- Generate 200MB of CUDA code data
- Push incrementally every 40-60MB
- Target: `cuda_data/` ≥ 200MB

---

## Phase 3: Technical Report

### Milestone P3.1: Report Writing (~30k tokens)

Create `TECHNICAL_REPORT.md` with following structure:

```markdown
# Technical Report: JIT Code Dataset for LLM Training

**Prepared for**: Chief Scientist
**Date**: [DATE]
**Author**: Agent [ID]

---

## Executive Summary
[2-3 paragraphs: What was done, key results, dataset statistics]

---

## 1. Just-In-Time (JIT) Compilation

### 1.1 Overview
- Definition and purpose of JIT compilation
- Advantages over ahead-of-time compilation
- Role in modern high-performance computing

### 1.2 JIT in the Context of This Work
- How JIT enables IR extraction
- Python→IR transformation pipeline
- Benefits for LLM training data

---

## 2. Intermediate Representations (IR)

### 2.1 What is IR?
- Definition and purpose
- Position in compilation pipeline
- Common IR formats (LLVM, PTX, etc.)

### 2.2 IR Extraction Process
- Warp's JIT compilation flow
- IR capture mechanism
- Example: Python kernel → IR transformation

### 2.3 IR Characteristics in Dataset
- IR syntax and structure
- Type system representation
- Control flow encoding

---

## 3. NVIDIA Warp

### 3.1 Overview
- What is NVIDIA Warp?
- Key features and use cases
- Why it's ideal for this project

### 3.2 Warp's JIT Architecture
- Kernel compilation process
- Backend support (CPU/CUDA)
- IR generation mechanism

### 3.3 Warp Features Utilized
- Type system (vectors, matrices)
- Kernel decorators and compilation
- Backend selection (CPU vs CUDA)

---

## 4. Dataset Description

### 4.1 CPU Code Dataset
- **Size**: [X] MB ([N] samples)
- **Format**: JSON (Python source + IR pairs)
- **Kernel Types**: [list types: arithmetic, loops, conditionals, etc.]
- **Coverage**: [describe variety]

### 4.2 CUDA Code Dataset
- **Size**: [X] MB ([N] samples)
- **Format**: JSON (Python source + CUDA IR pairs)
- **Kernel Types**: [list types]
- **GPU-Specific Features**: [describe CUDA-specific patterns]

### 4.3 Data Quality
- Validation methodology
- Error rate and handling
- Deduplication strategy

### 4.4 Sample Examples
[Include 2-3 representative examples showing Python→IR pairs]

---

## 5. Production Pipeline

### 5.1 Architecture
- Component overview diagram (text-based)
- Data flow: generation → extraction → validation → storage

### 5.2 Key Components
- `generator.py`: Kernel generation logic
- `ir_extractor.py`: IR extraction mechanism
- `batch_generator.py`: Scalable batch processing
- `pipeline.py`: End-to-end orchestration

### 5.3 Performance Metrics
- Generation rate (samples/minute)
- Resource usage (CPU/memory)
- Scalability characteristics

---

## 6. Dataset Statistics

### 6.1 Overall Statistics
| Metric | CPU Dataset | CUDA Dataset | Total |
|--------|-------------|--------------|-------|
| Size (MB) | X | Y | X+Y |
| Sample Count | N | M | N+M |
| Avg Sample Size (KB) | A | B | (A+B)/2 |
| Generation Time | T1 | T2 | T1+T2 |

### 6.2 Kernel Type Distribution
[Table or list showing breakdown by kernel type]

### 6.3 Complexity Distribution
[Analysis of code complexity metrics if available]

---

## 7. Use Cases for LLM Training

### 7.1 Potential Applications
- Code translation models (Python → IR)
- Optimization suggestion models
- Code generation with hardware awareness
- Performance prediction models

### 7.2 Training Recommendations
- Suggested model architectures
- Training strategies
- Evaluation metrics

---

## 8. Future Work

### 8.1 Dataset Improvements
- Additional kernel types to cover
- More complex patterns
- Domain-specific extensions (FEM, physics simulations)

### 8.2 Pipeline Enhancements
- Automated validation improvements
- Quality metrics and filtering
- Metadata enrichment

---

## 9. Conclusion
[Summary of achievements and next steps]

---

## Appendices

### Appendix A: Repository Structure
[Overview of code organization]

### Appendix B: Running the Pipeline
[Quick start guide]

### Appendix C: Data Format Specification
[Detailed JSON schema]

### Appendix D: Branch Analysis Summary
[Key findings from branch evaluations]
```

### Milestone P3.2: Report Refinement (~10k tokens)

1. **Add Concrete Examples**
   - Extract 3-5 representative Python→IR pairs from datasets
   - Include inline in report for illustration

2. **Generate Statistics**
   ```bash
   # Count samples
   find cpu_data/ -name "*.json" | wc -l
   find cuda_data/ -name "*.json" | wc -l
   
   # Calculate sizes
   du -sh cpu_data/ cuda_data/
   
   # Sample analysis (if time permits)
   python -c "
   import json, glob, statistics
   sizes = []
   for f in glob.glob('cpu_data/**/*.json', recursive=True)[:1000]:
       sizes.append(len(open(f).read()))
   print(f'Avg: {statistics.mean(sizes):.0f} bytes')
   print(f'Median: {statistics.median(sizes):.0f} bytes')
   "
   ```

3. **Final Review**
   - Technical accuracy
   - Clarity for chief scientist audience
   - All sections complete
   - Examples are illustrative

4. **Commit & Push**
   ```bash
   git add TECHNICAL_REPORT.md
   git commit -m "P3: Complete technical report"
   git push origin HEAD
   ```

---

## Validation Protocol

### Data Validity Checks
```bash
# Check JSON validity (sample)
find cpu_data/ -name "*.json" -print0 | head -z -100 | xargs -0 -n1 python -m json.tool > /dev/null

# Check required fields
python -c "
import json, glob
for f in glob.glob('cpu_data/**/*.json', recursive=True)[:100]:
    d = json.load(open(f))
    assert 'python_code' in d, f'{f}: missing python_code'
    assert 'ir_code' in d, f'{f}: missing ir_code'
    assert len(d['python_code']) > 0, f'{f}: empty python_code'
    assert len(d['ir_code']) > 0, f'{f}: empty ir_code'
print('✓ All checks passed')
"
```

### Size Verification
```bash
# Must meet targets
SIZE_CPU=$(du -sm cpu_data/ | cut -f1)
SIZE_CUDA=$(du -sm cuda_data/ | cut -f1)

[ $SIZE_CPU -ge 200 ] && echo "✓ CPU target met ($SIZE_CPU MB)" || echo "✗ CPU target not met ($SIZE_CPU MB)"
[ $SIZE_CUDA -ge 200 ] && echo "✓ CUDA target met ($SIZE_CUDA MB)" || echo "✗ CUDA target not met ($SIZE_CUDA MB)"
```

---

## Push Strategy

### Incremental Pushes (Critical for Large Data)
- Push every 40-60MB of new data
- Avoid single massive commits (Git performance)
- Use descriptive commit messages with size info

### Example Push Schedule
```bash
# After every significant batch
git add cpu_data/batch_1/ cpu_data/batch_2/ cpu_data/batch_3/
git commit -m "P1: CPU data batches 1-3 (62MB total)"
git push origin HEAD

# Continue until target met
```

---

## Success Criteria

Project is complete when:

1. **CPU Dataset**
   - ✓ Size ≥ 200MB
   - ✓ All files valid JSON
   - ✓ Pushed to remote
   - ✓ Production code documented

2. **CUDA Dataset**
   - ✓ Size ≥ 200MB
   - ✓ All files valid JSON
   - ✓ Pushed to remote
   - ✓ Production code documented

3. **Technical Report**
   - ✓ All sections complete
   - ✓ Includes concrete examples
   - ✓ Dataset statistics accurate
   - ✓ Suitable for chief scientist audience
   - ✓ Pushed to remote

4. **Documentation**
   - ✓ `DATASET_STATE.md` reflects completion
   - ✓ Evaluation notes for both CPU and CUDA
   - ✓ Production code is runnable

---

## Anti-Patterns (Avoid)

- ❌ Generating all 200MB without pushing (Git will struggle)
- ❌ Not validating data before pushing
- ❌ Starting Phase 2 before Phase 1 complete
- ❌ Writing report before datasets complete
- ❌ Overly technical report (remember audience: chief scientist, not engineer)
- ❌ Committing broken production code
- ❌ Not updating DATASET_STATE.md regularly

---

## Token Budget Estimate

| Phase | Estimated Tokens | Notes |
|-------|------------------|-------|
| P1.1: CPU Branch Eval | 20k | Test 7 branches |
| P1.2: CPU Setup | 10k | Copy and validate |
| P1.3: CPU Generation | 100-150k | Iterative, monitor progress |
| P2.1: CUDA Branch Eval | 20k | Similar to P1.1 |
| P2.2: CUDA Setup | 10k | Similar to P1.2 |
| P2.3: CUDA Generation | 100-150k | Similar to P1.3 |
| P3: Report | 40k | Write and refine |
| **Total** | **300-400k** | 8-10 sessions |

---

## Notes

- This is a production task: focus on deliverables, not exploration
- Data quality matters: validate regularly
- Push frequently: don't lose work
- The report is for a technical executive: balance depth with clarity
- Time estimate: This will take multiple context windows—that's expected
