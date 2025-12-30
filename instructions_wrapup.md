# JIT Code Synthesis - Branch Wrapup

## Objective
Wrap up YOUR branch's work on CPU code generation. Validate, reproduce, and document what was built.

---

## File Structure (create as needed in your branch)

```
jit/
├── instructions_wrapup.md   # This file (read-only)
├── WRAPUP_STATE.md          # Your progress tracker
├── README.md                # Documentation for your branch
├── code/                    # Your implementation
├── data/                    # Generated samples (keep ≤100 for git)
└── notes/                   # Technical findings
```

---

## State Management Protocol

### On Session Start
1. Read `WRAPUP_STATE.md` (create if missing)
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `WRAPUP_STATE.md` with exact next action
2. Commit working changes
3. Push to remote
4. Stop—do not start new tasks

### WRAPUP_STATE.md Template
```markdown
# Wrapup State
- **Phase**: P1/P2/P3
- **Task**: [current task]
- **Status**: in_progress | blocked | completed

## Next Action
[Specific next step]

## Session Log
- [session]: [what was done]
```

---

## Phases

### P1: Validate & Reproduce
**Goal**: Verify your branch's code works from a clean state

**Tasks**:
1. Check what milestone your branch reached (read STATE.md, git log, file structure)
2. Install dependencies: `pip install -U "jax[cpu]"` (and any in requirements.txt)
3. Run the main pipeline/scripts to verify they work:
   - If you have `code/synthesis/pipeline.py`: `python code/synthesis/pipeline.py --count 5`
   - If you have `code/extraction/ir_extractor.py`: run its `__main__` block
   - If you have tests: run them
4. Document what works and what doesn't in `WRAPUP_STATE.md`
5. Fix any minor issues (missing imports, path issues, etc.)

**Done when**: Core functionality runs without errors on at least one test case

### P2: Document
**Goal**: Write clear README for your branch

**README.md must include**:
```markdown
# JAX JIT Code Synthesis - [Branch Name]

## Progress Summary
- Milestone reached: M1/M2/M3/M4/M5
- Key deliverables: [list what was built]

## What Works
- [feature 1]: [brief description]
- [feature 2]: [brief description]

## Requirements
```bash
pip install -U "jax[cpu]"
```

## Quick Start
```bash
# [commands to run your code]
```

## File Structure
```
jit/
├── code/
│   ├── extraction/    # [what's here]
│   ├── synthesis/     # [what's here]
│   └── examples/      # [what's here]
├── data/              # [what's here]
└── notes/             # [what's here]
```

## Generated Data Format
```json
{
  "kernel_name": "...",
  "python_source": "...",
  "ir_code": "...",
  "device": "cpu"
}
```

## Known Issues / TODOs
- [any unfinished work]
- [any bugs found]
```

**Done when**: README.md accurately describes your branch's state

### P3: GPU Analysis
**Goal**: Analyze what's needed to run your code on a CUDA-backed JAX runtime and extract comparable IR

**Tasks**:
1. Check if your `ir_extractor.py` has a `device` parameter
2. If GPU available: try running with `device="cuda"` / default GPU placement and note results
3. If no GPU: document what a GPU run would change (install of CUDA-enabled `jaxlib`, device placement, supported ops)
4. Study differences in extracted IR across backends (e.g., StableHLO text differences, if any)
5. Document findings in `notes/gpu_analysis.md`

**notes/gpu_analysis.md template**:
```markdown
# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: [Yes/No]
- Tested with device="cuda": [pass/fail/no GPU]

## CPU vs GPU IR Differences
| Aspect | CPU (XLA:CPU / LLVM) | GPU (XLA:GPU / PTX) |
|--------|-----------------------|--------------------|
| [aspect] | [cpu behavior] | [gpu behavior] |

## Changes Needed for GPU
1. [change 1]
2. [change 2]

## New GPU-Specific Patterns to Add
- [ ] [pattern 1]
- [ ] [pattern 2]
```

**Done when**: `notes/gpu_analysis.md` has concrete findings

---

## Key Commands Reference

```bash
# Check your branch status
git status
git log --oneline -5

# Install JAX (CPU)
pip install -U "jax[cpu]"

# Common test commands
python code/extraction/ir_extractor.py
python code/synthesis/pipeline.py --count 5
python -m pytest code/ -v

# Commit and push
git add -A
git commit -m "wrapup: [brief description]"
git push origin HEAD
```

---

## Anti-Patterns (Avoid)

- ❌ Generating large datasets (≤100 samples in git)
- ❌ Major refactoring or new features
- ❌ Writing lengthy analysis documents
- ❌ Leaving code in broken state
- ❌ Skipping P1 validation

---

## Token Budget

| Phase | Budget | Activities |
|-------|--------|------------|
| P1 | ~50k | Understand branch, test, fix minor issues |
| P2 | ~30k | Write README |
| P3 | ~40k | GPU analysis |

Total estimate: 1-3 sessions per branch
