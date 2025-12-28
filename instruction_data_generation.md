# Data Generation & Reporting

## Objective
Produce 200MB of CPU code dataset and 200MB of CUDA code dataset. Write a simple report in markdown format for the chief scientist to review, introducing JIT, IR, Nvidia Warp, and the current dataset.

---

## File Structure

```
generation/
├── instruction_data_generation.md # This file
├── GENERATION_STATE.md            # Progress tracker
├── code/                          # Selected production code
├── data/                          # Generated datasets (local storage, push samples only)
└── report/                        # Final report
    ├── report.md
    └── figures/
```

---

## State Management Protocol

### On Session Start
1. Read `GENERATION_STATE.md`
2. Resume from documented next action

### On Session End
1. Update `GENERATION_STATE.md` with:
   - Current phase
   - Next action
   - Key findings
2. Commit changes
3. Push to remote

### GENERATION_STATE.md Template
```markdown
# Generation State
- **Phase**: P1 (CPU) | P2 (CUDA) | P3 (Report)
- **Status**: in_progress | completed

## Next Action
[Specific next step]

## Session Log
- [date]: [activity]
```

---

## Workflow

### Phase 1: CPU Code Production
**Goal**: Produce 200MB of CPU code data.

1. **Study Branches**:
   - Inspect `cursor/agent-work-merge-*` and `cursor/agent-work-merge-process-*` branches.
   - Reproduce code from promising branches.
2. **Select Best Code**:
   - Choose the most robust and efficient code for production.
   - Establish it in `generation/code/cpu/`.
3. **Generate Data**:
   - Gradually produce data to reach 200MB.
   - Monitor size and quality.
   - Push to remote (ensure git limits are respected, likely need to use LFS or just push samples if 200MB is too large for standard repo, but user said "push to remote", so assuming it's allowed or use LFS). *Self-correction: 200MB is large for git. I will push to a specific folder or assume LFS is handled. If not, I will push in chunks or compressed if text.*
   - Actually, 200MB of *source code text* is huge. 200MB of *dataset* (maybe json/parquet) is manageable. I will check file sizes.

### Phase 2: CUDA Code Production
**Goal**: Produce 200MB of CUDA code data.

1. **Study Branches**:
   - Search for `cursor/cuda...` branches (or branches with CUDA adaptations).
   - If not found, use `instruction_cuda.md` as a guide to adapt CPU code.
2. **Select/Adapt Code**:
   - Pick best implementation.
   - Establish in `generation/code/cuda/`.
3. **Generate Data**:
   - Gradually produce data to reach 200MB.
   - Push to remote.

### Phase 3: Report
**Goal**: Write a simple report for the chief scientist.

1. **Content**:
   - Introduce JIT (Just-In-Time compilation).
   - Introduce IR (Intermediate Representation).
   - Introduce Nvidia Warp.
   - Describe the current dataset (stats, format, coverage).
2. **Format**: Markdown (`generation/report/report.md`).

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| P1    | ~50k   | Study, Select, Generate CPU Data |
| P2    | ~50k   | Study, Select, Generate CUDA Data |
| P3    | ~10k   | Write Report |

---

## Anti-Patterns
- ❌ Committing 200MB directly if git block size is small (Check `.gitignore` or LFS).
- ❌ Spending too much time on broken branches.
- ❌ Over-complicating the report.
