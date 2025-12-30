# JIT Branch Merge

## Objective
Merge the best work from all 16 branches into a single production-ready branch. You are working on your current branch (check with `git branch --show-current`).

---

## File Structure

```
jit/
├── instructions_merge.md    # This file (read-only)
├── branch_progresses.md     # Branch analysis (read-only reference)
├── MERGE_STATE.md           # Current progress tracker
├── merge_notes/             # Findings from each branch
│   ├── 12c4_notes.md
│   ├── 9177_notes.md
│   └── ...
├── code/                    # Merged production code
├── data/                    # Sample data (keep ≤100 for git)
├── tests/                   # Merged test suite
└── README.md                # Final documentation
```

---

## State Management Protocol

### On Session Start
1. Identify and record your branch name:
   ```bash
   git branch --show-current
   ```
   Update `MERGE_STATE.md` with your branch name if not already recorded.
2. Read `MERGE_STATE.md` and `branch_progresses.md` for context
3. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `MERGE_STATE.md` with:
   - Current phase and branch
   - Exact next action
   - Key findings from this session
2. Commit all changes with descriptive message
3. Push to remote
4. Stop—do not start new work

### MERGE_STATE.md Template
```markdown
# Merge State
- **Phase**: P1/P2
- **Current Branch**: [branch suffix, e.g., 12c4]
- **Branches Completed**: [list]
- **Status**: in_progress | ready_for_next

## Next Action
[Specific next step with exact commands/files]

## Key Findings This Session
- [finding 1]
- [finding 2]

## Merge Decisions Made
- [decision 1]: [rationale]
- [decision 2]: [rationale]

## Session Log
- [session]: [what was done]
```

---

## Branch Processing Order

Process in this order (from `branch_progresses.md` ranking). If the ranking is stale, update it first so it reflects the current JAX-focused branches.

### Tier 1 - Production Ready (MUST process)
1. **12c4** - 10,727 pairs, full pipeline
2. **9177** - 10,320 pairs, complete project
3. **8631** - 10,101 pairs, synthesis pipeline

### Tier 2 - Complete Pipeline (process for features)
4. **82cf** - 775 pairs, README/reports
5. **aa30** - 628 pairs, QUICKSTART
6. **ff72** - 371 pairs, clean docs
7. **3576** - 239 pairs, test categories
8. **3a5b** - 100 pairs, batch generator

### Tier 3 - Partial (scan for unique code)
9. **25e7** - fast_generate scripts
10. **5d09** - analyze_dataset
11. **a4fd** - example kernels

### Tier 4 - M2-M3 Only (scan quickly)
12. **0fbe** - fixture_kernels
13. **7288** - example kernels
14. **3f34** - debug tools
15. **4b76** - basic extraction
16. **d623** - categorized test cases

---

## Phase 1: Explore & Document

**Goal**: Understand each branch, extract reusable components

### Per-Branch Workflow

For each branch (in order above):

#### Step 1: Quick Assessment (~2k tokens)
```bash
# Checkout branch files without switching
git show origin/cursor/following-instructions-md-{SUFFIX}:jit/code/synthesis/pipeline.py > /tmp/pipeline.py
git show origin/cursor/following-instructions-md-{SUFFIX}:jit/notes/data_stats.md 2>/dev/null
```

Check:
- Does pipeline.py exist and look complete?
- How many data samples generated?
- Any unique features not in previous branches?

#### Step 2: Test Run (~5k tokens)
```bash
# Create temp workspace
mkdir -p /tmp/test_{SUFFIX}
cd /tmp/test_{SUFFIX}

# Copy code from branch
git --git-dir=/path/to/jit/.git show origin/cursor/following-instructions-md-{SUFFIX}:jit/code/synthesis/pipeline.py > pipeline.py
git --git-dir=/path/to/jit/.git show origin/cursor/following-instructions-md-{SUFFIX}:jit/code/synthesis/generator.py > generator.py
git --git-dir=/path/to/jit/.git show origin/cursor/following-instructions-md-{SUFFIX}:jit/code/extraction/ir_extractor.py > ir_extractor.py

# Test
python pipeline.py --count 3
```

#### Step 3: Document Findings (~1k tokens)
Create `merge_notes/{SUFFIX}_notes.md`:
```markdown
# Branch {SUFFIX} Analysis

## Quick Stats
- Milestone: M?
- Data generated: N pairs
- Pipeline works: Yes/No

## Unique Features
- [feature]: [file:function]

## Code Quality
- Clean: Yes/No
- Tests: Yes/No
- Docs: Yes/No

## Recommended for Merge
- [ ] ir_extractor.py - [reason]
- [ ] generator.py - [reason]
- [ ] pipeline.py - [reason]
- [ ] [other files]

## Skip
- [file]: [reason to skip]
```

#### Step 4: Commit & Push
```bash
git add merge_notes/{SUFFIX}_notes.md
git commit -m "P1: Analyze branch {SUFFIX}"
git push origin HEAD
```

### Phase 1 Exit Criteria
- All 16 branches have notes in `merge_notes/`
- Clear list of which files to take from which branch
- MERGE_STATE.md updated with merge plan

---

## Phase 2: Merge & Build

**Goal**: Create unified codebase from best components

### Step 1: Initialize from Best Base (~10k tokens)

```bash
# You are already on the working branch (check: git branch --show-current)
# Pull code from 12c4 as base
git checkout origin/cursor/following-instructions-md-12c4 -- jit/code/
git checkout origin/cursor/following-instructions-md-12c4 -- jit/notes/

# Restructure if needed
mkdir -p code/extraction code/synthesis code/examples
mv jit/code/* code/
rm -rf jit/

git add -A
git commit -m "P2: Initialize from 12c4 base"
git push origin HEAD
```

### Step 2: Iterative Improvement

For each remaining branch (in order):

#### 2a. Baseline Test
```bash
# Run current pipeline
python code/synthesis/pipeline.py --count 5 --output /tmp/before
# Record: success/fail, time, output quality
```

#### 2b. Identify Improvements
Review `merge_notes/{SUFFIX}_notes.md`:
- What unique features does this branch have?
- What's better than current code?

#### 2c. Apply Improvements
Options:
- **Replace file**: `git show origin/cursor/following-instructions-md-{SUFFIX}:path/to/file > code/path/to/file`
- **Merge function**: Copy specific function into existing file
- **Add new file**: For unique utilities

#### 2d. Verify Improvement
```bash
# Run pipeline again
python code/synthesis/pipeline.py --count 5 --output /tmp/after
# Compare: same or better?
```

#### 2e. Commit & Push with Rationale
```bash
git add -A
git commit -m "P2: Merge {SUFFIX} - [what improved]"
git push origin HEAD
```

If no improvement:
```bash
git commit --allow-empty -m "P2: Skip {SUFFIX} - [why no improvement]"
git push origin HEAD
```

---

## Component Merge Reference

Based on `branch_progresses.md`:

| Component | Primary Source | Alternatives |
|-----------|---------------|--------------|
| `ir_extractor.py` | 12c4 | ff72, 0fbe |
| `generator.py` | 12c4 | ff72 (7 kernel types) |
| `pipeline.py` | 12c4 | 82cf (validation) |
| `batch_generator.py` | 12c4 | 9177, 8631 |
| Test cases | d623 | 3576 (by category) |
| README | 82cf | aa30 (QUICKSTART) |
| Data samples | 12c4 | 9177, 8631 |

### Kernel Types to Include
From generator.py across branches:
- arithmetic (basic ops)
- math (unary functions, transcendental)
- loop (for loops where supported, `lax.fori_loop`/`lax.scan`)
- conditional (`lax.cond` / `jnp.where`)
- vector (broadcasting / small-vector ops)
- matrix (matmul / linear algebra)
- combined (multi-pattern)

---

## Final Validation Checklist

Before marking Phase 2 complete:

```bash
# 1. Pipeline works
python code/synthesis/pipeline.py --count 10 --output data/test
# Should complete without errors

# 2. All kernel types generate
python -c "from code.synthesis.generator import GENERATORS; print(list(GENERATORS.keys()))"
# Should show 7 types

# 3. IR extraction works
python code/extraction/ir_extractor.py
# Should show example output

# 4. Tests pass (if any)
python -m pytest code/ -v 2>/dev/null || echo "No pytest tests"

# 5. Sample data valid
python -c "import json; d=json.load(open('data/test/arithmetic_*.json')); print('OK' if d.get('ir_code') else 'FAIL')"
```

---

## Git Commands Reference

```bash
# View file from branch without checkout
git show origin/cursor/following-instructions-md-{SUFFIX}:path/to/file

# Copy file from branch
git show origin/cursor/following-instructions-md-{SUFFIX}:path/to/file > local/path/file

# Checkout directory from branch
git checkout origin/cursor/following-instructions-md-{SUFFIX} -- path/to/dir/

# Compare files between branches
diff <(git show origin/cursor/following-instructions-md-12c4:jit/code/synthesis/generator.py) \
     <(git show origin/cursor/following-instructions-md-ff72:jit/code/synthesis/generator.py)

# List files in branch
git ls-tree -r --name-only origin/cursor/following-instructions-md-{SUFFIX}
```

---

## Token Budget

| Activity | Budget | Notes |
|----------|--------|-------|
| P1 per branch (Tier 1-2) | ~10k | Deep analysis |
| P1 per branch (Tier 3-4) | ~3k | Quick scan |
| P2 initialization | ~15k | Set up base |
| P2 per branch | ~8k | Merge + verify |
| Final validation | ~10k | Full test suite |

**Estimated total**: 200-300k tokens (5-8 sessions)

---

## Anti-Patterns (Avoid)

- ❌ Generating large datasets during merge (just verify small batches)
- ❌ Rewriting working code from scratch
- ❌ Merging without testing before/after
- ❌ Skipping Tier 1 branches
- ❌ Processing branches out of order
- ❌ Committing broken code
- ❌ Starting Phase 2 before Phase 1 complete

---

## Success Criteria

Phase 2 is complete when:
1. Single unified codebase in `code/`
2. All 7 kernel types in generator
3. Pipeline generates valid Python→IR pairs
4. README with quick start instructions
5. Sample data (50-100 pairs) committed
6. All merge decisions documented in MERGE_STATE.md
