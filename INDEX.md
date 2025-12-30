# Documentation Index

This index helps you navigate all documentation files in the JAX-based JIT code synthesis project.

---

## 🎯 For Getting Started

Start here if you're new to the project:

1. **[README.md](README.md)** - Project overview and quick start
2. **[JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md)** - Essential JAX commands and patterns
3. **[JAX_IR_EXAMPLES.md](JAX_IR_EXAMPLES.md)** - See actual Python→IR transformations

---

## 📋 For Implementation

Use these when building the system:

4. **[instructions_jax.md](instructions_jax.md)** - Complete 5-milestone implementation guide
5. **[STATE.md](STATE.md)** - Current progress and next actions
6. **Tasks files** (to be created):
   - `jit/tasks/m1_tasks.md` - M1 detailed task breakdown
   - `jit/tasks/m2_tasks.md` - M2 detailed task breakdown
   - `jit/tasks/m3_tasks.md` - M3 detailed task breakdown
   - `jit/tasks/m4_tasks.md` - M4 detailed task breakdown
   - `jit/tasks/m5_tasks.md` - M5 detailed task breakdown

---

## 🔄 For Understanding the Migration

Read these to understand why we chose JAX:

7. **[JAX_MIGRATION_SUMMARY.md](JAX_MIGRATION_SUMMARY.md)** - High-level overview of the migration
8. **[WARP_TO_JAX_MIGRATION.md](WARP_TO_JAX_MIGRATION.md)** - Detailed comparison: Warp vs JAX

---

## 📚 Original Instructions (Reference Only)

These are the original Warp-based instructions, kept for reference:

9. **[instructions.md](instructions.md)** - Original Warp-based instructions
10. **[instruction_cuda.md](instruction_cuda.md)** - CUDA adaptation instructions
11. **[instructions_merge.md](instructions_merge.md)** - Branch merge instructions
12. **[instructions_wrapup.md](instructions_wrapup.md)** - Wrap-up instructions

---

## 📖 Reading Order by Use Case

### Use Case 1: "I want to start implementing"
1. [README.md](README.md) - Understand the project
2. [instructions_jax.md](instructions_jax.md) - Follow the milestones
3. [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md) - Keep this open while coding
4. [STATE.md](STATE.md) - Track your progress here

### Use Case 2: "I want to understand JAX IR extraction"
1. [JAX_IR_EXAMPLES.md](JAX_IR_EXAMPLES.md) - See 10 concrete examples
2. [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md) - Learn the API
3. [instructions_jax.md](instructions_jax.md) → M2 section - Build the extractor

### Use Case 3: "I'm coming from the Warp-based approach"
1. [WARP_TO_JAX_MIGRATION.md](WARP_TO_JAX_MIGRATION.md) - See the mapping
2. [JAX_MIGRATION_SUMMARY.md](JAX_MIGRATION_SUMMARY.md) - Understand rationale
3. [instructions_jax.md](instructions_jax.md) - Follow adapted instructions

### Use Case 4: "I want to see what changes from Warp"
1. Compare [instructions.md](instructions.md) vs [instructions_jax.md](instructions_jax.md)
2. Read [WARP_TO_JAX_MIGRATION.md](WARP_TO_JAX_MIGRATION.md) for side-by-side comparison

---

## 📊 Document Relationships

```
README.md (central hub)
    ├── instructions_jax.md (implementation guide)
    │   ├── STATE.md (progress tracking)
    │   └── tasks/*.md (detailed breakdowns)
    │
    ├── JAX_QUICK_REFERENCE.md (API reference)
    │   └── JAX_IR_EXAMPLES.md (concrete examples)
    │
    └── WARP_TO_JAX_MIGRATION.md (migration guide)
        ├── JAX_MIGRATION_SUMMARY.md (overview)
        └── instructions.md (original Warp version)
```

---

## 📝 File Purposes at a Glance

| File | Type | Purpose | When to Read |
|------|------|---------|--------------|
| README.md | Overview | Project introduction | First time |
| instructions_jax.md | Guide | Complete implementation plan | Daily reference |
| STATE.md | Tracker | Current progress | Every session start/end |
| JAX_QUICK_REFERENCE.md | Reference | Commands and patterns | While coding |
| JAX_IR_EXAMPLES.md | Examples | Actual IR outputs | When writing extractors |
| WARP_TO_JAX_MIGRATION.md | Comparison | Warp vs JAX concepts | Understanding differences |
| JAX_MIGRATION_SUMMARY.md | Summary | Why JAX? | Understanding rationale |
| instructions.md | Archive | Original Warp guide | Reference only |

---

## 🎓 Learning Path

### Day 1: Understanding
1. Read [README.md](README.md) (15 min)
2. Read [JAX_MIGRATION_SUMMARY.md](JAX_MIGRATION_SUMMARY.md) (10 min)
3. Skim [JAX_IR_EXAMPLES.md](JAX_IR_EXAMPLES.md) (20 min)
4. **Action**: Install JAX, run first example

### Day 2: M1 - Environment Setup
1. Read [instructions_jax.md](instructions_jax.md) → M1 section (15 min)
2. Create `jit/tasks/m1_tasks.md` (10 min)
3. Keep [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md) open
4. **Action**: Complete M1 tasks, update STATE.md

### Day 3-4: M2 - IR Extraction
1. Read [instructions_jax.md](instructions_jax.md) → M2 section (15 min)
2. Study [JAX_IR_EXAMPLES.md](JAX_IR_EXAMPLES.md) thoroughly (30 min)
3. Create `jit/tasks/m2_tasks.md` (10 min)
4. **Action**: Build ir_extractor.py, test on 5+ functions

### Day 5-7: M3 - Transformations
1. Read [instructions_jax.md](instructions_jax.md) → M3 section (15 min)
2. Reference [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md) → Transformations
3. Create `jit/tasks/m3_tasks.md` (10 min)
4. **Action**: Extract IR from grad, vmap, scan, cond

### Day 8-10: M4 - Synthesis Pipeline
1. Read [instructions_jax.md](instructions_jax.md) → M4 section (15 min)
2. Create `jit/tasks/m4_tasks.md` (15 min)
3. **Action**: Build generator.py and pipeline.py

### Day 11-14: M5 - Scale Up
1. Read [instructions_jax.md](instructions_jax.md) → M5 section (15 min)
2. Create `jit/tasks/m5_tasks.md` (10 min)
3. **Action**: Generate 10k+ samples, validate quality

---

## 🔍 Quick Search

Need to find something specific? Use these keywords:

- **Installation**: README.md, JAX_QUICK_REFERENCE.md
- **Jaxpr extraction**: JAX_QUICK_REFERENCE.md, JAX_IR_EXAMPLES.md
- **HLO extraction**: JAX_QUICK_REFERENCE.md, JAX_IR_EXAMPLES.md
- **Gradients**: JAX_QUICK_REFERENCE.md → Section 2, JAX_IR_EXAMPLES.md → Example 4
- **Vectorization**: JAX_QUICK_REFERENCE.md → Section 3, JAX_IR_EXAMPLES.md → Example 5
- **Conditionals**: JAX_QUICK_REFERENCE.md → Section 4, JAX_IR_EXAMPLES.md → Example 6
- **Loops**: JAX_QUICK_REFERENCE.md → Section 5, JAX_IR_EXAMPLES.md → Example 7
- **Common errors**: JAX_QUICK_REFERENCE.md → Common Gotchas
- **Milestones**: instructions_jax.md → Milestones section
- **State management**: instructions_jax.md → State Management Protocol
- **Warp comparison**: WARP_TO_JAX_MIGRATION.md
- **Dataset format**: README.md → Expected Dataset section
- **Function categories**: README.md → Function Categories section

---

## 📈 Progress Tracking

Track your progress through the milestones:

- [ ] **M1**: Environment Setup & JAX Basics
  - [ ] Install JAX
  - [ ] Run 3+ examples
  - [ ] Extract basic IR
  - [ ] Document findings

- [ ] **M2**: IR Extraction Mechanism
  - [ ] Build ir_extractor.py
  - [ ] Support Jaxpr and HLO
  - [ ] Test on 5+ functions
  - [ ] Document IR format

- [ ] **M3**: Transformations Deep Dive
  - [ ] Extract from jax.grad
  - [ ] Extract from jax.vmap
  - [ ] Extract from jax.lax.scan
  - [ ] Extract from jax.lax.cond

- [ ] **M4**: Synthesis Pipeline
  - [ ] Build generator.py (10 categories)
  - [ ] Build pipeline.py
  - [ ] Generate 100+ samples
  - [ ] Validate quality

- [ ] **M5**: Scale Up
  - [ ] Build batch_generator.py
  - [ ] Generate 10k+ samples
  - [ ] Document statistics
  - [ ] Final validation

Update this checklist in [STATE.md](STATE.md)!

---

## 🤔 FAQ

**Q: Where do I start?**  
A: Read [README.md](README.md), then follow [instructions_jax.md](instructions_jax.md) starting with M1.

**Q: How do I extract IR from a JAX function?**  
A: See [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md) → "Basic IR Extraction" section.

**Q: What does the IR look like?**  
A: Check [JAX_IR_EXAMPLES.md](JAX_IR_EXAMPLES.md) for 10 concrete examples.

**Q: Why JAX instead of Warp?**  
A: Read [JAX_MIGRATION_SUMMARY.md](JAX_MIGRATION_SUMMARY.md) for the rationale.

**Q: How is this different from the original Warp instructions?**  
A: See [WARP_TO_JAX_MIGRATION.md](WARP_TO_JAX_MIGRATION.md) for side-by-side comparison.

**Q: Where do I track my progress?**  
A: Update [STATE.md](STATE.md) at the start and end of each session.

**Q: I'm stuck on a JAX issue, where should I look?**  
A: Check [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md) → "Common Gotchas" section first.

---

## 🔗 External Resources

- **JAX Official Docs**: https://jax.readthedocs.io/
- **JAX GitHub**: https://github.com/google/jax
- **JAX Examples**: https://github.com/google/jax/tree/main/examples
- **Autodiff Cookbook**: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
- **Common Gotchas**: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

---

**Last updated**: 2025-12-30  
**Branch**: `cursor/jax-migration-from-warp-f9b0`
