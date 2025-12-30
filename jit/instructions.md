# JIT Code Synthesis for LLM Training Data (JAX)

## Objective
Use JAX's JIT compilation to extract intermediate representations (jaxpr/XLA HLO) and synthesize Python→IR paired data for LLM training.

---

## File Structure

```
jit/
├── instructions.md          # This file (read-only reference)
├── STATE.md                  # CRITICAL: Current progress, next action, blockers
├── tasks/                    # Task lists for each milestone
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   └── ...
├── code/                     # All implementation code
│   ├── examples/             # Reproduced/new examples
│   ├── extraction/           # IR extraction utilities
│   └── synthesis/            # Data synthesis pipeline
├── data/                     # Generated training data samples
└── notes/                    # Technical findings (keep minimal)
```

---

## JAX IR Types

1. **Jaxpr** - JAX's internal intermediate representation
   - Obtained via `jax.make_jaxpr(fn)(*args)`
   - Shows primitive operations and data flow

2. **XLA HLO** - Lower-level IR for XLA compiler
   - Obtained via `jax.jit(fn).lower(*args).as_text()`
   - Hardware-agnostic optimization IR

3. **Optimized HLO** - After XLA optimizations
   - Via `.compile().as_text()`

---

## Milestones

### M1: Environment Setup & JAX Basics
**Goal**: Run JAX examples, understand JIT compilation flow
**Deliverables**:
- Working JAX installation
- 3+ examples demonstrating JIT compilation
- `notes/jax_basics.md`: How jit works, jaxpr structure (max 50 lines)

### M2: IR Extraction Mechanism
**Goal**: Programmatically extract jaxpr and HLO from JAX functions
**Deliverables**:
- `code/extraction/ir_extractor.py`: Function that takes a JAX function → returns IR (jaxpr + HLO)
- 5+ test cases showing Python function → IR pairs
- `notes/ir_format.md`: IR structure documentation (max 30 lines)

### M3: Advanced JAX Patterns
**Goal**: Handle vmap, pmap, scan, and custom primitives
**Deliverables**:
- `code/examples/advanced_patterns.py`: Examples with vmap, scan, while_loop
- `code/examples/test_patterns.py`: Validation tests
- Tests pass for 2+ consecutive runs

### M4: Synthesis Pipeline
**Goal**: Automated Python→IR data generation
**Deliverables**:
- `code/synthesis/generator.py`: Generates varied Python functions programmatically
- `code/synthesis/pipeline.py`: End-to-end: generate function → jit → extract IR → save pair
- `data/samples/`: 100+ sample pairs for validation

### M5: Scale Up
**Goal**: Generate large-scale training dataset
**Deliverables**:
- `code/synthesis/batch_generator.py`: Parallel/batched generation
- `data/`: 10k+ Python→IR pairs
- `notes/data_stats.md`: Dataset statistics (max 20 lines)

---

## Key Resources

- JAX docs: https://jax.readthedocs.io/
- Key APIs:
  - `jax.make_jaxpr()` - Get jaxpr representation
  - `jax.jit(fn).lower(*args)` - Lower to StableHLO
  - `jax.jit(fn).lower(*args).as_text()` - Get HLO text
  - `jax.jit(fn).lower(*args).compile()` - Compile to executable

---

## Validation Protocol

Before marking any task complete:
1. Run the code/test twice
2. Results must match both times
3. No uncommitted debug code or prints
4. Code runs from clean state (no hidden dependencies)

---

## Anti-Patterns (Avoid These)

- ❌ Writing summaries, READMEs, or reports
- ❌ Over-commenting code
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Reading entire large files (use targeted searches)
- ❌ Re-exploring already-documented findings
