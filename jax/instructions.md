# JIT Code Synthesis for LLM Training Data (JAX Edition)

## Objective
Use Google's `jax` package to extract JIT intermediate representations (XLA HLO) and synthesize Python→IR paired data for LLM training.

---

## File Structure

```
jax/
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

## Milestones

### M1: Environment Setup & JAX Basics
**Goal**: Run JAX JIT examples, understand compilation flow
**Deliverables**:
- Working JAX installation
- 3+ JIT examples run successfully
- `notes/jax_basics.md`: How JIT compiles, XLA HLO extraction (max 50 lines)

### M2: IR Extraction Mechanism
**Goal**: Programmatically extract XLA HLO from JAX JIT functions
**Deliverables**:
- `code/extraction/ir_extractor.py`: Function that takes a JAX function → returns XLA HLO
- 5+ test cases showing Python function → HLO pairs
- `notes/ir_format.md`: HLO structure documentation (max 30 lines)

### M3: Numerical Computing Deep Dive
**Goal**: Understand JAX for numerical computing, implement PDE solver
**Deliverables**:
- `code/examples/poisson_solver.py`: Working Poisson equation solver using JAX
- `code/examples/test_poisson.py`: Validation tests (compare to analytical solutions)
- Tests pass for 2+ consecutive runs

### M4: Synthesis Pipeline
**Goal**: Automated Python→XLA HLO data generation
**Deliverables**:
- `code/synthesis/generator.py`: Generates varied Python functions programmatically
- `code/synthesis/pipeline.py`: End-to-end: generate function → JIT compile → extract HLO → save pair
- `data/samples/`: 100+ sample pairs for validation

### M5: Scale Up
**Goal**: Generate large-scale training dataset
**Deliverables**:
- `code/synthesis/batch_generator.py`: Parallel/batched generation
- `data/`: 10k+ Python→HLO pairs
- `notes/data_stats.md`: Dataset statistics (max 20 lines)

---

## Key Resources

- JAX docs: https://jax.readthedocs.io/
- XLA: https://www.tensorflow.org/xla
- Key APIs for IR extraction:
  - `jax.jit` - JIT compilation
  - `jax.make_jaxpr` - Get JAX primitive representation
  - `jax.xla_computation` - Get XLA computation
  - `.as_hlo_text()` - Convert XLA computation to HLO text

---

## Validation Protocol

Before marking any task complete:
1. Run the code/test twice
2. Results must match both times
3. No uncommitted debug code or prints
4. Code runs from clean state (no hidden dependencies)
