# JIT Code Synthesis for LLM Training Data (JAX Edition)

## Objective
Use Google's `JAX` package to extract JIT intermediate representations (XLA HLO IR) and synthesize Python→IR paired data for LLM training.

---

## File Structure (create as needed)

```
jit-jax/
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

## State Management Protocol

### On Session Start
1. Read `STATE.md` first
2. Read the current milestone's task file (e.g., `tasks/m1_tasks.md`)
3. Resume from the documented next action

### On Session End (or ~20k tokens remaining)
1. Update `STATE.md` with:
   - Current milestone and task
   - Exact next action (be specific: file, function, line if applicable)
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states)
3. Stop—do not start new tasks

### STATE.md Template
```markdown
# Current State
- **Milestone**: M1/M2/M3/M4/M5
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next. Be specific enough that a new agent can execute immediately.]

## Blockers (if any)
[What's preventing progress, what was tried]

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Milestones

### M1: Environment Setup & JAX Basics
**Goal**: Run JAX examples, understand jax.jit compilation flow
**Deliverables**:
- Working JAX installation (with CPU or GPU backend)
- 3+ examples run successfully using jax.jit
- `notes/jax_basics.md`: How jax.jit compiles, where XLA HLO IR lives (max 50 lines)

### M2: IR Extraction Mechanism
**Goal**: Programmatically extract XLA HLO IR from JAX jitted functions
**Deliverables**:
- `code/extraction/ir_extractor.py`: Function that takes a JAX function → returns HLO IR
- 5+ test cases showing Python function → XLA HLO pairs
- `notes/ir_format.md`: HLO IR structure documentation (max 30 lines)

### M3: Advanced JAX Operations
**Goal**: Explore vmap, pmap, grad, and neural network patterns
**Deliverables**:
- `code/examples/advanced_ops.py`: Examples using vmap, grad, scan
- `code/examples/neural_net.py`: Simple neural network with JAX
- `code/examples/test_advanced.py`: Validation tests
- Tests pass for 2+ consecutive runs

### M4: Synthesis Pipeline
**Goal**: Automated Python→XLA HLO data generation
**Deliverables**:
- `code/synthesis/generator.py`: Generates varied Python functions programmatically
- `code/synthesis/pipeline.py`: End-to-end: generate function → jit compile → extract HLO → save pair
- `data/samples/`: 100+ sample pairs for validation

### M5: Scale Up
**Goal**: Generate large-scale training dataset
**Deliverables**:
- `code/synthesis/batch_generator.py`: Parallel/batched generation
- `data/`: 10k+ Python→XLA HLO pairs
- `notes/data_stats.md`: Dataset statistics (max 20 lines)

---

## Task Breakdown Rules

When starting a milestone, create `tasks/mX_tasks.md` with:
```markdown
# Milestone X Tasks

## Task 1: [name]
- [ ] Step 1.1: [specific action]
- [ ] Step 1.2: [specific action]
- **Done when**: [concrete success criterion]

## Task 2: [name]
...
```

Rules:
- Each step should be completable in <5k tokens
- "Done when" must be testable (not subjective)
- Mark completed steps with [x]

---

## Validation Protocol

Before marking any task complete:
1. Run the code/test twice
2. Results must match both times
3. No uncommitted debug code or prints
4. Code runs from clean state (no hidden dependencies)

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Orientation | ~5k | Read STATE.md, task file, understand context |
| Planning | ~10k | Break down next task, explore relevant code |
| Execution | ~150k | Implement, test, iterate |
| Handoff | ~10k | Update STATE.md, clean up, verify state |

If blocked for >20k tokens on same issue:
1. Document the blocker in STATE.md
2. Move to next task or milestone
3. Mark blocker for later resolution

---

## Key Resources

- JAX repo: https://github.com/google/jax
- JAX docs: https://jax.readthedocs.io/
- Key APIs to study:
  - `jax.jit` (JIT compilation)
  - `jax.make_jaxpr` (get JAX intermediate representation)
  - `jax.xla_computation` (get XLA HLO)
  - `jax.lowering` module (lowering to HLO)
  - `jax.vmap`, `jax.grad`, `jax.pmap` (transformations)

---

## JAX IR Extraction Methods

JAX provides several ways to inspect compiled representations:

1. **JAXPR (JAX Program Representation)**:
   ```python
   jax.make_jaxpr(fn)(*args)  # Returns JAXPR
   ```

2. **XLA HLO (High-Level Optimizer)**:
   ```python
   # Method 1: Using lowering API
   lowered = jax.jit(fn).lower(*args)
   hlo_text = lowered.as_text()
   
   # Method 2: Compiler output
   compiled = lowered.compile()
   compiled.as_text()
   ```

3. **StableHLO**:
   ```python
   lowered.as_text("stablehlo")
   ```

---

## Anti-Patterns (Avoid These)

- ❌ Writing summaries, READMEs, or reports
- ❌ Over-commenting code
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Reading entire large files (use targeted searches)
- ❌ Re-exploring already-documented findings
