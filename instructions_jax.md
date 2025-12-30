# JIT Code Synthesis for LLM Training Data (JAX Edition)

## Objective
Use Google's JAX library to extract JIT intermediate representations (IR) and synthesize Python→IR paired data for LLM training.

---

## File Structure (create as needed)

```
jit/
├── instructions_jax.md       # This file (read-only reference)
├── STATE.md                   # CRITICAL: Current progress, next action, blockers
├── tasks/                     # Task lists for each milestone
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   └── ...
├── code/                      # All implementation code
│   ├── examples/              # JAX examples and kernels
│   ├── extraction/            # IR extraction utilities
│   └── synthesis/             # Data synthesis pipeline
├── data/                      # Generated training data samples
└── notes/                     # Technical findings (keep minimal)
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
**Goal**: Run JAX examples, understand JIT compilation and XLA flow
**Deliverables**:
- Working JAX installation (with jaxlib)
- 3+ examples showing @jit decorated functions
- `notes/jax_basics.md`: How JIT works, XLA IR access (max 50 lines)

### M2: IR Extraction Mechanism
**Goal**: Programmatically extract XLA HLO IR from JAX functions
**Deliverables**:
- `code/extraction/ir_extractor.py`: Function that takes JAX code → returns HLO IR
- 5+ test cases showing Python function → HLO IR pairs
- `notes/ir_format.md`: HLO IR structure documentation (max 30 lines)

### M3: Scientific Computing Deep Dive
**Goal**: Implement non-trivial numerical algorithms with JAX
**Deliverables**:
- `code/examples/poisson_solver.py`: Poisson equation solver using JAX
- `code/examples/test_poisson.py`: Validation tests (compare to analytical solutions)
- Tests pass for 2+ consecutive runs

### M4: Synthesis Pipeline
**Goal**: Automated Python→IR data generation
**Deliverables**:
- `code/synthesis/generator.py`: Generates varied JAX functions programmatically
- `code/synthesis/pipeline.py`: End-to-end: generate function → JIT compile → extract IR → save pair
- `data/samples/`: 100+ sample pairs for validation

### M5: Scale Up
**Goal**: Generate large-scale training dataset
**Deliverables**:
- `code/synthesis/batch_generator.py`: Parallel/batched generation
- `data/`: 10k+ Python→IR pairs
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

- JAX Documentation: https://jax.readthedocs.io/
- XLA Documentation: https://www.tensorflow.org/xla
- Key JAX concepts:
  - `jax.jit()` for JIT compilation
  - `jax.xla_computation()` for extracting HLO
  - `jax.make_jaxpr()` for extracting Jaxpr (intermediate IR)
  - `jax.grad()`, `jax.vmap()` for transformations
- JAX on GitHub: https://github.com/google/jax

---

## JAX-Specific Implementation Notes

### IR Extraction Methods

JAX provides multiple levels of IR:
1. **Jaxpr**: JAX's high-level IR (easiest to access)
   ```python
   from jax import make_jaxpr
   jaxpr = make_jaxpr(my_function)(args)
   ```

2. **HLO (High-Level Optimizer)**: XLA's IR (most useful for compilation)
   ```python
   from jax import xla_computation
   xla_comp = xla_computation(my_function)(args)
   hlo_text = xla_comp.as_hlo_text()
   ```

3. **StableHLO**: Modern standardized HLO format
   ```python
   hlo_module = xla_comp.as_hlo_module()
   ```

### Function Types to Cover

1. **Basic operations**: Element-wise math, reductions
2. **Array operations**: Broadcasting, slicing, concatenation
3. **Linear algebra**: matmul, dot, solve
4. **Control flow**: `jax.lax.cond()`, `jax.lax.scan()`, `jax.lax.while_loop()`
5. **Transformations**: `vmap()`, `grad()`, `jvp()`, `vjp()`
6. **Custom derivatives**: `@custom_jvp`, `@custom_vjp`

### Backend Considerations

- Default CPU backend: `jax.default_backend() == 'cpu'`
- GPU/TPU: Can extract IR without actual hardware
- Use `with jax.default_device(jax.devices('cpu')[0]):` for consistency

---

## Anti-Patterns (Avoid These)

- ❌ Writing summaries, READMEs, or reports
- ❌ Over-commenting code
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Reading entire large files (use targeted searches)
- ❌ Re-exploring already-documented findings

---

## Differences from Warp Version

| Aspect | Warp | JAX |
|--------|------|-----|
| **Language** | Python with @wp.kernel | Python with @jax.jit |
| **IR Format** | PTX/CUDA C++ | HLO/Jaxpr |
| **Primary Use** | GPU kernels for simulation | Array operations, ML |
| **Extraction API** | `wp.get_module(kernel).ptx` | `xla_computation()`, `make_jaxpr()` |
| **Type System** | wp.vec3, wp.array | jax.numpy arrays, pytrees |
| **Control Flow** | Standard Python (compiled) | lax.cond, lax.scan, etc. |
| **Differentiation** | Manual | Automatic (jax.grad) |

---

## Quick Start Example

```python
import jax
import jax.numpy as jnp
from jax import jit, xla_computation, make_jaxpr

# Define a simple function
@jit
def add_vectors(x, y):
    return x + y

# Extract Jaxpr
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
jaxpr = make_jaxpr(add_vectors)(x, y)
print("Jaxpr:", jaxpr)

# Extract HLO
xla_comp = xla_computation(add_vectors)(x, y)
hlo_text = xla_comp.as_hlo_text()
print("HLO:", hlo_text)

# This creates a Python→IR pair for training data
```
