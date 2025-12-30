# JIT Code Synthesis for LLM Training Data (JAX Edition)

## Objective
Use Google's `jax` package to extract JIT intermediate representations (IR) and synthesize Python→IR paired data for LLM training.

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
│   ├── examples/              # Reproduced/new examples
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
**Goal**: Run JAX examples, understand JIT compilation flow
**Deliverables**:
- Working JAX installation (with jaxlib)
- 3+ examples using `@jax.jit` decorator
- `notes/jax_basics.md`: How functions compile, how to access IR (max 50 lines)

**Key JAX Concepts to Explore**:
- `jax.jit()` decorator for Just-In-Time compilation
- `jax.make_jaxpr()` for extracting intermediate representation (Jaxpr)
- `jax.xla_computation()` for XLA HLO representation
- Backend types: CPU, GPU (CUDA), TPU

### M2: IR Extraction Mechanism
**Goal**: Programmatically extract IR from JAX functions
**Deliverables**:
- `code/extraction/ir_extractor.py`: Function that takes a JAX function → returns IR
- Support for multiple IR formats:
  - Jaxpr (JAX's intermediate representation)
  - XLA HLO (High Level Operations)
  - StableHLO (for portability)
- 5+ test cases showing Python function → IR pairs
- `notes/ir_format.md`: IR structure documentation (max 30 lines)

**Technical Details**:
```python
import jax
import jax.numpy as jnp

# Extract Jaxpr
def my_function(x):
    return jnp.sin(x) + x * 2

jaxpr = jax.make_jaxpr(my_function)(jnp.array(1.0))
print(jaxpr)

# Extract XLA HLO
computation = jax.xla_computation(my_function)(jnp.array(1.0))
hlo_text = computation.as_hlo_text()
print(hlo_text)

# Extract StableHLO
stablehlo = computation.as_serialized_hlo_module_proto()
```

### M3: JAX Gradient and Transformation Deep Dive
**Goal**: Extract IR from transformed functions (grad, vmap, etc.)
**Deliverables**:
- `code/examples/gradient_examples.py`: Functions using `jax.grad`, `jax.value_and_grad`
- `code/examples/vmap_examples.py`: Functions using `jax.vmap` (vectorization)
- `code/examples/scan_examples.py`: Functions using `jax.lax.scan` (loops)
- `code/examples/test_transformations.py`: Validation tests
- Tests pass for 2+ consecutive runs

**JAX Transformations to Cover**:
- `jax.grad()` - automatic differentiation
- `jax.value_and_grad()` - function value + gradient
- `jax.vmap()` - automatic vectorization
- `jax.pmap()` - parallel map (multi-device)
- `jax.lax.scan()` - efficient loops
- `jax.lax.cond()` - conditional execution
- `jax.lax.while_loop()` - while loops

### M4: Synthesis Pipeline
**Goal**: Automated Python→IR data generation
**Deliverables**:
- `code/synthesis/generator.py`: Generates varied JAX functions programmatically
- `code/synthesis/pipeline.py`: End-to-end: generate function → jit compile → extract IR → save pair
- `data/samples/`: 100+ sample pairs for validation

**Function Categories to Generate**:
1. **Basic Arithmetic**: +, -, *, /, %, **
2. **Array Operations**: reshape, transpose, concatenate, slice
3. **Math Functions**: sin, cos, exp, log, sqrt, tanh
4. **Linear Algebra**: matmul, dot, norm, svd, eig
5. **Reductions**: sum, mean, max, min, prod
6. **Indexing**: advanced indexing, dynamic updates
7. **Gradients**: functions with grad/value_and_grad
8. **Vectorization**: functions with vmap
9. **Conditionals**: lax.cond, lax.select
10. **Loops**: lax.scan, lax.while_loop, lax.fori_loop

### M5: Scale Up
**Goal**: Generate large-scale training dataset
**Deliverables**:
- `code/synthesis/batch_generator.py`: Parallel/batched generation
- `data/`: 10k+ Python→IR pairs
- `notes/data_stats.md`: Dataset statistics (max 20 lines)

**Optimization Strategies**:
- Use `jax.jit` compilation caching
- Parallel generation across CPU cores
- Generate variations: different dtypes (float32/64, int32/64), shapes, dimensions
- Include both forward and backward (gradient) passes

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
2. Results must match both times (deterministic)
3. No uncommitted debug code or prints
4. Code runs from clean state (no hidden dependencies)

**Note on JAX Randomness**: Use `jax.random.PRNGKey` for reproducible random number generation.

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
- JAX documentation: https://jax.readthedocs.io/
- JAX examples: https://github.com/google/jax/tree/main/examples
- Key files to study:
  - `jax/_src/interpreters/mlir.py` (MLIR/StableHLO generation)
  - `jax/_src/core.py` (Jaxpr core)
  - `jax/_src/api.py` (JIT, grad, vmap implementations)
  - `jax/_src/lax/lax.py` (Primitive operations)

**Installation**:
```bash
# CPU-only
pip install jax jaxlib

# CUDA 12 support (if GPU available)
pip install -U jax[cuda12]

# CUDA 11 support
pip install -U jax[cuda11_local]
```

---

## JAX-Specific Considerations

### 1. Functional Programming Paradigm
JAX functions must be:
- **Pure**: No side effects, same input → same output
- **Stateless**: No mutable state (use functional updates)

### 2. Array Operations
- Use `jax.numpy` instead of `numpy` for differentiable operations
- Arrays are immutable (use `.at[].set()` for updates)
- Shapes must be static (known at compile time) for most operations

### 3. IR Formats Available
- **Jaxpr**: JAX's intermediate representation (readable, high-level)
- **XLA HLO**: Lower-level, closer to hardware operations
- **StableHLO**: Portable format for interoperability
- **MLIR**: Multi-Level Intermediate Representation (advanced)

### 4. Compilation Behavior
- First call to jitted function: compilation (slow)
- Subsequent calls with same shape/dtype: cached (fast)
- Different shapes/dtypes: recompilation

### 5. Backend Selection
```python
# Default (first available: GPU > TPU > CPU)
jax.jit(fn)(x)

# Force CPU
with jax.default_device(jax.devices('cpu')[0]):
    jax.jit(fn)(x)

# Force GPU (if available)
with jax.default_device(jax.devices('gpu')[0]):
    jax.jit(fn)(x)
```

---

## Sample Code Structure

### Basic IR Extraction Example

```python
# code/extraction/basic_extractor.py
import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any

def extract_jaxpr(fn: Callable, *args) -> str:
    """Extract Jaxpr IR from a JAX function."""
    jaxpr = jax.make_jaxpr(fn)(*args)
    return str(jaxpr)

def extract_hlo(fn: Callable, *args) -> str:
    """Extract XLA HLO IR from a JAX function."""
    computation = jax.xla_computation(fn)(*args)
    return computation.as_hlo_text()

def extract_all_ir(fn: Callable, *args) -> Dict[str, str]:
    """Extract all available IR formats."""
    return {
        'jaxpr': extract_jaxpr(fn, *args),
        'hlo': extract_hlo(fn, *args),
    }

# Example usage
def example_fn(x):
    return jnp.sin(x) * 2 + jnp.cos(x)

x = jnp.array(1.0)
ir_dict = extract_all_ir(example_fn, x)
print(ir_dict['jaxpr'])
```

### Generator Template

```python
# code/synthesis/generator_template.py
import jax
import jax.numpy as jnp
from typing import Callable, List
import random

class JAXFunctionGenerator:
    """Generate random JAX functions for IR extraction."""
    
    def generate_arithmetic(self) -> Callable:
        """Generate function with arithmetic operations."""
        ops = ['+', '-', '*', '/', '**']
        op = random.choice(ops)
        
        def fn(x, y):
            if op == '+': return x + y
            elif op == '-': return x - y
            elif op == '*': return x * y
            elif op == '/': return x / y
            elif op == '**': return x ** y
        
        return fn
    
    def generate_math(self) -> Callable:
        """Generate function with math operations."""
        ops = ['sin', 'cos', 'exp', 'log', 'sqrt', 'tanh']
        op = random.choice(ops)
        
        def fn(x):
            if op == 'sin': return jnp.sin(x)
            elif op == 'cos': return jnp.cos(x)
            elif op == 'exp': return jnp.exp(x)
            elif op == 'log': return jnp.log(x)
            elif op == 'sqrt': return jnp.sqrt(x)
            elif op == 'tanh': return jnp.tanh(x)
        
        return fn
    
    def generate_gradient(self) -> Callable:
        """Generate function with gradient computation."""
        def base_fn(x):
            return jnp.sum(x ** 2)
        
        return jax.grad(base_fn)
    
    def generate_vmap(self) -> Callable:
        """Generate vectorized function."""
        def base_fn(x):
            return jnp.sin(x) + x ** 2
        
        return jax.vmap(base_fn)
```

---

## Anti-Patterns (Avoid These)

- ❌ Writing summaries, READMEs, or reports
- ❌ Over-commenting code
- ❌ Starting new tasks with <30k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Reading entire large files (use targeted searches)
- ❌ Re-exploring already-documented findings
- ❌ Using mutable state in JAX functions (will break JIT)
- ❌ Ignoring shape/dtype constraints (causes recompilation)
- ❌ Using Python control flow without `jax.lax` (won't trace properly)

---

## Common JAX Pitfalls & Solutions

### Problem: Python control flow in JIT functions
```python
# ❌ Bad - Python if won't trace
@jax.jit
def bad_fn(x):
    if x > 0:  # Error: uses Python bool
        return x * 2
    return x

# ✅ Good - Use jax.lax.cond
@jax.jit
def good_fn(x):
    return jax.lax.cond(x > 0, lambda x: x * 2, lambda x: x, x)
```

### Problem: In-place array updates
```python
# ❌ Bad - arrays are immutable
x = jnp.array([1, 2, 3])
x[0] = 5  # Error

# ✅ Good - functional update
x = jnp.array([1, 2, 3])
x = x.at[0].set(5)
```

### Problem: Shape changes in JIT
```python
# ❌ Bad - dynamic shapes
@jax.jit
def bad_fn(x):
    return x[:len(x)//2]  # len(x) not known at compile time

# ✅ Good - static shapes
@jax.jit
def good_fn(x, n):
    return x[:n]  # n is traced as abstract value
```

---

## Success Criteria

Project is complete when:
1. Can extract Jaxpr and HLO IR from any JAX function
2. Generator creates 10+ categories of functions
3. Pipeline generates Python→IR pairs automatically
4. 10k+ diverse training samples generated
5. All samples validated (IR parseable, matches function semantics)
6. Code is clean, documented, and reproducible
