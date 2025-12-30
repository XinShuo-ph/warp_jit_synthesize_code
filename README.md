# JIT Code Synthesis using JAX

This project extracts intermediate representations (IR) from JAX-compiled functions to create training data for Large Language Models (LLMs). The goal is to generate high-quality Python→IR paired data for training models on compiler optimization and code transformation tasks.

---

## 📋 Quick Start

### 1. Install JAX
```bash
# CPU-only (recommended for getting started)
pip install jax jaxlib

# With CUDA 12 support (if you have a GPU)
pip install -U "jax[cuda12]"
```

### 2. Understand the Basics
Read the quick reference to see JAX in action:
```bash
cat JAX_QUICK_REFERENCE.md
```

### 3. Follow the Instructions
Start implementing the 5-milestone roadmap:
```bash
cat instructions_jax.md
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| **[instructions_jax.md](instructions_jax.md)** | Complete implementation guide (5 milestones) |
| **[JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md)** | Essential commands, patterns, and gotchas |
| **[JAX_IR_EXAMPLES.md](JAX_IR_EXAMPLES.md)** | 10 concrete Python→IR transformation examples |
| **[WARP_TO_JAX_MIGRATION.md](WARP_TO_JAX_MIGRATION.md)** | Comparison with original Warp-based approach |
| **[JAX_MIGRATION_SUMMARY.md](JAX_MIGRATION_SUMMARY.md)** | High-level overview and rationale |
| **[STATE.md](STATE.md)** | Current project status and next actions |

---

## 🎯 What This Project Does

### Input: Python Functions
```python
def my_function(x, y):
    return jnp.sin(x) + y * 2
```

### Output 1: Jaxpr (High-level IR)
```
{ lambda ; a:f32[] b:f32[]. let
    c:f32[] = sin a
    d:f32[] = mul b 2.0
    e:f32[] = add c d
  in (e,) }
```

### Output 2: XLA HLO (Low-level IR)
```
HloModule jit_my_function

ENTRY main.5 {
  Arg_0.1 = f32[] parameter(0)
  Arg_1.2 = f32[] parameter(1)
  sine.3 = f32[] sine(Arg_0.1)
  constant.4 = f32[] constant(2)
  multiply.5 = f32[] multiply(Arg_1.2, constant.4)
  ROOT add.6 = f32[] add(sine.3, multiply.5)
}
```

---

## 🏗️ Project Structure

```
jit/
├── instructions_jax.md       # Main implementation guide
├── STATE.md                   # Progress tracking
├── tasks/                     # Task breakdowns per milestone
│   ├── m1_tasks.md           # Environment setup
│   ├── m2_tasks.md           # IR extraction
│   ├── m3_tasks.md           # Transformations (grad, vmap)
│   ├── m4_tasks.md           # Synthesis pipeline
│   └── m5_tasks.md           # Scale up to 10k+ samples
├── code/
│   ├── extraction/           # IR extraction utilities
│   │   └── ir_extractor.py  # Extract Jaxpr/HLO from functions
│   ├── synthesis/            # Data generation pipeline
│   │   ├── generator.py     # Generate diverse JAX functions
│   │   ├── pipeline.py      # End-to-end Python→IR generation
│   │   └── batch_generator.py # Parallel generation
│   └── examples/             # Example functions & tests
│       ├── gradient_examples.py
│       ├── vmap_examples.py
│       └── scan_examples.py
├── data/                      # Generated training data
│   └── samples/              # Python→IR pairs (JSON)
└── notes/                     # Technical findings (minimal)
    ├── jax_basics.md         # How JAX compilation works
    └── ir_format.md          # IR structure documentation
```

---

## 🚀 5 Milestones

### M1: Environment Setup & JAX Basics
- Install JAX and verify installation
- Run 3+ basic examples using `@jax.jit`
- Extract Jaxpr and HLO from simple functions
- Document JAX compilation flow

### M2: IR Extraction Mechanism
- Build `ir_extractor.py` supporting Jaxpr and HLO
- Create 5+ test cases with varied functions
- Document IR format structure
- Validate extraction works reliably

### M3: Transformations Deep Dive
- Extract IR from `jax.grad()` (automatic differentiation)
- Extract IR from `jax.vmap()` (vectorization)
- Extract IR from `jax.lax.scan()` (loops)
- Extract IR from `jax.lax.cond()` (conditionals)
- Validate all transformations work correctly

### M4: Synthesis Pipeline
- Build function generator for 10+ categories:
  - Arithmetic, Math, Array ops, Linear algebra
  - Reductions, Indexing, Gradients, Vectorization
  - Conditionals, Loops
- Create end-to-end pipeline: generate → compile → extract → save
- Generate 100+ validation samples

### M5: Scale Up
- Implement parallel batch generation
- Generate 10k+ diverse Python→IR pairs
- Include variations: dtypes (float32/64, int32/64), shapes, dimensions
- Generate both forward and backward (gradient) passes
- Document dataset statistics

---

## 🎨 Function Categories

The generator will create diverse JAX functions across these categories:

1. **Arithmetic**: `x + y`, `x * y - z`, `x ** 2`
2. **Math**: `jnp.sin(x)`, `jnp.exp(x)`, `jnp.log(x)`
3. **Array**: `reshape`, `transpose`, `concatenate`, `slice`
4. **Linear Algebra**: `matmul`, `dot`, `norm`, `svd`
5. **Reductions**: `sum`, `mean`, `max`, `min`, `prod`
6. **Indexing**: advanced indexing, dynamic updates
7. **Gradients**: `jax.grad(fn)`, `jax.value_and_grad(fn)`
8. **Vectorization**: `jax.vmap(fn)`, batched operations
9. **Conditionals**: `jax.lax.cond`, `jax.lax.select`
10. **Loops**: `jax.lax.scan`, `jax.lax.while_loop`, `jax.lax.fori_loop`

Each category will have multiple variations with different:
- Input shapes (scalars, vectors, matrices, tensors)
- Data types (float32, float64, int32, int64)
- Complexity levels (simple to nested operations)

---

## 💡 Why JAX?

### Advantages over Warp
1. **Multiple IR formats**: Jaxpr (readable) + HLO (optimized) + StableHLO (portable)
2. **Richer transformations**: grad, vmap, pmap, scan, cond, while_loop
3. **More ML-relevant**: Training data directly applicable to ML compilers
4. **Easier to generate**: No explicit GPU kernel writing required
5. **Better documentation**: Extensive examples and active community

### Advantages over Plain Python
1. **JIT compilation**: Functions are compiled, not just interpreted
2. **Multiple backends**: CPU, GPU (CUDA), TPU support
3. **Automatic differentiation**: Built-in gradient computation
4. **Functional paradigm**: Pure functions with explicit data flow
5. **Optimization opportunities**: Real compiler IR, not bytecode

---

## 📊 Expected Dataset

### Quantitative Goals
- **10,000+ Python functions** across 10 categories
- **20,000+ IR samples** (2 formats × 10k functions)
- **Variations**: Multiple dtypes, shapes, transformations
- **Both directions**: Forward pass + backward (gradient) pass

### Quality Goals
- **Diverse operations**: Cover JAX's operation space
- **Valid IR**: All samples compile and execute successfully
- **Readable Python**: Clean, well-structured source code
- **Documented**: Each sample includes metadata (shapes, dtypes)

### Data Format (JSON)
```json
{
  "source_code": "def fn(x): return jnp.sin(x)",
  "jaxpr": "{ lambda ; a:f32[]. let b:f32[] = sin a in (b,) }",
  "hlo": "HloModule jit_fn\n\nENTRY main.2 { ... }",
  "category": "math",
  "input_shapes": ["()"],
  "input_dtypes": ["float32"],
  "test_success": true
}
```

---

## 🔧 Development Workflow

### Session Start
1. Read `STATE.md` to understand current progress
2. Read current milestone's task file (e.g., `tasks/m1_tasks.md`)
3. Resume from documented next action

### During Development
1. Work on one task at a time
2. Test frequently (run code twice, verify determinism)
3. Keep notes minimal (focus on implementation)
4. Commit working code regularly

### Session End
1. Update `STATE.md` with:
   - Current milestone and task
   - Exact next action
   - Any blockers or findings
2. Commit all working code
3. No broken states left behind

---

## 🚦 Validation Protocol

Before marking any task complete:
1. ✅ Run the code twice
2. ✅ Results must match both times (deterministic)
3. ✅ No uncommitted debug code or print statements
4. ✅ Code runs from clean state (no hidden dependencies)
5. ✅ IR extraction succeeds for all test cases

---

## 🎓 Learning Resources

### Official JAX Documentation
- **Main docs**: https://jax.readthedocs.io/
- **GitHub**: https://github.com/google/jax
- **Examples**: https://github.com/google/jax/tree/main/examples

### Key Topics
- **Autodiff cookbook**: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
- **Common gotchas**: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
- **Thinking in JAX**: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html

### JAX Source Code
- `jax/_src/api.py` - JIT, grad, vmap implementations
- `jax/_src/core.py` - Jaxpr core
- `jax/_src/interpreters/mlir.py` - MLIR/StableHLO generation
- `jax/_src/lax/lax.py` - Primitive operations

---

## ⚠️ Common Pitfalls

### ❌ Don't use Python control flow in JIT
```python
@jax.jit
def bad(x):
    if x > 0:  # Error! Python bool() of traced value
        return x * 2
    return x
```

### ✅ Use jax.lax instead
```python
@jax.jit
def good(x):
    return jax.lax.cond(x > 0, lambda x: x * 2, lambda x: x, x)
```

### ❌ Don't mutate arrays
```python
x[0] = 5  # Error! Arrays are immutable
```

### ✅ Use functional updates
```python
x = x.at[0].set(5)  # Correct!
```

See [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md) for more examples.

---

## 🎯 Success Criteria

Project is complete when:
1. ✅ Can extract Jaxpr and HLO IR from any JAX function
2. ✅ Generator creates 10+ categories of functions
3. ✅ Pipeline generates Python→IR pairs automatically
4. ✅ 10k+ diverse training samples generated
5. ✅ All samples validated (IR parseable, matches function semantics)
6. ✅ Code is clean, documented, and reproducible

---

## 🤝 Contributing

This is a solo project following a structured milestone-based approach. See `instructions_jax.md` for the complete implementation plan.

---

## 📝 License

This project is for research and educational purposes.

---

## 🔗 Quick Links

- **Start here**: [instructions_jax.md](instructions_jax.md)
- **Quick reference**: [JAX_QUICK_REFERENCE.md](JAX_QUICK_REFERENCE.md)
- **Examples**: [JAX_IR_EXAMPLES.md](JAX_IR_EXAMPLES.md)
- **Progress**: [STATE.md](STATE.md)
- **Migration info**: [WARP_TO_JAX_MIGRATION.md](WARP_TO_JAX_MIGRATION.md)

---

**Status**: Ready to begin M1 (Environment Setup & JAX Basics)

Last updated: 2025-12-30
