# Warp to JAX Migration Guide

## Key Differences

| Aspect | Warp | JAX |
|--------|------|-----|
| **Purpose** | Physics simulation, GPU kernels | ML research, automatic differentiation |
| **Language** | Python with kernel decorators | Pure Python (functional) |
| **Backend** | CUDA, CPU | XLA (CPU, GPU, TPU) |
| **IR Format** | PTX, CUDA C++ | Jaxpr, XLA HLO, StableHLO |
| **Paradigm** | Imperative kernels | Functional transformations |
| **Compilation** | `@wp.kernel` decorator | `@jax.jit` decorator |
| **Gradients** | `wp.Tape()` for autodiff | `jax.grad()`, `jax.value_and_grad()` |

---

## Concept Mapping

### 1. Kernel Definition

**Warp:**
```python
import warp as wp

@wp.kernel
def add_kernel(a: wp.array(dtype=float), 
               b: wp.array(dtype=float),
               c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = a[i] + b[i]
```

**JAX Equivalent:**
```python
import jax
import jax.numpy as jnp

@jax.jit
def add_function(a, b):
    return a + b

# Or with explicit vectorization
@jax.jit
@jax.vmap
def add_function_vmap(a, b):
    return a + b
```

### 2. IR Extraction

**Warp:**
```python
# Access compiled kernel's PTX or generated C++
kernel_module = wp.get_module(add_kernel)
ptx_code = kernel_module.ptx
```

**JAX:**
```python
# Extract Jaxpr (high-level IR)
jaxpr = jax.make_jaxpr(add_function)(a, b)
print(jaxpr)

# Extract XLA HLO (low-level IR)
computation = jax.xla_computation(add_function)(a, b)
hlo = computation.as_hlo_text()
print(hlo)
```

### 3. Gradients

**Warp:**
```python
# Use tape for reverse-mode autodiff
tape = wp.Tape()
with tape:
    wp.launch(kernel, dim=n, inputs=[a, b], outputs=[c])
tape.backward()
```

**JAX:**
```python
# Built-in automatic differentiation
grad_fn = jax.grad(loss_function)
gradients = grad_fn(params)

# Or get both value and gradient
value_and_grad_fn = jax.value_and_grad(loss_function)
loss, gradients = value_and_grad_fn(params)
```

### 4. Array Types

**Warp:**
```python
# Warp arrays with explicit types
a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device='cuda')
```

**JAX:**
```python
# JAX arrays (similar to numpy)
a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

# Backend selection via device context
with jax.default_device(jax.devices('gpu')[0]):
    a = jnp.array([1.0, 2.0, 3.0])
```

### 5. Vectorization

**Warp:**
```python
# Implicit parallelization via kernel launch
wp.launch(kernel, dim=1000, inputs=[a, b], outputs=[c])
```

**JAX:**
```python
# Explicit vectorization with vmap
scalar_fn = lambda x: x ** 2
vector_fn = jax.vmap(scalar_fn)
result = vector_fn(array)
```

---

## IR Format Comparison

### Warp PTX Example
```ptx
.version 7.0
.target sm_70
.address_size 64

.visible .entry add_kernel(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd3], %f3;
}
```

### JAX Jaxpr Example
```python
{ lambda ; a:f32[3] b:f32[3]. let
    c:f32[3] = add a b
  in (c,) }
```

### JAX XLA HLO Example
```
HloModule jit_add_function

ENTRY main.3 {
  Arg_0.1 = f32[3]{0} parameter(0)
  Arg_1.2 = f32[3]{0} parameter(1)
  ROOT add.3 = f32[3]{0} add(Arg_0.1, Arg_1.2)
}
```

---

## Migration Strategy

### For Data Generation Pipeline:

1. **Replace Warp kernel definitions** → JAX jitted functions
2. **Replace `wp.launch()`** → Direct function calls or `jax.vmap()`
3. **Replace Warp types** → JAX numpy types
4. **Replace PTX extraction** → Jaxpr/HLO extraction
5. **Keep same overall pipeline structure**

### Advantages of JAX for IR Dataset:

1. **Multiple IR levels**: Jaxpr (readable) + HLO (detailed) + StableHLO (portable)
2. **Richer transformations**: grad, vmap, pmap, scan, cond
3. **Easier to generate diverse code**: No explicit parallelization needed
4. **Better type inference**: Automatic shape/dtype propagation
5. **More ML-relevant**: Training data directly applicable to ML compilers

### Challenges:

1. **Functional paradigm**: Must avoid mutable state
2. **Static shapes**: Many operations require compile-time known shapes
3. **Control flow**: Must use `jax.lax` primitives, not Python control flow
4. **Different debugging**: Less imperative, more tracing-based

---

## Dataset Quality Considerations

### Python → IR Pairs

**Warp pairs**: Python kernel code → PTX assembly
- Good for: GPU programming, kernel optimization
- Limited by: GPU-specific operations

**JAX pairs**: Python function code → Jaxpr/HLO
- Good for: ML compilers, optimization passes, general compute
- Richer because:
  - Multiple transformation types (grad, vmap, etc.)
  - Multiple IR levels
  - Shape/type polymorphism
  - Control flow variations

### Diversity Opportunities with JAX:

1. **Transformation chains**: 
   - `jax.jit(jax.grad(fn))`
   - `jax.jit(jax.vmap(jax.grad(fn)))`
   
2. **Different dtypes**: float16/32/64, int8/16/32/64, complex64/128

3. **Shape variations**: scalars, vectors, matrices, tensors

4. **Control patterns**: cond, select, scan, while_loop, fori_loop

5. **Math operations**: 100+ numpy-compatible operations

---

## Recommended Approach

1. **Start simple** (M1): Basic arithmetic, math functions
2. **Add transformations** (M3): grad, vmap on simple functions
3. **Complex control flow** (M4): scan, cond, while_loop
4. **Scale up** (M5): Generate all combinations

This will produce a high-quality dataset covering:
- Basic compute patterns
- Automatic differentiation (crucial for ML)
- Vectorization patterns
- Control flow compilation
- Multiple IR representations
