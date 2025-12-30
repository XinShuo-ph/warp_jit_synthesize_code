# JAX IR Examples

This document shows actual examples of Python code and their corresponding IR representations in both Jaxpr and XLA HLO formats.

---

## Example 1: Simple Arithmetic

### Python Code
```python
def simple_add(x, y):
    return x + y
```

### Input
```python
x = jnp.array(1.0)  # shape: (), dtype: float32
y = jnp.array(2.0)  # shape: (), dtype: float32
```

### Jaxpr Output
```
{ lambda ; a:f32[] b:f32[]. let
    c:f32[] = add a b
  in (c,) }
```

### XLA HLO Output
```
HloModule jit_simple_add

ENTRY main.3 {
  Arg_0.1 = f32[] parameter(0)
  Arg_1.2 = f32[] parameter(1)
  ROOT add.3 = f32[] add(Arg_0.1, Arg_1.2)
}
```

---

## Example 2: Math Operations

### Python Code
```python
def math_ops(x):
    return jnp.sin(x) + jnp.cos(x) * 2
```

### Input
```python
x = jnp.array(1.0)  # shape: (), dtype: float32
```

### Jaxpr Output
```
{ lambda ; a:f32[]. let
    b:f32[] = sin a
    c:f32[] = cos a
    d:f32[] = mul c 2.0
    e:f32[] = add b d
  in (e,) }
```

### XLA HLO Output
```
HloModule jit_math_ops

ENTRY main.5 {
  Arg_0.1 = f32[] parameter(0)
  sine.2 = f32[] sine(Arg_0.1)
  cosine.3 = f32[] cosine(Arg_0.1)
  constant.4 = f32[] constant(2)
  multiply.5 = f32[] multiply(cosine.3, constant.4)
  ROOT add.6 = f32[] add(sine.2, multiply.5)
}
```

---

## Example 3: Array Operations

### Python Code
```python
def array_sum(x):
    return jnp.sum(x ** 2)
```

### Input
```python
x = jnp.array([1.0, 2.0, 3.0])  # shape: (3,), dtype: float32
```

### Jaxpr Output
```
{ lambda ; a:f32[3]. let
    b:f32[3] = integer_pow[y=2] a
    c:f32[] = reduce_sum[axes=(0,)] b
  in (c,) }
```

### XLA HLO Output
```
HloModule jit_array_sum

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main.4 {
  Arg_0.1 = f32[3]{0} parameter(0)
  multiply.2 = f32[3]{0} multiply(Arg_0.1, Arg_0.1)
  constant.3 = f32[] constant(0)
  ROOT reduce.4 = f32[] reduce(multiply.2, constant.3), dimensions={0}, to_apply=add
}
```

---

## Example 4: Gradient

### Python Code
```python
def loss_fn(x):
    return jnp.sum(x ** 2)

grad_fn = jax.grad(loss_fn)
```

### Input
```python
x = jnp.array([1.0, 2.0, 3.0])  # shape: (3,), dtype: float32
```

### Jaxpr Output (of grad_fn)
```
{ lambda ; a:f32[3]. let
    b:f32[3] = mul 2.0 a
  in (b,) }
```

### XLA HLO Output (of grad_fn)
```
HloModule jit_grad

ENTRY main.2 {
  Arg_0.1 = f32[3]{0} parameter(0)
  constant.2 = f32[] constant(2)
  broadcast.3 = f32[3]{0} broadcast(constant.2), dimensions={}
  ROOT multiply.4 = f32[3]{0} multiply(Arg_0.1, broadcast.3)
}
```

---

## Example 5: Vectorized Function (vmap)

### Python Code
```python
def scalar_fn(x):
    return x ** 2 + jnp.sin(x)

vector_fn = jax.vmap(scalar_fn)
```

### Input
```python
x = jnp.array([1.0, 2.0, 3.0])  # shape: (3,), dtype: float32
```

### Jaxpr Output
```
{ lambda ; a:f32[3]. let
    b:f32[3] = integer_pow[y=2] a
    c:f32[3] = sin a
    d:f32[3] = add b c
  in (d,) }
```

### XLA HLO Output
```
HloModule jit_vector_fn

ENTRY main.4 {
  Arg_0.1 = f32[3]{0} parameter(0)
  multiply.2 = f32[3]{0} multiply(Arg_0.1, Arg_0.1)
  sine.3 = f32[3]{0} sine(Arg_0.1)
  ROOT add.4 = f32[3]{0} add(multiply.2, sine.3)
}
```

---

## Example 6: Conditional

### Python Code
```python
def conditional_fn(x):
    return jax.lax.cond(
        x > 0,
        lambda x: x * 2,
        lambda x: x / 2,
        x
    )
```

### Input
```python
x = jnp.array(1.0)  # shape: (), dtype: float32
```

### Jaxpr Output
```
{ lambda ; a:f32[]. let
    b:bool[] = gt a 0.0
    c:f32[] = cond[
      false_jaxpr={ lambda ; d:f32[]. let e:f32[] = div d 2.0 in (e,) }
      true_jaxpr={ lambda ; d:f32[]. let e:f32[] = mul d 2.0 in (e,) }
    ] b a
  in (c,) }
```

### XLA HLO Output
```
HloModule jit_conditional_fn

region_0.true {
  Arg_0.1 = f32[] parameter(0)
  constant.2 = f32[] constant(2)
  ROOT multiply.3 = f32[] multiply(Arg_0.1, constant.2)
}

region_1.false {
  Arg_0.4 = f32[] parameter(0)
  constant.5 = f32[] constant(2)
  ROOT divide.6 = f32[] divide(Arg_0.4, constant.5)
}

ENTRY main.9 {
  Arg_0.7 = f32[] parameter(0)
  constant.8 = f32[] constant(0)
  compare.9 = pred[] compare(Arg_0.7, constant.8), direction=GT
  ROOT conditional.10 = f32[] conditional(compare.9, Arg_0.7, Arg_0.7),
    true_computation=region_0.true, false_computation=region_1.false
}
```

---

## Example 7: Loop (scan)

### Python Code
```python
def scan_fn(init, n):
    def body(carry, i):
        carry = carry + i
        return carry, carry
    
    final, history = jax.lax.scan(body, init, jnp.arange(n))
    return final
```

### Input
```python
init = jnp.array(0.0)  # shape: (), dtype: float32
n = 5
```

### Jaxpr Output
```
{ lambda ; a:f32[] b:i32. let
    c:i32[5] = iota[dimension=0 shape=(5,) dtype=int32]
    d:f32[] e:f32[5] = scan[
      jaxpr={ lambda ; f:f32[] g:i32. let
          h:f32[] = convert_element_type[new_dtype=float32] g
          i:f32[] = add f h
        in (i, i) }
      length=5
      reverse=False
      unroll=1
    ] a c
  in (d,) }
```

### XLA HLO Output (simplified)
```
HloModule jit_scan_fn

body {
  param.0 = (f32[], i32[], f32[5]) parameter(0)
  get-tuple-element.1 = f32[] get-tuple-element(param.0), index=0
  get-tuple-element.2 = i32[] get-tuple-element(param.0), index=1
  convert.3 = f32[] convert(get-tuple-element.2)
  add.4 = f32[] add(get-tuple-element.1, convert.3)
  ...
  ROOT tuple.5 = (f32[], ...) tuple(add.4, ...)
}

ENTRY main {
  Arg_0 = f32[] parameter(0)
  iota.1 = i32[5] iota(), iota_dimension=0
  ROOT while.2 = (f32[], ...) while(...), condition=cond, body=body
}
```

---

## Example 8: Matrix Multiplication

### Python Code
```python
def matmul_fn(A, B):
    return jnp.matmul(A, B)
```

### Input
```python
A = jnp.ones((3, 4))  # shape: (3, 4), dtype: float32
B = jnp.ones((4, 5))  # shape: (4, 5), dtype: float32
```

### Jaxpr Output
```
{ lambda ; a:f32[3,4] b:f32[4,5]. let
    c:f32[3,5] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] a b
  in (c,) }
```

### XLA HLO Output
```
HloModule jit_matmul_fn

ENTRY main.3 {
  Arg_0.1 = f32[3,4]{1,0} parameter(0)
  Arg_1.2 = f32[4,5]{1,0} parameter(1)
  ROOT dot.3 = f32[3,5]{1,0} dot(Arg_0.1, Arg_1.2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
```

---

## Example 9: Complex Combination

### Python Code
```python
def complex_fn(x, y):
    # Combine multiple operations
    z = jnp.sin(x) + jnp.cos(y)
    w = jnp.exp(z)
    return jnp.sum(w ** 2)
```

### Input
```python
x = jnp.array([1.0, 2.0])  # shape: (2,), dtype: float32
y = jnp.array([3.0, 4.0])  # shape: (2,), dtype: float32
```

### Jaxpr Output
```
{ lambda ; a:f32[2] b:f32[2]. let
    c:f32[2] = sin a
    d:f32[2] = cos b
    e:f32[2] = add c d
    f:f32[2] = exp e
    g:f32[2] = integer_pow[y=2] f
    h:f32[] = reduce_sum[axes=(0,)] g
  in (h,) }
```

### XLA HLO Output
```
HloModule jit_complex_fn

add_reducer {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main.9 {
  Arg_0.1 = f32[2]{0} parameter(0)
  Arg_1.2 = f32[2]{0} parameter(1)
  sine.3 = f32[2]{0} sine(Arg_0.1)
  cosine.4 = f32[2]{0} cosine(Arg_1.2)
  add.5 = f32[2]{0} add(sine.3, cosine.4)
  exponential.6 = f32[2]{0} exponential(add.5)
  multiply.7 = f32[2]{0} multiply(exponential.6, exponential.6)
  constant.8 = f32[] constant(0)
  ROOT reduce.9 = f32[] reduce(multiply.7, constant.8),
    dimensions={0}, to_apply=add_reducer
}
```

---

## Example 10: Gradient of Complex Function

### Python Code
```python
def loss_fn(params, x):
    return jnp.sum((params['w'] * x + params['b']) ** 2)

grad_fn = jax.grad(loss_fn)
```

### Input
```python
params = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array([0.5, 0.5])}
x = jnp.array([1.0, 2.0])
```

### Jaxpr Output (of grad_fn)
```
{ lambda ; a:f32[2] b:f32[2] c:f32[2]. let
    d:f32[2] = mul a c
    e:f32[2] = add d b
    f:f32[2] = mul 2.0 e
    g:f32[2] = mul f c
    h:f32[2] = mul f 1.0
  in ({'b': h, 'w': g},) }
```

---

## Key Observations

### Jaxpr Characteristics:
1. **High-level**: Close to Python operations
2. **Readable**: Easy to understand operation flow
3. **Typed**: Every variable has explicit type/shape
4. **Functional**: Immutable variables, explicit data flow

### XLA HLO Characteristics:
1. **Low-level**: Closer to hardware operations
2. **Optimized**: Includes fusion opportunities, layout hints
3. **Explicit**: All operations fully specified
4. **Backend-specific**: May vary slightly by device

### Training Data Value:
- **Jaxpr**: Good for learning high-level transformations
- **HLO**: Good for learning optimization passes
- **Both together**: Ideal for multi-level compiler learning

### Diversity in Dataset:
- Different shapes: scalars, vectors, matrices, tensors
- Different dtypes: float32, float64, int32, int64
- Different operations: math, linalg, control flow
- Different transformations: grad, vmap, scan, cond
- Different complexities: simple ops to complex pipelines

---

## How to Generate These Examples

```python
import jax
import jax.numpy as jnp

def show_ir(fn, *args, name="function"):
    """Display both Jaxpr and HLO for a function."""
    print(f"\n{'='*60}")
    print(f"Function: {name}")
    print(f"{'='*60}")
    
    # Show Jaxpr
    print("\nJaxpr:")
    print("-" * 60)
    jaxpr = jax.make_jaxpr(fn)(*args)
    print(jaxpr)
    
    # Show HLO
    print("\nXLA HLO:")
    print("-" * 60)
    computation = jax.xla_computation(fn)(*args)
    print(computation.as_hlo_text())
    
    # Test execution
    print("\nTest execution:")
    print("-" * 60)
    result = fn(*args)
    print(f"Result: {result}")
    print(f"Shape: {result.shape}, Dtype: {result.dtype}")

# Example usage
def my_fn(x, y):
    return jnp.sin(x) + y ** 2

x = jnp.array([1.0, 2.0])
y = jnp.array([3.0, 4.0])

show_ir(my_fn, x, y, name="my_fn")
```

This will print formatted output showing both IR representations along with test execution results.
