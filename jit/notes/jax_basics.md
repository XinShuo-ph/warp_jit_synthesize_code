# JAX JIT Basics

## IR Types

1. **Jaxpr** - JAX's traced representation
   - Functional IR with primitives (add, mul, sin, dot_general, etc.)
   - Shows explicit data flow: `{ lambda ; inputs. let ops in (outputs,) }`
   - Obtained via `jax.make_jaxpr(fn)(*args)`

2. **StableHLO** - XLA's high-level IR
   - MLIR-based format with `stablehlo.*` operations
   - More verbose, includes type annotations on every op
   - Obtained via `jax.jit(fn).lower(*args).as_text()`

## Key Primitives (Jaxpr)

| Python | Jaxpr Primitive |
|--------|-----------------|
| `+` | `add` |
| `*` | `mul` |
| `jnp.sin` | `sin` |
| `jnp.dot` | `dot_general` |
| `jnp.sum` | `reduce_sum` |
| `jax.lax.cond` | `cond` |

## Extraction API

```python
jaxpr = jax.make_jaxpr(fn)(*example_args)
lowered = jax.jit(fn).lower(*example_args)
hlo_text = lowered.as_text()
compiled = lowered.compile()  # For optimized HLO
```

## Notes

- Jaxpr requires concrete shapes for tracing
- `make_jaxpr` doesn't compile, just traces
- `.lower()` produces StableHLO (not legacy HLO)
- Multiple outputs become tuple in jaxpr
