# Kernel Type Inventory

## Overview
The CPU pipeline (branch 12c4) implements 6 kernel categories with various operations. This document catalogs each type and notes CUDA-specific considerations.

## Kernel Types

### 1. Arithmetic
**Description**: Chains of arithmetic operations (add, sub, mul, div, min, max)

**Operations**:
- Binary: `+`, `-`, `*`, `/`, `wp.min()`, `wp.max()`
- Unary: `-`, `wp.abs()`
- Chained: 1-4 operations in sequence

**Example**:
```python
@wp.kernel
def arith_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    var_0 = a[tid] + b[tid]
    var_1 = wp.max(var_0, b[tid])
    c[tid] = var_1
```

**CUDA Considerations**:
- ✓ Direct translation, no special handling needed
- Uses standard arithmetic operations available on GPU
- Thread-independent computations

---

### 2. Math
**Description**: Mathematical functions (sin, cos, exp, log, sqrt, abs)

**Operations**:
- Trigonometric: `wp.sin()`, `wp.cos()`
- Exponential: `wp.exp()`, `wp.log()`
- Other: `wp.sqrt()`, `wp.abs()`
- Chained: 1-3 functions in sequence

**Example**:
```python
@wp.kernel
def math_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.sin(wp.sqrt(wp.abs(a[tid])))
```

**CUDA Considerations**:
- ✓ All operations map to CUDA math library
- GPU has hardware acceleration for trig functions
- May benefit from fast math flags

---

### 3. Vector
**Description**: Vector operations on wp.vec2, wp.vec3, wp.vec4

**Operations**:
- `wp.dot()` - dot product (returns scalar)
- `wp.cross()` - cross product (vec3 only)
- `wp.length()` - vector length (returns scalar)
- `wp.normalize()` - normalize vector

**Example**:
```python
@wp.kernel
def vec_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])
```

**CUDA Considerations**:
- ✓ Warp vectors map to CUDA vector types
- Efficient GPU implementation
- Consider vec4 for better memory alignment

---

### 4. Matrix
**Description**: Matrix operations on wp.mat22, wp.mat33, wp.mat44

**Operations**:
- Matrix-vector multiply: `mat * vec`
- Matrix-matrix multiply: `mat * mat`
- Transpose: `wp.transpose(mat)`

**Example**:
```python
@wp.kernel
def mat_kernel(m: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]
```

**CUDA Considerations**:
- ✓ Matrix operations are element-wise per thread
- Could benefit from shared memory for larger operations
- Consider using tensor cores for larger matrices (future)

---

### 5. Control Flow
**Description**: Conditionals (if/else) and loops (for)

**Patterns**:
- `clamp`: Conditional clamping between min/max
- `abs_diff`: Absolute difference using if
- `step`: Step function (threshold)
- `loop_sum`: For loop accumulation
- `loop_product`: For loop multiplication

**Example**:
```python
@wp.kernel
def ctrl_kernel(a: wp.array(dtype=float), threshold: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    if a[tid] > threshold:
        out[tid] = 1.0
    else:
        out[tid] = 0.0
```

**CUDA Considerations**:
- ⚠️ Branch divergence can impact performance
- Threads in same warp with different branches serialize
- Small loops unroll well on GPU
- Consider warp-uniform branches for better performance

---

### 6. Atomic
**Description**: Atomic operations for reductions

**Operations**:
- `wp.atomic_add()` - atomic addition
- `wp.atomic_min()` - atomic minimum
- `wp.atomic_max()` - atomic maximum

**Example**:
```python
@wp.kernel
def atom_kernel(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])
```

**CUDA Considerations**:
- ✓✓ Atomics are critical for CUDA
- GPU atomics are highly optimized
- Potential for contention on same memory location
- Could add more atomic ops: `wp.atomic_sub()`, `wp.atomic_and()`, etc.

---

## CUDA-Specific Patterns to Add

These patterns are GPU-specific and should be added for comprehensive CUDA coverage:

### 7. Shared Memory (NEW)
**Description**: Using shared memory for thread cooperation

**Example**:
```python
@wp.kernel
def shared_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    # Shared memory pattern
    # (requires Warp tile API)
```

**Status**: TODO - needs implementation

---

### 8. Thread Synchronization (NEW)
**Description**: Block-level synchronization primitives

**Operations**:
- `wp.synchronize()` - block barrier
- Warp shuffle operations

**Status**: TODO - needs implementation

---

## Summary Statistics

| Category | Count | CUDA Ready | Notes |
|----------|-------|------------|-------|
| Arithmetic | 1 | ✓ | Direct translation |
| Math | 1 | ✓ | Hardware accelerated |
| Vector | 1 | ✓ | Efficient on GPU |
| Matrix | 1 | ✓ | Element-wise operations |
| Control Flow | 1 | ⚠️ | Watch branch divergence |
| Atomic | 1 | ✓✓ | Critical for GPU |
| **Total** | **6** | **6/6** | All types supported |

## Next Steps for CUDA

1. **M2**: Adapt IR extractor to generate CUDA code for all 6 types
2. **M3**: Test each type individually with CUDA backend
3. **M3**: Add CUDA-specific patterns (shared memory, sync)
4. **M4**: Create validation tests for GPU execution
5. **M5**: Performance comparison CPU vs CUDA
