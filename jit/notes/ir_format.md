# JAX HLO IR Format

## Overview

JAX uses XLA's HLO (High-Level Optimizer) intermediate representation. This document describes the HLO format generated from JAX-JIT compiled functions.

## HLO Module Structure

Generated HLO consists of:
1. **Module declaration**: `HloModule jit_{function_name}`
2. **Main entry point**: `func.func public @main` with input/output signature
3. **Function implementations**: Private functions for the actual computation
4. **Backward pass**: Gradient computation functions (when autodiff is enabled)

## Example HLO IR

```hlo
module @jit_elementwise_add attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>) {
    %0 = call @elementwise_add(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
  func.func private @elementwise_add(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
```

## Key HLO Operations

### Basic Arithmetic
- Addition: `stablehlo.add`
- Multiplication: `stablehlo.mul`
- Division: `stablehlo.div`
- Subtraction: `stablehlo.sub`

### Element-wise Operations
- `stablehlo.sqrt` - Square root
- `stablehlo.abs` - Absolute value
- `stablehlo.sin`, `stablehlo.cos` - Trigonometric functions
- `stablehlo.exp`, `stablehlo.log` - Exponential and logarithm

### Array Operations
- `stablehlo.broadcast_in_dim` - Broadcasting
- `stablehlo.reduce` - Reduction operations
- `stablehlo.dot_general` - Matrix/vector products
- `stablehlo.reshape` - Reshaping arrays

### Control Flow
- `stablehlo.select` / `jnp.where` - Conditional selection
- `stablehlo.while` - While loops
- Custom operations via `call` to other functions

## Optimized HLO

After XLA optimization passes, you get optimized HLO with:
- **Fusion**: Multiple operations combined into single kernels
- **Memory optimization**: In-place operations where possible
- **Scheduling**: Explicit computation order
- **Metadata**: Source location information preserved

Example optimized HLO:
```hlo
HloModule jit_elementwise_add, is_scheduled=true, entry_computation_layout={(f32[3]{0}, f32[3]{0})->f32[3]{0}}

%fused_computation (param_0: f32[3], param_1: f32[3]) -> f32[3] {
  %param_0 = f32[3]{0} parameter(0)
  %param_1 = f32[3]{0} parameter(1)
  ROOT %add = f32[3]{0} add(%param_0, %param_1)
}

ENTRY %main (a: f32[3], b: f32[3]) -> f32[3] {
  %a = f32[3]{0} parameter(0)
  %b = f32[3]{0} parameter(1)
  ROOT %fusion = f32[3]{0} fusion(%a, %b), kind=kLoop, calls=%fused_computation
}
```

## Backward Pass (Gradient Computation)

JAX automatically generates gradient computation via autodiff. The backward pass:
- Computes partial derivatives for all operations
- Uses reverse-mode autodiff (backpropagation)
- Appears as additional functions in the HLO module

Example with gradients:
```hlo
# ===== BACKWARD PASS (GRADIENT) =====

module @jit_func_with_grad {
  func.func public @main(%arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
    %0 = call @forward_function(%arg0)
    %1 = call @gradient_function(%arg0)
    return %0, %1
  }
  // ... gradient computation functions
}
```

## Type System

- `tensor<3xf32>` - Float32 tensor of shape [3]
- `tensor<f32>` - Scalar float32
- `tensor<3x4xf32>` - 2D tensor (matrix)
- `{0}` - Layout annotation (memory order)

## Metadata

HLO preserves source information:
- `FileNames` - Source file paths
- `FunctionNames` - Function names
- `FileLocations` - Line/column information
- `stack_frame_id` - Call stack for debugging

## Benefits of HLO IR

1. **Hardware-Agnostic**: Can target CPU, GPU, TPU
2. **Optimizable**: XLA applies sophisticated optimizations
3. **Differentiable**: Automatic gradient computation
4. **Explicit**: Shows computation graph clearly
5. **Well-Documented**: Part of TensorFlow/JAX ecosystem
