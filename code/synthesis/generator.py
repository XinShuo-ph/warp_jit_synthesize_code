"""
Kernel Generator - Programmatically generates diverse Warp kernels

This module creates varied Python kernels for training data synthesis.
"""

import random
import string
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OpType(Enum):
    """Types of operations we can generate"""
    ARITHMETIC = "arithmetic"
    VECTOR = "vector"
    TRIGONOMETRY = "trigonometry"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    ATOMIC = "atomic"


@dataclass
class KernelSpec:
    """Specification for a generated kernel"""
    name: str
    op_type: OpType
    num_inputs: int
    num_outputs: int
    has_scalar_param: bool = False
    complexity: int = 1  # 1=simple, 2=medium, 3=complex


class KernelGenerator:
    """Generates diverse Warp kernel code"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.kernel_count = 0
    
    def generate_kernel(self, spec: KernelSpec) -> str:
        """
        Generate a complete kernel based on specification
        
        Returns:
            Python code as string
        """
        if spec.op_type == OpType.ARITHMETIC:
            return self._generate_arithmetic(spec)
        elif spec.op_type == OpType.VECTOR:
            return self._generate_vector(spec)
        elif spec.op_type == OpType.TRIGONOMETRY:
            return self._generate_trig(spec)
        elif spec.op_type == OpType.CONDITIONAL:
            return self._generate_conditional(spec)
        elif spec.op_type == OpType.LOOP:
            return self._generate_loop(spec)
        elif spec.op_type == OpType.ATOMIC:
            return self._generate_atomic(spec)
        else:
            raise ValueError(f"Unknown op type: {spec.op_type}")
    
    def _generate_arithmetic(self, spec: KernelSpec) -> str:
        """Generate arithmetic kernel"""
        inputs = [f"a{i}" for i in range(spec.num_inputs)]
        outputs = [f"b{i}" for i in range(spec.num_outputs)]
        
        # Build signature
        params = []
        for inp in inputs:
            params.append(f"{inp}: wp.array(dtype=float)")
        for out in outputs:
            params.append(f"{out}: wp.array(dtype=float)")
        
        if spec.has_scalar_param:
            params.append("scale: float")
        
        signature = f"@wp.kernel\ndef {spec.name}({', '.join(params)}):\n"
        
        # Build body
        body = "    tid = wp.tid()\n"
        
        # Generate expressions based on complexity
        ops = ['+', '-', '*']
        for i, out in enumerate(outputs):
            if spec.complexity == 1:
                # Simple: b = a * scale
                if spec.has_scalar_param:
                    expr = f"{inputs[0]}[tid] * scale"
                else:
                    expr = f"{inputs[0]}[tid] * 2.0"
            elif spec.complexity == 2:
                # Medium: combine 2 inputs
                if len(inputs) >= 2:
                    op = random.choice(ops)
                    expr = f"{inputs[0]}[tid] {op} {inputs[1]}[tid]"
                    if spec.has_scalar_param:
                        expr = f"({expr}) * scale"
                else:
                    expr = f"{inputs[0]}[tid] * 2.0 + 1.0"
            else:
                # Complex: combine all inputs
                expr_parts = [f"{inp}[tid]" for inp in inputs[:3]]
                expr = f"({expr_parts[0]} + {expr_parts[1]}) * {expr_parts[2] if len(expr_parts) > 2 else '2.0'}"
            
            body += f"    {out}[tid] = {expr}\n"
        
        return signature + body
    
    def _generate_vector(self, spec: KernelSpec) -> str:
        """Generate vector operations kernel"""
        signature = f"@wp.kernel\ndef {spec.name}("
        signature += "positions: wp.array(dtype=wp.vec3), "
        signature += "velocities: wp.array(dtype=wp.vec3), "
        signature += "forces: wp.array(dtype=wp.vec3)"
        if spec.has_scalar_param:
            signature += ", dt: float"
        signature += "):\n"
        
        body = "    tid = wp.tid()\n"
        body += "    pos = positions[tid]\n"
        body += "    vel = velocities[tid]\n"
        body += "    \n"
        
        if spec.complexity == 1:
            # Simple: spring force
            body += "    dist = wp.length(pos)\n"
            body += "    spring = wp.normalize(pos) * (-1.0 * dist)\n"
            body += "    forces[tid] = spring\n"
        elif spec.complexity == 2:
            # Medium: spring + damping
            body += "    dist = wp.length(pos)\n"
            body += "    spring = wp.normalize(pos) * (-10.0 * dist)\n"
            body += "    damping = vel * (-0.5)\n"
            body += "    forces[tid] = spring + damping\n"
        else:
            # Complex: spring + damping + velocity-dependent
            body += "    dist = wp.length(pos)\n"
            body += "    vel_mag = wp.length(vel)\n"
            body += "    spring = wp.normalize(pos) * (-10.0 * dist)\n"
            body += "    damping = vel * (-0.5 * vel_mag)\n"
            body += "    forces[tid] = spring + damping\n"
        
        return signature + body
    
    def _generate_trig(self, spec: KernelSpec) -> str:
        """Generate trigonometric functions kernel"""
        signature = f"@wp.kernel\ndef {spec.name}("
        signature += "x: wp.array(dtype=float), "
        signature += "result: wp.array(dtype=float)"
        if spec.has_scalar_param:
            signature += ", freq: float"
        signature += "):\n"
        
        body = "    tid = wp.tid()\n"
        body += "    val = x[tid]\n"
        
        trig_funcs = ['sin', 'cos', 'tan']
        
        if spec.complexity == 1:
            func = random.choice(trig_funcs[:2])  # sin or cos
            if spec.has_scalar_param:
                body += f"    result[tid] = wp.{func}(val * freq)\n"
            else:
                body += f"    result[tid] = wp.{func}(val)\n"
        elif spec.complexity == 2:
            # Combine two trig functions
            if spec.has_scalar_param:
                body += f"    result[tid] = wp.sin(val * freq) * wp.cos(val * freq * 2.0)\n"
            else:
                body += f"    result[tid] = wp.sin(val) + wp.cos(val * 2.0)\n"
        else:
            # Complex expression
            body += f"    s = wp.sin(val)\n"
            body += f"    c = wp.cos(val)\n"
            body += f"    result[tid] = s * s + c * c\n"  # Should be ~1
        
        return signature + body
    
    def _generate_conditional(self, spec: KernelSpec) -> str:
        """Generate conditional kernel"""
        signature = f"@wp.kernel\ndef {spec.name}("
        signature += "values: wp.array(dtype=float), "
        signature += "categories: wp.array(dtype=int), "
        signature += "results: wp.array(dtype=float)"
        signature += "):\n"
        
        body = "    tid = wp.tid()\n"
        body += "    val = values[tid]\n"
        body += "    \n"
        
        if spec.complexity == 1:
            # Simple if-else
            body += "    if val < 0.0:\n"
            body += "        categories[tid] = 0\n"
            body += "        results[tid] = -val\n"
            body += "    else:\n"
            body += "        categories[tid] = 1\n"
            body += "        results[tid] = val\n"
        elif spec.complexity == 2:
            # Three-way conditional
            body += "    if val < -1.0:\n"
            body += "        categories[tid] = 0\n"
            body += "        results[tid] = val * val\n"
            body += "    elif val < 1.0:\n"
            body += "        categories[tid] = 1\n"
            body += "        results[tid] = val\n"
            body += "    else:\n"
            body += "        categories[tid] = 2\n"
            body += "        results[tid] = wp.sqrt(val)\n"
        else:
            # Complex nested conditionals
            body += "    if val < 0.0:\n"
            body += "        if val < -5.0:\n"
            body += "            categories[tid] = 0\n"
            body += "            results[tid] = val * val\n"
            body += "        else:\n"
            body += "            categories[tid] = 1\n"
            body += "            results[tid] = -val\n"
            body += "    else:\n"
            body += "        if val > 5.0:\n"
            body += "            categories[tid] = 2\n"
            body += "            results[tid] = wp.sqrt(val)\n"
            body += "        else:\n"
            body += "            categories[tid] = 3\n"
            body += "            results[tid] = val * 2.0\n"
        
        return signature + body
    
    def _generate_loop(self, spec: KernelSpec) -> str:
        """Generate loop kernel"""
        signature = f"@wp.kernel\ndef {spec.name}("
        signature += "matrix: wp.array2d(dtype=float), "
        signature += "result: wp.array(dtype=float)"
        signature += "):\n"
        
        body = "    tid = wp.tid()\n"
        body += "    total = float(0.0)\n"
        body += "    \n"
        
        if spec.complexity == 1:
            # Simple accumulation
            body += "    for j in range(matrix.shape[1]):\n"
            body += "        total = total + matrix[tid, j]\n"
        elif spec.complexity == 2:
            # Conditional accumulation
            body += "    for j in range(matrix.shape[1]):\n"
            body += "        val = matrix[tid, j]\n"
            body += "        if val > 0.0:\n"
            body += "            total = total + val\n"
        else:
            # Complex: weighted sum
            body += "    for j in range(matrix.shape[1]):\n"
            body += "        weight = float(j + 1)\n"
            body += "        total = total + matrix[tid, j] * weight\n"
        
        body += "    \n"
        body += "    result[tid] = total\n"
        
        return signature + body
    
    def _generate_atomic(self, spec: KernelSpec) -> str:
        """Generate atomic operations kernel"""
        signature = f"@wp.kernel\ndef {spec.name}("
        signature += "values: wp.array(dtype=float), "
        signature += "result: wp.array(dtype=float)"
        if spec.has_scalar_param:
            signature += ", threshold: float"
        signature += "):\n"
        
        body = "    tid = wp.tid()\n"
        body += "    val = values[tid]\n"
        body += "    \n"
        
        threshold = "threshold" if spec.has_scalar_param else "0.0"
        
        if spec.complexity == 1:
            # Simple atomic add
            body += f"    if val > {threshold}:\n"
            body += "        wp.atomic_add(result, 0, val)\n"
        else:
            # Multiple atomic operations
            body += f"    if val > {threshold}:\n"
            body += "        wp.atomic_add(result, 0, val)\n"
            body += "        wp.atomic_add(result, 1, 1.0)\n"
        
        return signature + body
    
    def generate_random_spec(self) -> KernelSpec:
        """Generate a random kernel specification"""
        op_type = random.choice(list(OpType))
        complexity = random.randint(1, 2)  # Mostly simple/medium
        
        if op_type == OpType.ARITHMETIC:
            num_inputs = random.randint(1, 3)
            num_outputs = random.randint(1, 2)
            has_scalar = random.random() < 0.5
        elif op_type == OpType.VECTOR:
            num_inputs = 2
            num_outputs = 1
            has_scalar = random.random() < 0.3
        elif op_type in [OpType.LOOP, OpType.ATOMIC, OpType.TRIGONOMETRY, OpType.CONDITIONAL]:
            num_inputs = 1
            num_outputs = 1
            has_scalar = random.random() < 0.3
        else:
            num_inputs = 1
            num_outputs = 1
            has_scalar = False
        
        name = f"kernel_{self.kernel_count}"
        self.kernel_count += 1
        
        return KernelSpec(
            name=name,
            op_type=op_type,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            has_scalar_param=has_scalar,
            complexity=complexity
        )


if __name__ == "__main__":
    # Test generator
    generator = KernelGenerator(seed=42)
    
    print("KERNEL GENERATOR TEST")
    print("=" * 80)
    
    # Generate one of each type
    specs = [
        KernelSpec("test_arith", OpType.ARITHMETIC, 2, 1, True, 2),
        KernelSpec("test_vec", OpType.VECTOR, 2, 1, False, 1),
        KernelSpec("test_trig", OpType.TRIGONOMETRY, 1, 1, True, 1),
        KernelSpec("test_cond", OpType.CONDITIONAL, 1, 2, False, 2),
        KernelSpec("test_loop", OpType.LOOP, 1, 1, False, 1),
        KernelSpec("test_atomic", OpType.ATOMIC, 1, 1, True, 1),
    ]
    
    for spec in specs:
        print(f"\n{spec.op_type.value.upper()}: (complexity={spec.complexity})")
        print("-" * 80)
        code = generator.generate_kernel(spec)
        print(code)
    
    print("\n" + "=" * 80)
    print("✓ Generator test complete")
