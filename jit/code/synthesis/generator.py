"""
JAX Kernel Generator
Programmatically generates varied JAX functions for IR extraction
"""

import jax.numpy as jnp
import random
import string


class KernelGenerator:
    """Generate JAX kernels programmatically."""
    
    def __init__(self, seed=None):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        self.function_counter = 0
    
    def _get_unique_name(self, prefix):
        """Generate unique function name."""
        name = f"{prefix}_{self.function_counter}"
        self.function_counter += 1
        return name
    
    # ========================================================================
    # Arithmetic Operations
    # ========================================================================
    
    def generate_arithmetic(self, operation='add'):
        """
        Generate arithmetic operation kernel.
        
        Args:
            operation: One of 'add', 'sub', 'mul', 'div', 'mod'
        
        Returns:
            (function, test_args, metadata)
        """
        ops = {
            'add': ('+', 'addition'),
            'sub': ('-', 'subtraction'),
            'mul': ('*', 'multiplication'),
            'div': ('/', 'division'),
        }
        
        if operation not in ops:
            raise ValueError(f"Unknown operation: {operation}")
        
        op_symbol, op_name = ops[operation]
        name = self._get_unique_name(f'arith_{operation}')
        
        # Create function dynamically
        func_code = f"""
def {name}(x, y):
    \"\"\"Arithmetic {op_name}: x {op_symbol} y\"\"\"
    return x {op_symbol} y
"""
        
        namespace = {}
        exec(func_code, {'jnp': jnp}, namespace)
        func = namespace[name]
        
        # Test arguments
        test_args = (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
        
        metadata = {
            'category': 'arithmetic',
            'operation': operation,
            'complexity': 'simple'
        }
        
        return func, test_args, metadata
    
    # ========================================================================
    # Math Functions
    # ========================================================================
    
    def generate_math_function(self, func_name='sin'):
        """
        Generate math function kernel.
        
        Args:
            func_name: One of 'sin', 'cos', 'exp', 'log', 'tanh', 'sqrt'
        
        Returns:
            (function, test_args, metadata)
        """
        valid_funcs = ['sin', 'cos', 'exp', 'log', 'tanh', 'sqrt', 'abs']
        
        if func_name not in valid_funcs:
            raise ValueError(f"Unknown function: {func_name}")
        
        name = self._get_unique_name(f'math_{func_name}')
        
        func_code = f"""
def {name}(x):
    \"\"\"Math function: jnp.{func_name}(x)\"\"\"
    return jnp.{func_name}(x)
"""
        
        namespace = {}
        exec(func_code, {'jnp': jnp}, namespace)
        func = namespace[name]
        
        # Test arguments (positive for log, sqrt)
        if func_name in ['log', 'sqrt']:
            test_args = (jnp.array([1.0, 2.0, 3.0]),)
        else:
            test_args = (jnp.array([0.5, 1.0, 1.5]),)
        
        metadata = {
            'category': 'math',
            'function': func_name,
            'complexity': 'simple'
        }
        
        return func, test_args, metadata
    
    # ========================================================================
    # Array Operations
    # ========================================================================
    
    def generate_array_op(self, op_type='dot'):
        """
        Generate array operation kernel.
        
        Args:
            op_type: One of 'dot', 'matmul', 'sum', 'mean', 'transpose'
        
        Returns:
            (function, test_args, metadata)
        """
        name = self._get_unique_name(f'array_{op_type}')
        
        if op_type == 'dot':
            func_code = f"""
def {name}(x, y):
    \"\"\"Dot product\"\"\"
    return jnp.dot(x, y)
"""
            test_args = (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
        
        elif op_type == 'matmul':
            func_code = f"""
def {name}(A, B):
    \"\"\"Matrix multiplication\"\"\"
    return jnp.matmul(A, B)
"""
            test_args = (jnp.array([[1.0, 2.0], [3.0, 4.0]]), 
                        jnp.array([[5.0, 6.0], [7.0, 8.0]]))
        
        elif op_type == 'sum':
            func_code = f"""
def {name}(x):
    \"\"\"Sum reduction\"\"\"
    return jnp.sum(x)
"""
            test_args = (jnp.array([1.0, 2.0, 3.0, 4.0]),)
        
        elif op_type == 'mean':
            func_code = f"""
def {name}(x):
    \"\"\"Mean reduction\"\"\"
    return jnp.mean(x)
"""
            test_args = (jnp.array([1.0, 2.0, 3.0, 4.0]),)
        
        elif op_type == 'transpose':
            func_code = f"""
def {name}(A):
    \"\"\"Matrix transpose\"\"\"
    return A.T
"""
            test_args = (jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),)
        
        else:
            raise ValueError(f"Unknown array operation: {op_type}")
        
        namespace = {}
        exec(func_code, {'jnp': jnp}, namespace)
        func = namespace[name]
        
        metadata = {
            'category': 'array',
            'operation': op_type,
            'complexity': 'simple'
        }
        
        return func, test_args, metadata
    
    # ========================================================================
    # Control Flow
    # ========================================================================
    
    def generate_control_flow(self, flow_type='where'):
        """
        Generate control flow kernel.
        
        Args:
            flow_type: One of 'where', 'maximum', 'minimum'
        
        Returns:
            (function, test_args, metadata)
        """
        name = self._get_unique_name(f'control_{flow_type}')
        
        if flow_type == 'where':
            func_code = f"""
def {name}(x):
    \"\"\"Conditional: where(x > 0, x**2, -x)\"\"\"
    return jnp.where(x > 0, x ** 2, -x)
"""
            test_args = (jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]),)
        
        elif flow_type == 'maximum':
            func_code = f"""
def {name}(x, y):
    \"\"\"Element-wise maximum\"\"\"
    return jnp.maximum(x, y)
"""
            test_args = (jnp.array([1.0, 2.0, 3.0]), jnp.array([2.5, 1.5, 3.5]))
        
        elif flow_type == 'minimum':
            func_code = f"""
def {name}(x, y):
    \"\"\"Element-wise minimum\"\"\"
    return jnp.minimum(x, y)
"""
            test_args = (jnp.array([1.0, 2.0, 3.0]), jnp.array([2.5, 1.5, 3.5]))
        
        else:
            raise ValueError(f"Unknown control flow: {flow_type}")
        
        namespace = {}
        exec(func_code, {'jnp': jnp}, namespace)
        func = namespace[name]
        
        metadata = {
            'category': 'control_flow',
            'type': flow_type,
            'complexity': 'medium'
        }
        
        return func, test_args, metadata
    
    # ========================================================================
    # Combined/Complex Patterns
    # ========================================================================
    
    def generate_combined(self, pattern='linear'):
        """
        Generate combined/complex kernel.
        
        Args:
            pattern: One of 'linear', 'quadratic', 'sigmoid', 'softmax'
        
        Returns:
            (function, test_args, metadata)
        """
        name = self._get_unique_name(f'combined_{pattern}')
        
        if pattern == 'linear':
            func_code = f"""
def {name}(W, x, b):
    \"\"\"Linear layer: Wx + b\"\"\"
    return jnp.dot(W, x) + b
"""
            W = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            x = jnp.array([1.0, 2.0, 3.0])
            b = jnp.array([0.5, 1.5])
            test_args = (W, x, b)
        
        elif pattern == 'quadratic':
            func_code = f"""
def {name}(x):
    \"\"\"Quadratic: x^2 + 2x + 1\"\"\"
    return x ** 2 + 2 * x + 1
"""
            test_args = (jnp.array([1.0, 2.0, 3.0]),)
        
        elif pattern == 'sigmoid':
            func_code = f"""
def {name}(x):
    \"\"\"Sigmoid: 1 / (1 + exp(-x))\"\"\"
    return 1.0 / (1.0 + jnp.exp(-x))
"""
            test_args = (jnp.array([0.0, 1.0, 2.0]),)
        
        elif pattern == 'softmax':
            func_code = f"""
def {name}(x):
    \"\"\"Softmax: exp(x) / sum(exp(x))\"\"\"
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x)
"""
            test_args = (jnp.array([1.0, 2.0, 3.0]),)
        
        elif pattern == 'relu':
            func_code = f"""
def {name}(x):
    \"\"\"ReLU: max(0, x)\"\"\"
    return jnp.maximum(0, x)
"""
            test_args = (jnp.array([-1.0, 0.0, 1.0, 2.0]),)
        
        elif pattern == 'mse':
            func_code = f"""
def {name}(pred, target):
    \"\"\"Mean squared error\"\"\"
    return jnp.mean((pred - target) ** 2)
"""
            test_args = (jnp.array([1.0, 2.0, 3.0]), jnp.array([1.5, 2.5, 3.5]))
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        namespace = {}
        exec(func_code, {'jnp': jnp}, namespace)
        func = namespace[name]
        
        metadata = {
            'category': 'combined',
            'pattern': pattern,
            'complexity': 'medium'
        }
        
        return func, test_args, metadata
    
    # ========================================================================
    # Generate Random Kernel
    # ========================================================================
    
    def generate_random(self):
        """Generate a random kernel from all categories."""
        categories = [
            ('arithmetic', ['add', 'sub', 'mul', 'div']),
            ('math', ['sin', 'cos', 'exp', 'tanh', 'sqrt']),
            ('array', ['dot', 'matmul', 'sum', 'mean', 'transpose']),
            ('control_flow', ['where', 'maximum', 'minimum']),
            ('combined', ['linear', 'quadratic', 'sigmoid', 'softmax', 'relu', 'mse'])
        ]
        
        category, options = random.choice(categories)
        option = random.choice(options)
        
        if category == 'arithmetic':
            return self.generate_arithmetic(option)
        elif category == 'math':
            return self.generate_math_function(option)
        elif category == 'array':
            return self.generate_array_op(option)
        elif category == 'control_flow':
            return self.generate_control_flow(option)
        elif category == 'combined':
            return self.generate_combined(option)


# Test the generator
if __name__ == "__main__":
    print("=" * 80)
    print("JAX Kernel Generator - Test Suite")
    print("=" * 80)
    
    generator = KernelGenerator(seed=42)
    
    # Test arithmetic
    print("\n1. Arithmetic Operations")
    print("-" * 80)
    for op in ['add', 'sub', 'mul', 'div']:
        func, args, meta = generator.generate_arithmetic(op)
        result = func(*args)
        print(f"  ✓ {func.__name__}: {meta['operation']} -> {result[0]:.2f}")
    
    # Test math functions
    print("\n2. Math Functions")
    print("-" * 80)
    for fn in ['sin', 'cos', 'exp', 'tanh']:
        func, args, meta = generator.generate_math_function(fn)
        result = func(*args)
        print(f"  ✓ {func.__name__}: {meta['function']} -> {result[0]:.4f}")
    
    # Test array operations
    print("\n3. Array Operations")
    print("-" * 80)
    for op in ['dot', 'sum', 'mean']:
        func, args, meta = generator.generate_array_op(op)
        result = func(*args)
        print(f"  ✓ {func.__name__}: {meta['operation']} -> {result}")
    
    # Test control flow
    print("\n4. Control Flow")
    print("-" * 80)
    for cf in ['where', 'maximum']:
        func, args, meta = generator.generate_control_flow(cf)
        result = func(*args)
        print(f"  ✓ {func.__name__}: {meta['type']} -> shape {result.shape}")
    
    # Test combined
    print("\n5. Combined Patterns")
    print("-" * 80)
    for pat in ['quadratic', 'sigmoid', 'relu']:
        func, args, meta = generator.generate_combined(pat)
        result = func(*args)
        print(f"  ✓ {func.__name__}: {meta['pattern']} -> {result}")
    
    # Test random generation
    print("\n6. Random Generation")
    print("-" * 80)
    for i in range(5):
        func, args, meta = generator.generate_random()
        result = func(*args)
        print(f"  ✓ {func.__name__}: {meta['category']} -> OK")
    
    print("\n" + "=" * 80)
    print("All generator tests passed!")
    print("=" * 80)
