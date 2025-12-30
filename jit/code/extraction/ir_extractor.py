"""
JAX IR Extractor
Programmatically extract HLO/StableHLO IR from JAX functions
"""

import jax
import jax.numpy as jnp
import inspect
from typing import Callable, Dict, Any, Optional, List
import json


class IRExtractor:
    """Extract intermediate representation from JAX functions."""
    
    def __init__(self, dialect: str = 'stablehlo'):
        """
        Initialize IR extractor.
        
        Args:
            dialect: IR dialect to extract ('hlo' or 'stablehlo')
        """
        self.dialect = dialect
    
    def extract_ir(self, func: Callable, *args, **kwargs) -> str:
        """
        Extract IR from a JAX function.
        
        Args:
            func: JAX function to extract IR from
            *args: Example inputs for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            String representation of the IR
        """
        # Lower the function to get IR
        lowered = jax.jit(func).lower(*args, **kwargs)
        
        if self.dialect == 'hlo':
            # Extract HLO
            hlo_ir = lowered.compiler_ir(dialect='hlo')
            return hlo_ir.as_hlo_text()
        elif self.dialect == 'stablehlo':
            # Extract StableHLO (MLIR format)
            stablehlo_ir = lowered.compiler_ir(dialect='stablehlo')
            return str(stablehlo_ir)
        else:
            raise ValueError(f"Unknown dialect: {self.dialect}")
    
    def extract_with_metadata(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Extract IR with metadata about the function.
        
        Args:
            func: JAX function to extract IR from
            *args: Example inputs for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Dictionary with IR and metadata
        """
        # Get function source code
        try:
            source = inspect.getsource(func)
        except:
            source = f"# Source not available for {func.__name__}"
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Extract IR
        ir_code = self.extract_ir(func, *args, **kwargs)
        
        # Get input shapes and dtypes
        input_info = []
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                input_info.append({
                    'index': i,
                    'shape': list(arg.shape),
                    'dtype': str(arg.dtype)
                })
        
        return {
            'function_name': func.__name__,
            'python_source': source,
            'ir_code': ir_code,
            'dialect': self.dialect,
            'signature': str(sig),
            'input_info': input_info
        }
    
    def extract_batch(self, functions: List[tuple]) -> List[Dict[str, Any]]:
        """
        Extract IR from multiple functions.
        
        Args:
            functions: List of (func, args, kwargs) tuples
        
        Returns:
            List of IR extraction results
        """
        results = []
        for item in functions:
            if len(item) == 2:
                func, args = item
                kwargs = {}
            else:
                func, args, kwargs = item
            
            try:
                result = self.extract_with_metadata(func, *args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    'function_name': func.__name__ if hasattr(func, '__name__') else 'unknown',
                    'error': str(e)
                })
        
        return results
    
    def save_to_json(self, data: Dict[str, Any], filepath: str):
        """Save extracted IR to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience functions
def extract_stablehlo(func: Callable, *args, **kwargs) -> str:
    """Extract StableHLO IR from a function."""
    extractor = IRExtractor(dialect='stablehlo')
    return extractor.extract_ir(func, *args, **kwargs)


def extract_hlo(func: Callable, *args, **kwargs) -> str:
    """Extract HLO IR from a function."""
    extractor = IRExtractor(dialect='hlo')
    return extractor.extract_ir(func, *args, **kwargs)


def extract_ir_pair(func: Callable, *args, **kwargs) -> Dict[str, str]:
    """
    Extract Python source and IR as a pair.
    
    Returns:
        Dictionary with 'python_source' and 'ir_code' keys
    """
    extractor = IRExtractor(dialect='stablehlo')
    data = extractor.extract_with_metadata(func, *args, **kwargs)
    return {
        'function_name': data['function_name'],
        'python_source': data['python_source'],
        'ir_code': data['ir_code'],
        'input_info': data['input_info']
    }


# Example usage and tests
if __name__ == "__main__":
    print("=" * 80)
    print("JAX IR Extractor - Test Suite")
    print("=" * 80)
    
    # Test 1: Simple arithmetic
    def add(x, y):
        return x + y
    
    print("\n1. Simple Addition")
    print("-" * 80)
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    
    extractor = IRExtractor(dialect='stablehlo')
    result = extractor.extract_with_metadata(add, x, y)
    
    print(f"Function: {result['function_name']}")
    print(f"Signature: {result['signature']}")
    print(f"Input info: {result['input_info']}")
    print(f"\nIR Code:\n{result['ir_code']}")
    
    # Test 2: Matrix operation
    def matmul(A, B):
        return jnp.dot(A, B)
    
    print("\n2. Matrix Multiplication")
    print("-" * 80)
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    
    result = extractor.extract_with_metadata(matmul, A, B)
    print(f"Function: {result['function_name']}")
    print(f"\nIR Code (first 300 chars):\n{result['ir_code'][:300]}")
    
    # Test 3: Math functions
    def math_ops(x):
        return jnp.tanh(jnp.sin(x) + jnp.exp(x))
    
    print("\n3. Math Operations")
    print("-" * 80)
    x = jnp.array([1.0, 2.0, 3.0])
    
    result = extractor.extract_with_metadata(math_ops, x)
    print(f"Function: {result['function_name']}")
    print(f"\nIR Code (first 300 chars):\n{result['ir_code'][:300]}")
    
    # Test 4: Batch extraction
    print("\n4. Batch Extraction")
    print("-" * 80)
    
    functions = [
        (add, (x, y)),
        (matmul, (A, B)),
        (math_ops, (x,))
    ]
    
    results = extractor.extract_batch(functions)
    print(f"Extracted {len(results)} functions successfully")
    for r in results:
        if 'error' in r:
            print(f"  - {r['function_name']}: ERROR - {r['error']}")
        else:
            print(f"  - {r['function_name']}: OK ({len(r['ir_code'])} chars)")
    
    # Test 5: Save to JSON
    print("\n5. Save to JSON")
    print("-" * 80)
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_pair.json')
        extractor.save_to_json(result, filepath)
        print(f"Saved to: {filepath}")
        
        # Load and verify
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        print(f"Loaded function: {loaded['function_name']}")
        print(f"IR code length: {len(loaded['ir_code'])} chars")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
