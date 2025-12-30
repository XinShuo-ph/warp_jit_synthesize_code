#!/usr/bin/env python3
"""
IR Extractor: Extract Jaxpr and StableHLO IR from JAX functions

This module provides utilities to extract intermediate representations (IR)
from JAX functions for training data generation.
"""
import jax
import jax.numpy as jnp
from jax import make_jaxpr, jit
from typing import Callable, Dict, List, Any, Tuple
import json


class IRExtractor:
    """Extract IR from JAX functions"""
    
    def __init__(self, ir_type: str = "both"):
        """
        Initialize IR extractor
        
        Args:
            ir_type: Type of IR to extract ("jaxpr", "stablehlo", or "both")
        """
        if ir_type not in ["jaxpr", "stablehlo", "both"]:
            raise ValueError(f"ir_type must be 'jaxpr', 'stablehlo', or 'both', got {ir_type}")
        self.ir_type = ir_type
    
    def extract_jaxpr(self, func: Callable, *args) -> str:
        """
        Extract Jaxpr from a function
        
        Args:
            func: JAX function to extract IR from
            *args: Example inputs (with correct shapes/dtypes)
        
        Returns:
            Jaxpr as string
        """
        jaxpr = make_jaxpr(func)(*args)
        return str(jaxpr)
    
    def extract_stablehlo(self, func: Callable, *args) -> str:
        """
        Extract StableHLO from a function
        
        Args:
            func: JAX function to extract IR from
            *args: Example inputs (with correct shapes/dtypes)
        
        Returns:
            StableHLO (MLIR) as string
        """
        lowered = jit(func).lower(*args)
        return lowered.as_text()
    
    def extract(self, func: Callable, *args) -> Dict[str, str]:
        """
        Extract IR from a function (based on ir_type setting)
        
        Args:
            func: JAX function to extract IR from
            *args: Example inputs (with correct shapes/dtypes)
        
        Returns:
            Dictionary with IR types as keys and IR strings as values
        """
        result = {}
        
        if self.ir_type in ["jaxpr", "both"]:
            result["jaxpr"] = self.extract_jaxpr(func, *args)
        
        if self.ir_type in ["stablehlo", "both"]:
            result["stablehlo"] = self.extract_stablehlo(func, *args)
        
        return result
    
    def create_training_pair(
        self,
        python_code: str,
        func: Callable,
        example_inputs: List[Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a complete Python→IR training pair
        
        Args:
            python_code: Source code of the function (as string)
            func: The actual function object
            example_inputs: List of example inputs for IR extraction
            metadata: Optional metadata dict (function name, description, etc.)
        
        Returns:
            Training pair dictionary with Python code, IR, and metadata
        """
        # Extract IR
        ir_dict = self.extract(func, *example_inputs)
        
        # Get input information
        input_info = []
        for inp in example_inputs:
            if isinstance(inp, jnp.ndarray):
                input_info.append({
                    "shape": list(inp.shape),
                    "dtype": str(inp.dtype)
                })
            else:
                input_info.append({
                    "type": type(inp).__name__,
                    "value": str(inp)
                })
        
        # Build training pair
        pair = {
            "python_code": python_code,
            "input_info": input_info,
            **ir_dict
        }
        
        # Add metadata if provided
        if metadata:
            pair["metadata"] = metadata
        
        return pair
    
    def save_pair(self, pair: Dict[str, Any], filepath: str):
        """Save a training pair to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(pair, f, indent=2)
    
    def save_pairs(self, pairs: List[Dict[str, Any]], filepath: str):
        """Save multiple training pairs to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(pairs, f, indent=2)


def extract_from_function(
    func: Callable,
    example_inputs: List[Any],
    ir_type: str = "both"
) -> Dict[str, str]:
    """
    Convenience function to extract IR from a function
    
    Args:
        func: JAX function
        example_inputs: Example inputs
        ir_type: "jaxpr", "stablehlo", or "both"
    
    Returns:
        Dictionary with IR
    """
    extractor = IRExtractor(ir_type=ir_type)
    return extractor.extract(func, *example_inputs)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("IR Extractor Test")
    print("=" * 70)
    
    # Define test functions
    def add_vectors(x, y):
        return x + y
    
    def complex_math(x):
        return jnp.sum(jnp.sin(x) ** 2)
    
    # Create extractor
    extractor = IRExtractor(ir_type="both")
    
    # Test 1: Simple addition
    print("\n1. Extracting IR from add_vectors")
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    ir = extractor.extract(add_vectors, x, y)
    print(f"   Jaxpr length: {len(ir['jaxpr'])} chars")
    print(f"   StableHLO length: {len(ir['stablehlo'])} chars")
    
    # Test 2: Create training pair
    print("\n2. Creating training pair")
    python_code = """def add_vectors(x, y):
    return x + y"""
    
    pair = extractor.create_training_pair(
        python_code=python_code,
        func=add_vectors,
        example_inputs=[x, y],
        metadata={"function_name": "add_vectors", "category": "arithmetic"}
    )
    print(f"   Keys in pair: {list(pair.keys())}")
    print(f"   Input info: {pair['input_info']}")
    
    # Test 3: Multiple functions
    print("\n3. Extracting from multiple functions")
    functions = [
        (add_vectors, [x, y]),
        (complex_math, [x])
    ]
    
    for i, (func, inputs) in enumerate(functions, 1):
        ir = extractor.extract(func, *inputs)
        print(f"   Function {i}: Jaxpr={len(ir['jaxpr'])} chars, "
              f"StableHLO={len(ir['stablehlo'])} chars")
    
    print("\n" + "=" * 70)
    print("SUCCESS: IR Extractor working correctly!")
    print("=" * 70)
