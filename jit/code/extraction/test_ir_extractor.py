"""Tests for JAX IR extractor."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent))

from ir_extractor import extract_ir, extract_ir_pair, extract_vmap_ir, ExtractedIR


class TestExtractIR:
    """Tests for extract_ir function."""
    
    def test_simple_add(self):
        """Test IR extraction for simple addition."""
        def add(a, b):
            return a + b
        
        a = jnp.ones((64,), dtype=jnp.float32)
        b = jnp.ones((64,), dtype=jnp.float32)
        
        ir = extract_ir(add, (a, b))
        
        assert isinstance(ir, ExtractedIR)
        assert ir.function_name == "add"
        assert "add" in ir.jaxpr.lower()
        assert "HloModule" in ir.hlo or "module" in ir.hlo.lower()
    
    def test_unary_function(self):
        """Test IR extraction for unary function."""
        def sin_func(x):
            return jnp.sin(x)
        
        x = jnp.ones((32,), dtype=jnp.float32)
        ir = extract_ir(sin_func, (x,))
        
        assert "sin" in ir.jaxpr.lower()
        assert len(ir.hlo) > 0
    
    def test_reduction(self):
        """Test IR extraction for reduction."""
        def sum_func(x):
            return jnp.sum(x)
        
        x = jnp.ones((64,), dtype=jnp.float32)
        ir = extract_ir(sum_func, (x,))
        
        assert "reduce" in ir.jaxpr.lower() or "sum" in ir.jaxpr.lower()
    
    def test_backward_included(self):
        """Test that backward pass is included."""
        def func(a, b):
            return a * b
        
        a = jnp.ones((32,), dtype=jnp.float32)
        b = jnp.ones((32,), dtype=jnp.float32)
        
        ir = extract_ir(func, (a, b), enable_backward=True)
        
        assert "BACKWARD" in ir.jaxpr or "GRADIENT" in ir.jaxpr
        assert "BACKWARD" in ir.hlo or "GRADIENT" in ir.hlo
    
    def test_no_backward(self):
        """Test extraction without backward pass."""
        def func(x):
            return x * 2
        
        x = jnp.ones((16,), dtype=jnp.float32)
        ir = extract_ir(func, (x,), enable_backward=False)
        
        assert "BACKWARD" not in ir.jaxpr
        assert len(ir.hlo) > 0


class TestExtractIRPair:
    """Tests for extract_ir_pair function."""
    
    def test_jaxpr_pair(self):
        """Test jaxpr pair extraction."""
        def func(x):
            return x + 1
        
        x = jnp.ones((32,))
        python_src, jaxpr = extract_ir_pair(func, (x,), ir_type="jaxpr")
        
        assert "def func" in python_src
        assert "add" in jaxpr.lower() or "+" in jaxpr
    
    def test_hlo_pair(self):
        """Test HLO pair extraction."""
        def func(x):
            return jnp.abs(x)
        
        x = jnp.ones((32,))
        python_src, hlo = extract_ir_pair(func, (x,), ir_type="hlo")
        
        assert "jnp.abs" in python_src
        assert len(hlo) > 100  # HLO should be substantial


class TestVmapExtraction:
    """Tests for vmap IR extraction."""
    
    def test_vmap_extraction(self):
        """Test vmapped function IR extraction."""
        def element_func(x):
            return jnp.sin(x) + jnp.cos(x)
        
        x = jnp.array(1.0)
        ir = extract_vmap_ir(element_func, x, batch_size=32)
        
        assert isinstance(ir, ExtractedIR)
        assert len(ir.jaxpr) > 0
        assert len(ir.hlo) > 0


class TestComplexFunctions:
    """Tests for more complex functions."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        def matmul(a, b):
            return jnp.matmul(a, b)
        
        a = jnp.ones((32, 64), dtype=jnp.float32)
        b = jnp.ones((64, 16), dtype=jnp.float32)
        
        ir = extract_ir(matmul, (a, b))
        
        assert "dot" in ir.jaxpr.lower() or "matmul" in ir.jaxpr.lower()
    
    def test_conditional(self):
        """Test conditional function."""
        def cond_func(x):
            return jnp.where(x > 0, x * 2, x * 0.5)
        
        x = jnp.linspace(-1, 1, 64)
        ir = extract_ir(cond_func, (x,))
        
        assert "select" in ir.jaxpr.lower() or "where" in ir.jaxpr.lower()
    
    def test_multi_output(self):
        """Test function with multiple operations."""
        def multi_func(a, b):
            c = a + b
            d = jnp.sin(c)
            e = d * a
            return e
        
        a = jnp.ones((32,))
        b = jnp.ones((32,))
        
        ir = extract_ir(multi_func, (a, b))
        
        # Should contain multiple operations
        assert "add" in ir.jaxpr.lower()
        assert "sin" in ir.jaxpr.lower()
        assert "mul" in ir.jaxpr.lower()


if __name__ == "__main__":
    # Run basic tests
    print("=== Running IR Extractor Tests ===\n")
    
    test_cases = [
        TestExtractIR(),
        TestExtractIRPair(),
        TestVmapExtraction(),
        TestComplexFunctions(),
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_cases:
        class_name = test_class.__class__.__name__
        print(f"\n--- {class_name} ---")
        
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    getattr(test_class, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
