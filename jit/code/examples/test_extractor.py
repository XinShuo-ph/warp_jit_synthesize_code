import jax
import jax.numpy as jnp
import unittest
import sys
import os

# Add root to path so we can import from jit/code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from jit.code.extraction.ir_extractor import get_ir

class TestIRExtractor(unittest.TestCase):
    def test_simple_scalar(self):
        def add(x, y): return x + y
        ir = get_ir(add, 1.0, 2.0)
        self.assertIn("stablehlo.add", ir)

    def test_vector(self):
        def dot(x, y): return jnp.dot(x, y)
        x = jnp.ones((10,))
        y = jnp.ones((10,))
        ir = get_ir(dot, x, y)
        self.assertIn("stablehlo.dot", ir)

    def test_tuple_return(self):
        def f(x): return x, x*2
        ir = get_ir(f, 5.0)
        # Check if output signature reflects tuple (usually implicitly handled in HLO)
        self.assertIn("func.func", ir)

if __name__ == '__main__':
    unittest.main()
