import jax
import jax.numpy as jnp
from ir_extractor import extract_ir
import unittest

class TestIRExtractor(unittest.TestCase):
    
    def test_scalar_add(self):
        def add(x, y):
            return x + y
        
        res = extract_ir(add, 1.0, 2.0)
        self.assertTrue(res['success'])
        self.assertIn("add", res['jaxpr'])
        self.assertIn("stablehlo.add", res['hlo'])

    def test_vector_dot(self):
        def dot(v1, v2):
            return jnp.dot(v1, v2)
            
        v1 = jnp.array([1., 2., 3.])
        v2 = jnp.array([4., 5., 6.])
        res = extract_ir(dot, v1, v2)
        self.assertTrue(res['success'])
        self.assertIn("dot_general", res['jaxpr']) # or similar
        self.assertIn("stablehlo.dot_general", res['hlo'])

    def test_control_flow_where(self):
        def relu(x):
            return jnp.where(x > 0, x, 0.0)
            
        x = jnp.array([-1.0, 1.0, 2.0])
        res = extract_ir(relu, x)
        self.assertTrue(res['success'])
        self.assertIn("select", res['hlo']) # select is usually used for where

    def test_scan_loop(self):
        def cumsum(res, x):
            res = res + x
            return res, res
            
        def run_scan(xs):
            # scan carries state, iterates over xs
            final, stack = jax.lax.scan(cumsum, 0.0, xs)
            return stack
            
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])
        res = extract_ir(run_scan, xs)
        self.assertTrue(res['success'])
        self.assertIn("scan", res['jaxpr'])
        # HLO might inline or show while/loop

    def test_matmul(self):
        def matmul(m1, m2):
            return m1 @ m2
            
        m1 = jnp.ones((10, 20))
        m2 = jnp.ones((20, 30))
        res = extract_ir(matmul, m1, m2)
        self.assertTrue(res['success'])

if __name__ == '__main__':
    unittest.main()
