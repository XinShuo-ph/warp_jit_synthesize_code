@wp.kernel
def test_arithmetic(a: wp.array(dtype=float), 
                    b: wp.array(dtype=float),
                    c: wp.array(dtype=float)):
    """Simple element-wise operations."""
    i = wp.tid()
    c[i] = a[i] * 2.0 + b[i] - 1.0
