@wp.kernel
def test_loops(matrix: wp.array(dtype=float, ndim=2),
               vector: wp.array(dtype=float),
               result: wp.array(dtype=float),
               n: int):
    """Matrix-vector multiplication with explicit loop."""
    i = wp.tid()

    sum_val = float(0.0)  # Must use float() for mutable loop variable
    for j in range(n):
        sum_val = sum_val + matrix[i, j] * vector[j]

    result[i] = sum_val
