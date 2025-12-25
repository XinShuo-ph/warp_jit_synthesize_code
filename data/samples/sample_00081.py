import warp as wp

@wp.kernel
def loop_0081(matrix: wp.array(dtype=float, ndim=2),
           vector: wp.array(dtype=float),
           result: wp.array(dtype=float),
           n: int):
    i = wp.tid()
    
    sum_val = float(0.0)
    for j in range(n):
        sum_val = sum_val + matrix[i, j] * vector[j]
    
    result[i] = sum_val
