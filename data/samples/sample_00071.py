import warp as wp

@wp.kernel
def arithmetic_0071(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = ((a[i] - b[i]) + b[i])
