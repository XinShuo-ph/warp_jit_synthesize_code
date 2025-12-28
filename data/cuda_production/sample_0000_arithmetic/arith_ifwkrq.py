@wp.kernel
def arith_ifwkrq(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    var_0 = a[tid] + b[tid]
    c[tid] = var_0
