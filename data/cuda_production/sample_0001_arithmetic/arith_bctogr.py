@wp.kernel
def arith_bctogr(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    var_0 = wp.exp(a[tid])
    var_1 = var_0 - b[tid]
    var_2 = wp.exp(var_1)
    c[tid] = var_2
