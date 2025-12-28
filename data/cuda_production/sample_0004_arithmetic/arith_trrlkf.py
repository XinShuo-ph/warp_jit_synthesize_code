@wp.kernel
def arith_trrlkf(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    var_0 = wp.abs(a[tid])
    var_1 = wp.abs(var_0)
    c[tid] = var_1
