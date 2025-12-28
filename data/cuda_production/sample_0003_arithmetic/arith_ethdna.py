@wp.kernel
def arith_ethdna(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    var_0 = a[tid] * b[tid]
    var_1 = wp.log(wp.abs(var_0) + 1.0)
    c[tid] = var_1
