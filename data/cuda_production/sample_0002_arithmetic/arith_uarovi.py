@wp.kernel
def arith_uarovi(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    var_0 = -a[tid]
    var_1 = wp.max(var_0, b[tid])
    var_2 = wp.log(wp.abs(var_1) + 1.0)
    var_3 = wp.max(var_2, b[tid])
    c[tid] = var_3
