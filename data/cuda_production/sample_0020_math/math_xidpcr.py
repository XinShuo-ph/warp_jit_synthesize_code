@wp.kernel
def math_xidpcr(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.sqrt(wp.abs(wp.log(wp.abs(a[tid]) + 1.0)))
