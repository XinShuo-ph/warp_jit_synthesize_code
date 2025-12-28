@wp.kernel
def math_wgnhxn(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.log(wp.abs(a[tid]) + 1.0)
