@wp.kernel
def math_ciiams(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.log(wp.abs(wp.exp(-a[tid])) + 1.0)
