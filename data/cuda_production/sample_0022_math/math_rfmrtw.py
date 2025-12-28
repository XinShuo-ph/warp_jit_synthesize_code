@wp.kernel
def math_rfmrtw(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.sqrt(wp.abs(wp.abs(a[tid])))
