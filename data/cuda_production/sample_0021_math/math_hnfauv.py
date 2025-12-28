@wp.kernel
def math_hnfauv(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = -wp.abs(wp.cos(a[tid]))
