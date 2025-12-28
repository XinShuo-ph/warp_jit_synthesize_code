@wp.kernel
def ctrl_jquzcd(a: wp.array(dtype=float), lo: float, hi: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    val = a[tid]
    if val < lo:
        out[tid] = lo
    elif val > hi:
        out[tid] = hi
    else:
        out[tid] = val
