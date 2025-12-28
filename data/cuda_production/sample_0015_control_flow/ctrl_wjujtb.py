@wp.kernel
def ctrl_wjujtb(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    diff = a[tid] - b[tid]
    if diff < 0.0:
        out[tid] = -diff
    else:
        out[tid] = diff
