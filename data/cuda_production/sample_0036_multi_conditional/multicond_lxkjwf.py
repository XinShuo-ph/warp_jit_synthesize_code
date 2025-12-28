@wp.kernel
def multicond_lxkjwf(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    if val < -0.08:
        out[tid] = val * 0.5
    elif val < 4.54:
        out[tid] = val * 1.0
    else:
        out[tid] = val * 2.0
