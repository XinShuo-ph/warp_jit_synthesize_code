@wp.kernel
def multicond_tdzjrm(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    if val < 1.21:
        out[tid] = val * 0.5
    elif val < 3.79:
        out[tid] = val * 1.0
    else:
        out[tid] = val * 2.0
