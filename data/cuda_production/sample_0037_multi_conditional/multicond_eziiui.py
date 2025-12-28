@wp.kernel
def multicond_eziiui(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    if val < -4.89:
        out[tid] = val * 0.5
    elif val < -1.18:
        out[tid] = val * 1.0
    else:
        out[tid] = val * 2.0
