@wp.kernel
def multicond_rmmlsj(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    if val < -2.44:
        out[tid] = val * 0.5
    elif val < 0.95:
        out[tid] = val * 1.0
    else:
        out[tid] = val * 2.0
