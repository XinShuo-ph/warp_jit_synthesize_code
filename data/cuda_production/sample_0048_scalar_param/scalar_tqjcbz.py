@wp.kernel
def scalar_tqjcbz(x: wp.array(dtype=float), out: wp.array(dtype=float), scale: float, offset: float):
    tid = wp.tid()
    out[tid] = (x[tid] - scale) + offset
