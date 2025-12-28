@wp.kernel
def combined_nrvzok(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    acc = float(0.0)
    for i in range(5):
        if a[tid] * float(i) > -0.15:
            acc = acc + wp.abs(b[tid])
        else:
            acc = acc + b[tid]
    out[tid] = acc
