@wp.kernel
def combined_sfdqao(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    acc = float(0.0)
    for i in range(3):
        if a[tid] * float(i) > 0.19:
            acc = acc + wp.sin(b[tid])
        else:
            acc = acc + b[tid]
    out[tid] = acc
