@wp.kernel
def ctrl_scjoaa(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    total = float(0.0)
    for i in range(5):
        total = total + a[tid]
    out[tid] = total
