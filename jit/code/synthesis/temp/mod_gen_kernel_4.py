import warp as wp

@wp.kernel
def gen_kernel_4(data: wp.array(dtype=float), arg_0: int):
    tid = wp.tid()
    t0 = tid - tid
    t2 = wp.min(tid, t0)
    t3 = t2 + tid
    t4 = data[tid]
    t5 = t0 + t2
    data[t2] = t4
    t7 = t2 * t2
    t8 = wp.max(t4, t4)
    t9 = t3 - t3
    t10 = int(t4)
