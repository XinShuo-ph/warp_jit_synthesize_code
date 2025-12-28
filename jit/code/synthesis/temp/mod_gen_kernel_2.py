import warp as wp

@wp.kernel
def gen_kernel_2(data: wp.array(dtype=float), arg_0: int, arg_1: float, arg_2: float):
    tid = wp.tid()
    t2 = float(tid)
    t3 = wp.max(t2, t2)
    t4 = int(2)
    t5 = data[tid]
    t6 = wp.min(tid, t4)
    data[tid] = t3
    t8 = t5 + t2
    t9 = data[t6]
    t10 = t4 * t4
    t11 = float(-5.46)
    t12 = t8 - t5
