import warp as wp

@wp.kernel
def gen_kernel_1(data: wp.array(dtype=float), arg_0: int, arg_1: wp.array(dtype=float), arg_2: int, arg_3: int, arg_4: float):
    tid = wp.tid()
    t0 = tid + tid
    t1 = float(4.88)
    t2 = wp.max(t0, tid)
    data[t2] = t1
    t4 = data[t2]
    t5 = float(tid)
    t6 = int(t1)
    t7 = data[t0]
    t8 = data[t2]
    t9 = float(t2)
