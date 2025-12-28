from __future__ import annotations

import warp as wp


@wp.kernel
def add_constant(a: wp.array(dtype=wp.float32), c: float):
    i = wp.tid()
    a[i] = a[i] + c


@wp.kernel
def conditional_scale(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x = a[i]
    if x > 0.0:
        out[i] = x * 2.0
    else:
        out[i] = x * 0.5


@wp.struct
class Pair:
    x: float
    y: float


@wp.kernel
def struct_math(p: wp.array(dtype=Pair), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = p[i]
    out[i] = v.x * v.y + 1.0


@wp.kernel
def atomic_accumulate(x: wp.array(dtype=wp.float32), acc: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(acc, 0, x[i])


@wp.kernel
def trig_mix(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = x[i]
    out[i] = wp.sin(v) + wp.cos(v * 0.5)

