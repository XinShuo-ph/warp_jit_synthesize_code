import warp as wp
import warp._src.context
import warp._src.codegen

wp.init()

@wp.kernel
def loop_kernel(n: int, out: wp.array(dtype=float)):
    sum = float(0.0)
    for i in range(n):
        sum = sum + float(i)
    out[0] = sum

options = loop_kernel.module.options.copy()
builder = warp._src.context.ModuleBuilder(loop_kernel.module, options=options)
builder.build_kernel(loop_kernel)
source = warp._src.codegen.codegen_kernel(loop_kernel, device="cpu", options=options)
print(source)
