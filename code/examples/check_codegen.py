import warp as wp
import warp._src.context
import warp._src.codegen

wp.init()

@wp.kernel
def simple_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Get the kernel object
kernel = simple_kernel

# Use ModuleBuilder
options = kernel.module.options.copy()
builder = warp._src.context.ModuleBuilder(kernel.module, options=options)

# This should build the kernel and compute hash
builder.build_kernel(kernel)

print(f"Options keys: {options.keys()}")

# Now generate code
source = warp._src.codegen.codegen_kernel(kernel, device="cpu", options=options)

print("Generated C++ Code (IR):")
print(source)
