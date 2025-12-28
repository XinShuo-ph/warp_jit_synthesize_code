"""Test CUDA code generation vs CPU code generation."""
import warp as wp
import warp._src.context
import warp._src.codegen

wp.init()

@wp.kernel
def test_kernel(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0

@wp.kernel
def vec_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0

def generate_code(kernel, device):
    """Generate code for a kernel on a specific device."""
    options = kernel.module.options.copy()
    builder = warp._src.context.ModuleBuilder(kernel.module, options=options)
    builder.build_kernel(kernel)
    return warp._src.codegen.codegen_kernel(kernel, device=device, options=options)

if __name__ == "__main__":
    print("=" * 60)
    print("CPU Code Generation")
    print("=" * 60)
    cpu_code = generate_code(test_kernel, "cpu")
    print(cpu_code)
    
    print("\n" + "=" * 60)
    print("CUDA Code Generation")
    print("=" * 60)
    try:
        cuda_code = generate_code(test_kernel, "cuda")
        print(cuda_code)
        
        print("\n" + "=" * 60)
        print("Key Differences")
        print("=" * 60)
        print(f"CPU code length: {len(cpu_code)} chars")
        print(f"CUDA code length: {len(cuda_code)} chars")
        print(f"CPU has '_cpu_kernel_forward': {'_cpu_kernel_forward' in cpu_code}")
        print(f"CUDA has '_cuda_kernel_forward': {'_cuda_kernel_forward' in cuda_code}")
        print(f"CUDA has '__global__': {'__global__' in cuda_code}")
        print(f"CUDA has 'threadIdx': {'threadIdx' in cuda_code}")
    except Exception as e:
        print(f"Error generating CUDA code: {e}")
