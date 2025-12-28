import warp as wp
wp.init()

@wp.kernel
def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Try to extract CUDA IR
import warp._src.context as ctx
module = simple_add.module
hasher = ctx.ModuleHasher(module)
options = {'block_dim': 256, 'enable_backward': False, 'mode': 'release'}
builder = ctx.ModuleBuilder(module, options, hasher)

print('Attempting CPU code generation...')
cpu_code = builder.codegen('cpu')
print(f'CPU code generated: {len(cpu_code)} chars')
print(f'Contains "cpu_kernel_forward": {"cpu_kernel_forward" in cpu_code}')

print('\nAttempting CUDA code generation...')
try:
    cuda_code = builder.codegen('cuda')
    print(f'CUDA code generated: {len(cuda_code)} chars')
    print(f'Contains "cuda_kernel_forward": {"cuda_kernel_forward" in cuda_code}')
    print(f'\n=== CUDA code sample (first 500 chars) ===')
    print(cuda_code[:500])
    
    # Save for analysis
    with open('/workspace/cuda/data/test_cuda_code.cu', 'w') as f:
        f.write(cuda_code)
    print('\nSaved CUDA code to data/test_cuda_code.cu')
    
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

# Also save CPU code for comparison
with open('/workspace/cuda/data/test_cpu_code.cpp', 'w') as f:
    f.write(cpu_code)
print('Saved CPU code to data/test_cpu_code.cpp')
