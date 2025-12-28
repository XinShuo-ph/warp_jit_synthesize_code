"""
Advanced CUDA Kernel Generator

Generates kernels with advanced GPU patterns:
- Explicit shared memory usage
- 2D/3D thread grids
- Warp-level operations
- Cooperative patterns
"""
import random
import string
from typing import Any
from dataclasses import dataclass, field


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""
    name: str
    category: str
    source: str
    arg_types: dict[str, str]
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


def random_name(prefix: str = "kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=6))
    return f"{prefix}_{suffix}"


def generate_shared_memory_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a kernel with explicit shared memory usage."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("shared")
    pattern = random.choice(["block_sum", "block_max", "prefix_sum", "transpose"])
    
    if pattern == "block_sum":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), n: int):
    tid = wp.tid()
    block_size = 256
    
    # Allocate shared memory for block
    s_data = wp.shared_array(shape=(256,), dtype=float)
    
    # Load data into shared memory
    if tid < n:
        s_data[tid % block_size] = input[tid]
    else:
        s_data[tid % block_size] = 0.0
    
    # Synchronize threads in block
    wp.syncthreads()
    
    # Block-level reduction
    stride = block_size // 2
    while stride > 0:
        if (tid % block_size) < stride:
            s_data[tid % block_size] = s_data[tid % block_size] + s_data[(tid % block_size) + stride]
        wp.syncthreads()
        stride = stride // 2
    
    # First thread writes result
    if (tid % block_size) == 0:
        block_idx = tid // block_size
        output[block_idx] = s_data[0]
'''
        desc = "Block-level sum reduction with shared memory"
    
    elif pattern == "block_max":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), n: int):
    tid = wp.tid()
    block_size = 256
    
    s_data = wp.shared_array(shape=(256,), dtype=float)
    
    if tid < n:
        s_data[tid % block_size] = input[tid]
    else:
        s_data[tid % block_size] = -1e10
    
    wp.syncthreads()
    
    stride = block_size // 2
    while stride > 0:
        if (tid % block_size) < stride:
            val_a = s_data[tid % block_size]
            val_b = s_data[(tid % block_size) + stride]
            if val_b > val_a:
                s_data[tid % block_size] = val_b
        wp.syncthreads()
        stride = stride // 2
    
    if (tid % block_size) == 0:
        block_idx = tid // block_size
        output[block_idx] = s_data[0]
'''
        desc = "Block-level max reduction with shared memory"
    
    elif pattern == "prefix_sum":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), n: int):
    tid = wp.tid()
    block_size = 256
    
    s_data = wp.shared_array(shape=(256,), dtype=float)
    
    if tid < n:
        s_data[tid % block_size] = input[tid]
    else:
        s_data[tid % block_size] = 0.0
    
    wp.syncthreads()
    
    # Up-sweep phase
    offset = 1
    d = block_size // 2
    while d > 0:
        if (tid % block_size) < d:
            ai = offset * (2 * (tid % block_size) + 1) - 1
            bi = offset * (2 * (tid % block_size) + 2) - 1
            s_data[bi] = s_data[ai] + s_data[bi]
        offset = offset * 2
        d = d // 2
        wp.syncthreads()
    
    if tid < n:
        output[tid] = s_data[tid % block_size]
'''
        desc = "Prefix sum (scan) with shared memory"
    
    else:  # transpose
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), width: int, height: int):
    tid = wp.tid()
    x = tid % width
    y = tid // width
    
    tile_size = 16
    s_tile = wp.shared_array(shape=(16, 16), dtype=float)
    
    if x < width and y < height:
        tx = x % tile_size
        ty = y % tile_size
        s_tile[ty, tx] = input[y * width + x]
        
        wp.syncthreads()
        
        out_x = (y // tile_size) * tile_size + tx
        out_y = (x // tile_size) * tile_size + ty
        if out_x < height and out_y < width:
            output[out_y * height + out_x] = s_tile[tx, ty]
'''
        desc = "Matrix transpose with shared memory tiling"
    
    return KernelSpec(
        name=name,
        category="shared_memory",
        source=source,
        arg_types={"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)", 
                   "n": "int"} if pattern != "transpose" else 
                  {"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)", 
                   "width": "int", "height": "int"},
        description=desc,
        metadata={"pattern": pattern, "uses_shared_memory": True, "uses_syncthreads": True, "seed": seed}
    )


def generate_2d_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a 2D grid kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("grid2d")
    operation = random.choice(["conv", "blur", "sobel"])
    
    if operation == "conv":
        source = f'''@wp.kernel
def {name}(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float), 
           width: int, height: int):
    i, j = wp.tid()
    
    if i < height and j < width:
        # 3x3 convolution
        result = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ii = wp.clamp(i + di, 0, height - 1)
                jj = wp.clamp(j + dj, 0, width - 1)
                result = result + input[ii, jj]
        output[i, j] = result / 9.0
'''
        desc = "2D convolution (3x3 averaging)"
    
    elif operation == "blur":
        source = f'''@wp.kernel
def {name}(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float), 
           width: int, height: int):
    i, j = wp.tid()
    
    if i > 0 and i < height - 1 and j > 0 and j < width - 1:
        center = input[i, j]
        up = input[i - 1, j]
        down = input[i + 1, j]
        left = input[i, j - 1]
        right = input[i, j + 1]
        output[i, j] = (center * 2.0 + up + down + left + right) / 6.0
    else:
        output[i, j] = input[i, j]
'''
        desc = "2D Gaussian blur approximation"
    
    else:  # sobel
        source = f'''@wp.kernel
def {name}(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float), 
           width: int, height: int):
    i, j = wp.tid()
    
    if i > 0 and i < height - 1 and j > 0 and j < width - 1:
        # Sobel edge detection
        gx = -input[i-1, j-1] - 2.0*input[i, j-1] - input[i+1, j-1] + \
              input[i-1, j+1] + 2.0*input[i, j+1] + input[i+1, j+1]
        gy = -input[i-1, j-1] - 2.0*input[i-1, j] - input[i-1, j+1] + \
              input[i+1, j-1] + 2.0*input[i+1, j] + input[i+1, j+1]
        output[i, j] = wp.sqrt(gx * gx + gy * gy)
    else:
        output[i, j] = 0.0
'''
        desc = "2D Sobel edge detection"
    
    return KernelSpec(
        name=name,
        category="grid_2d",
        source=source,
        arg_types={"input": "wp.array2d(dtype=float)", "output": "wp.array2d(dtype=float)", 
                   "width": "int", "height": "int"},
        description=desc,
        metadata={"operation": operation, "dimension": "2D", "seed": seed}
    )


def generate_3d_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a 3D grid kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("grid3d")
    operation = random.choice(["laplacian", "gradient", "diffusion"])
    
    if operation == "laplacian":
        source = f'''@wp.kernel
def {name}(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float),
           nx: int, ny: int, nz: int):
    i, j, k = wp.tid()
    
    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
        center = input[i, j, k]
        neighbors = input[i-1, j, k] + input[i+1, j, k] + \
                   input[i, j-1, k] + input[i, j+1, k] + \
                   input[i, j, k-1] + input[i, j, k+1]
        output[i, j, k] = neighbors - 6.0 * center
    else:
        output[i, j, k] = 0.0
'''
        desc = "3D Laplacian operator"
    
    elif operation == "gradient":
        source = f'''@wp.kernel
def {name}(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float),
           nx: int, ny: int, nz: int):
    i, j, k = wp.tid()
    
    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
        dx = (input[i+1, j, k] - input[i-1, j, k]) * 0.5
        dy = (input[i, j+1, k] - input[i, j-1, k]) * 0.5
        dz = (input[i, j, k+1] - input[i, j, k-1]) * 0.5
        output[i, j, k] = wp.sqrt(dx*dx + dy*dy + dz*dz)
    else:
        output[i, j, k] = 0.0
'''
        desc = "3D gradient magnitude"
    
    else:  # diffusion
        source = f'''@wp.kernel
def {name}(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float),
           nx: int, ny: int, nz: int, dt: float):
    i, j, k = wp.tid()
    
    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
        center = input[i, j, k]
        neighbors = input[i-1, j, k] + input[i+1, j, k] + \
                   input[i, j-1, k] + input[i, j+1, k] + \
                   input[i, j, k-1] + input[i, j, k+1]
        laplacian = neighbors - 6.0 * center
        output[i, j, k] = center + dt * laplacian
    else:
        output[i, j, k] = input[i, j, k]
'''
        desc = "3D heat diffusion step"
    
    return KernelSpec(
        name=name,
        category="grid_3d",
        source=source,
        arg_types={"input": "wp.array3d(dtype=float)", "output": "wp.array3d(dtype=float)", 
                   "nx": "int", "ny": "int", "nz": "int"} if operation != "diffusion" else
                  {"input": "wp.array3d(dtype=float)", "output": "wp.array3d(dtype=float)", 
                   "nx": "int", "ny": "int", "nz": "int", "dt": "float"},
        description=desc,
        metadata={"operation": operation, "dimension": "3D", "seed": seed}
    )


# Generator dispatch table
ADVANCED_GENERATORS = {
    "shared_memory": generate_shared_memory_kernel,
    "grid_2d": generate_2d_kernel,
    "grid_3d": generate_3d_kernel,
}


def generate_advanced_kernel(category: str | None = None, seed: int | None = None) -> KernelSpec:
    """Generate an advanced CUDA kernel."""
    if category is None:
        category = random.choice(list(ADVANCED_GENERATORS.keys()))
    
    if category not in ADVANCED_GENERATORS:
        raise ValueError(f"Unknown category: {category}. Available: {list(ADVANCED_GENERATORS.keys())}")
    
    return ADVANCED_GENERATORS[category](seed)


if __name__ == "__main__":
    # Demo: Generate one kernel of each type
    print("=" * 60)
    print("Advanced CUDA Kernel Generator Demo")
    print("=" * 60)
    
    for cat in ADVANCED_GENERATORS.keys():
        spec = generate_advanced_kernel(cat, seed=42)
        print(f"\n--- {cat.upper()} ---")
        print(f"Name: {spec.name}")
        print(f"Description: {spec.description}")
        print(f"Source (first 300 chars):\n{spec.source[:300]}...")
