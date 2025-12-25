"""Wave equation solver example - simplified without rendering."""

import math
import warp as wp

wp.init()

@wp.func
def sample(f: wp.array(dtype=float), x: int, y: int, width: int, height: int):
    # clamp texture coords
    x = wp.clamp(x, 0, width - 1)
    y = wp.clamp(y, 0, height - 1)
    s = f[y * width + x]
    return s

@wp.func
def laplacian(f: wp.array(dtype=float), x: int, y: int, width: int, height: int):
    ddx = sample(f, x + 1, y, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x - 1, y, width, height)
    ddy = sample(f, x, y + 1, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x, y - 1, width, height)
    return ddx + ddy

@wp.kernel
def wave_solve(
    hprevious: wp.array(dtype=float),
    hcurrent: wp.array(dtype=float),
    width: int,
    height: int,
    inv_cell: float,
    k_speed: float,
    k_damp: float,
    dt: float,
):
    tid = wp.tid()
    x = tid % width
    y = tid // width
    
    l = laplacian(hcurrent, x, y, width, height) * inv_cell * inv_cell
    
    # integrate
    h1 = hcurrent[tid]
    h0 = hprevious[tid]
    
    h = 2.0 * h1 - h0 + dt * dt * (k_speed * l - k_damp * (h1 - h0))
    
    # buffers get swapped each iteration
    hprevious[tid] = h

# Simulation parameters
sim_width = 32
sim_height = 32
grid_size = 0.1
k_speed = 1.0
k_damp = 0.0
sim_dt = 0.01

# Create simulation grids
sim_grid0 = wp.zeros(sim_width * sim_height, dtype=float)
sim_grid1 = wp.zeros(sim_width * sim_height, dtype=float)

# Initialize with a wave in the center
center_idx = (sim_height // 2) * sim_width + (sim_width // 2)
sim_grid0_numpy = sim_grid0.numpy()
sim_grid0_numpy[center_idx] = 1.0
sim_grid0 = wp.array(sim_grid0_numpy, dtype=float)

print(f"Running wave simulation on {sim_width}x{sim_height} grid")
print(f"Initial center value: {sim_grid0.numpy()[center_idx]}")

# Run simulation for 10 steps
for step in range(10):
    wp.launch(
        kernel=wave_solve,
        dim=sim_width * sim_height,
        inputs=[
            sim_grid0,
            sim_grid1,
            sim_width,
            sim_height,
            1.0 / grid_size,
            k_speed,
            k_damp,
            sim_dt,
        ],
    )
    # Swap grids
    sim_grid0, sim_grid1 = sim_grid1, sim_grid0
    
    if step % 3 == 0:
        print(f"Step {step}: center value = {sim_grid0.numpy()[center_idx]:.6f}")

wp.synchronize()
print("\nWave simulation completed successfully!")
