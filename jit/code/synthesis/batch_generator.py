import multiprocessing
import time
import os
import sys

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from jit.code.synthesis.generator import JAXGenerator
from jit.code.synthesis.pipeline import generate_sample

def worker(args):
    idx, seed = args
    gen = JAXGenerator(seed=seed)
    return generate_sample(idx, gen)

def run_batch(num_samples=100, num_workers=4):
    start_time = time.time()
    
    tasks = [(i, i + int(time.time())) for i in range(num_samples)]
    
    # Use spawn to avoid JAX initialization issues across processes if any
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        results = pool.map(worker, tasks)
        
    successes = sum(results)
    duration = time.time() - start_time
    print(f"Generated {successes}/{num_samples} samples in {duration:.2f}s using {num_workers} workers.")

if __name__ == "__main__":
    run_batch(200, 4)
