import os
import sys
import json
import multiprocessing
import time
import warp as wp

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from jit.code.synthesis.pipeline import SynthesisPipeline

def worker_init():
    # Initialize warp in each worker process
    # Redirect output to devnull to avoid spamming stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        wp.init()
    except Exception:
        pass

def worker_func(args):
    idx, output_dir, temp_dir = args
    
    # Instantiate pipeline locally
    pipeline = SynthesisPipeline(output_dir=output_dir, temp_dir=temp_dir)
    
    # Generate single pair
    res = pipeline.generate_pair(idx)
    return res

class BatchGenerator:
    def __init__(self, output_dir="jit/data/dataset", num_samples=1000, num_workers=4):
        self.output_dir = output_dir
        self.temp_dir = os.path.join(output_dir, "temp_modules")
        self.num_samples = num_samples
        self.num_workers = num_workers
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(os.path.join(self.temp_dir, "__init__.py"), "w") as f:
            f.write("")

    def run(self):
        print(f"Starting generation of {self.num_samples} samples with {self.num_workers} workers...")
        start_time = time.time()
        
        # Prepare arguments
        tasks = [(i, self.output_dir, self.temp_dir) for i in range(self.num_samples)]
        
        # Run in parallel
        # Note: 'spawn' context is often safer for CUDA/Warp but 'fork' is default on Linux.
        # Warp context might not be picklable, so recreating pipeline in worker is correct.
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=self.num_workers, initializer=worker_init) as pool:
            # chunksize=10 to reduce IPC overhead
            results = pool.map(worker_func, tasks, chunksize=10)
            
        # Filter None results
        valid_results = [r for r in results if r is not None]
        
        # Save to JSONL
        output_file = os.path.join(self.output_dir, "dataset.jsonl")
        print(f"Saving {len(valid_results)} samples to {output_file}...")
        
        with open(output_file, "w") as f:
            for res in valid_results:
                f.write(json.dumps(res) + "\n")
                
        duration = time.time() - start_time
        print(f"Completed in {duration:.2f}s ({len(valid_results)/duration:.2f} samples/s)")

if __name__ == "__main__":
    # Full run
    # Use max available CPUs or limit to 8 to be polite
    workers = min(os.cpu_count() or 4, 8)
    gen = BatchGenerator(output_dir="jit/data/large_dataset", num_samples=10000, num_workers=workers)
    gen.run()
