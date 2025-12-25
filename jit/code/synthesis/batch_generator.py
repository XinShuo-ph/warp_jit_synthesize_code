import multiprocessing
import warp as wp
import os
import json
import time
import sys

# Add workspace to path
sys.path.append(os.getcwd())

from jit.code.synthesis.pipeline import SynthesisPipeline

def worker_task(args):
    """
    Worker function to synthesize a single kernel.
    """
    idx, output_dir, temp_dir = args
    
    # Initialize Warp in the worker process
    # Suppress output if possible
    try:
        if not wp.config.initialized:
            wp.init()
    except Exception:
        pass
        
    pipeline = SynthesisPipeline(output_dir=output_dir, temp_dir=temp_dir)
    return pipeline.synthesize_pair(idx)

class BatchGenerator:
    def __init__(self, output_dir="jit/data/large_scale", temp_dir="jit/code/synthesis/temp_batch"):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
    def generate(self, total_count=100, num_workers=4):
        print(f"Starting batch generation of {total_count} samples with {num_workers} workers...")
        start_time = time.time()
        
        # Prepare arguments
        tasks = [(i, self.output_dir, self.temp_dir) for i in range(total_count)]
        
        results = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap_unordered for better responsiveness
            for i, result in enumerate(pool.imap_unordered(worker_task, tasks)):
                if result:
                    results.append(result)
                
                if (i+1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i+1) / elapsed
                    print(f"Processed {i+1}/{total_count} ({rate:.2f} samples/s)")
        
        # Save results
        output_file = os.path.join(self.output_dir, "dataset_large.jsonl")
        print(f"Saving {len(results)} samples to {output_file}...")
        
        with open(output_file, "w") as f:
            for sample in results:
                f.write(json.dumps(sample) + "\n")
                
        total_time = time.time() - start_time
        print(f"Completed in {total_time:.2f}s. Average: {len(results)/total_time:.2f} samples/s")

if __name__ == "__main__":
    # Ensure warp is not initialized in main process if we spawn?
    # Actually 'fork' (default on Linux) copies memory. 'spawn' is cleaner for CUDA.
    # Since we use CPU, fork is fine, but wp.init() might need care.
    # We'll rely on worker_task calling init.
    
    generator = BatchGenerator()
    # For testing, generate 50 samples with 4 workers
    generator.generate(total_count=50, num_workers=4)
