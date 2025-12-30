"""JAX Batch Generator - Efficient large-scale function pair generation."""
import json
import os
import sys
import time
import tempfile
import importlib.util
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional
import argparse

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))


def init_worker():
    """Initialize JAX in each worker process."""
    import jax
    # Disable GPU for workers to avoid memory issues
    jax.config.update('jax_platform_name', 'cpu')


def generate_single_pair(args):
    """Generate a single training pair. Designed for multiprocessing."""
    seed, output_dir, pair_id = args
    
    # Import here to avoid pickling issues
    import random
    import re
    import tempfile
    import importlib.util
    import jax
    import jax.numpy as jnp
    
    from jax_generator import GENERATORS
    from jax_ir_extractor import extract_ir
    
    try:
        # Generate kernel
        random.seed(seed)
        generator = random.choice(GENERATORS)
        kernel_source = generator(seed)
        
        # Extract kernel name
        match = re.search(r'def (\w+)\(', kernel_source)
        kernel_name = match.group(1) if match else f"kernel_{pair_id}"
        
        # Create temp module
        module_name = f"batch_module_{pair_id}"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import jax\n")
            f.write("import jax.numpy as jnp\n")
            f.write("import jax.lax\n\n")
            f.write(kernel_source)
            temp_path = f.name
        
        try:
            # Load and compile
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find function
            func = None
            func_name = None
            for name in dir(module):
                if name.startswith('_'):
                    continue
                obj = getattr(module, name)
                if callable(obj) and name not in ('jax', 'jnp', 'lax'):
                    func = obj
                    func_name = name
                    break
            
            if func is None:
                raise ValueError("No function found")
            
            # Generate sample inputs based on generator type
            key = jax.random.PRNGKey(seed)
            generator_name = generator.__name__
            
            if 'elementwise' in generator_name:
                sample_inputs = (jax.random.normal(key, (16,)), 
                               jax.random.normal(jax.random.PRNGKey(seed+1), (16,)))
            elif 'scalar_arr' in generator_name:
                sample_inputs = (2.0, jax.random.normal(key, (16,)),
                               jax.random.normal(jax.random.PRNGKey(seed+1), (16,)))
            elif 'unary' in generator_name:
                sample_inputs = (jnp.abs(jax.random.normal(key, (16,))) + 0.1,)
            elif 'branch' in generator_name or 'nested' in generator_name:
                sample_inputs = (jax.random.normal(key, (16,)),)
            elif 'loop' in generator_name:
                sample_inputs = (jax.random.normal(key, (16,)), 5)
            elif 'reduce' in generator_name:
                sample_inputs = (jax.random.normal(key, (16,)),)
            elif 'vec' in generator_name:
                sample_inputs = (jax.random.normal(key, (8, 3)),
                               jax.random.normal(jax.random.PRNGKey(seed+1), (8, 3)))
            elif 'multi' in generator_name:
                sample_inputs = (jnp.abs(jax.random.normal(key, (16,))) + 0.1,
                               jnp.abs(jax.random.normal(jax.random.PRNGKey(seed+1), (16,))) + 0.1)
            elif 'compound' in generator_name:
                sample_inputs = (jax.random.normal(key, (16,)),
                               jax.random.normal(jax.random.PRNGKey(seed+1), (16,)), 2.5)
            elif 'matmul' in generator_name:
                sample_inputs = (jax.random.normal(key, (8, 8)),
                               jax.random.normal(jax.random.PRNGKey(seed+1), (8, 8)))
            elif 'softmax' in generator_name:
                sample_inputs = (jax.random.normal(key, (8, 10)),)
            elif 'scan' in generator_name:
                sample_inputs = (jax.random.normal(key, (16,)),)
            elif 'vmap' in generator_name:
                sample_inputs = (jax.random.normal(key, (8, 4)),
                               jax.random.normal(jax.random.PRNGKey(seed+1), (8, 4)))
            elif 'grad' in generator_name:
                sample_inputs = (jax.random.normal(key, (16,)),)
            else:
                sample_inputs = (jax.random.normal(key, (16,)),)
            
            # Inspect function signature to get correct number of arguments
            import inspect
            try:
                sig = inspect.signature(func)
                num_params = len([p for p in sig.parameters.values() 
                                if p.default == inspect.Parameter.empty])
                # Adjust sample_inputs if needed
                if len(sample_inputs) > num_params:
                    sample_inputs = sample_inputs[:num_params]
            except (ValueError, TypeError):
                pass  # Fall back to default inputs
            
            # Extract IR
            ir = extract_ir(func, sample_inputs)
            
            result = {
                "id": pair_id,
                "kernel_name": func_name,
                "python": kernel_source.strip(),
                "hlo": ir.hlo_text,
                "type": generator.__name__
            }
            
            if ir.hlo_optimized:
                result["hlo_optimized"] = ir.hlo_optimized
            
            # Cleanup
            os.unlink(temp_path)
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return result
            
        except Exception as e:
            os.unlink(temp_path)
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise e
            
    except Exception as e:
        return {"id": pair_id, "error": str(e)}


def batch_generate_sequential(
    count: int,
    output_path: str,
    base_seed: int = 42,
    checkpoint_every: int = 100
):
    """Generate pairs sequentially with checkpointing."""
    output_path = Path(output_path)
    checkpoint_path = output_path.with_suffix('.checkpoint')
    
    # Load checkpoint if exists
    start_id = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            start_id = int(f.read().strip())
        print(f"Resuming from checkpoint at pair {start_id}")
    
    mode = 'a' if start_id > 0 else 'w'
    
    generated = start_id
    failed = 0
    start_time = time.time()
    
    with open(output_path, mode) as f:
        for i in range(start_id, count):
            result = generate_single_pair((base_seed + i, None, i))
            
            if "error" in result:
                failed += 1
                print(f"Error at {i}: {result['error']}")
                continue
            
            f.write(json.dumps(result) + '\n')
            f.flush()
            generated += 1
            
            if generated % checkpoint_every == 0:
                elapsed = time.time() - start_time
                rate = (generated - start_id) / elapsed if elapsed > 0 else 0
                eta = (count - generated) / rate if rate > 0 else 0
                
                print(f"Generated {generated}/{count} ({failed} failed) "
                      f"- {rate:.2f}/s, ETA: {eta/60:.1f}min")
                
                with open(checkpoint_path, 'w') as cp:
                    cp.write(str(generated))
    
    # Remove checkpoint on completion
    if checkpoint_path.exists():
        os.unlink(checkpoint_path)
    
    elapsed = time.time() - start_time
    print(f"\nComplete: {generated} pairs in {elapsed/60:.1f} minutes")
    print(f"Rate: {generated/elapsed:.2f} pairs/second" if elapsed > 0 else "N/A")
    
    return generated


def batch_generate_parallel(
    count: int,
    output_path: str,
    base_seed: int = 42,
    num_workers: int = None
):
    """Generate pairs in parallel using multiprocessing."""
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # Limit workers due to memory
    
    print(f"Starting parallel generation with {num_workers} workers...")
    start_time = time.time()
    
    # Prepare arguments for each pair
    args = [(base_seed + i, None, i) for i in range(count)]
    
    with Pool(num_workers, initializer=init_worker) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(generate_single_pair, args)):
            results.append(result)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"Generated {i + 1}/{count} - {rate:.2f}/s")
    
    # Filter and save
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["id"])
    
    with open(output_path, 'w') as f:
        for result in valid_results:
            f.write(json.dumps(result) + '\n')
    
    elapsed = time.time() - start_time
    print(f"\nComplete: {len(valid_results)} pairs in {elapsed/60:.1f} minutes")
    print(f"Failed: {len(results) - len(valid_results)}")
    
    return len(valid_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JAX Python→HLO training pairs")
    parser.add_argument("--count", type=int, default=1000, help="Number of pairs")
    parser.add_argument("--output", type=str, default="data/jax_training_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", action="store_true", help="Use parallel generation")
    parser.add_argument("--workers", type=int, default=None)
    
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent.parent.parent)  # cd to jit/
    os.makedirs(Path(args.output).parent, exist_ok=True)
    
    if args.parallel:
        batch_generate_parallel(args.count, args.output, args.seed, args.workers)
    else:
        batch_generate_sequential(args.count, args.output, args.seed)
