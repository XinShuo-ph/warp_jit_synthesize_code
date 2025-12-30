"""Batch Generator - Efficient large-scale function pair generation for JAX."""
import json
import os
import sys
import time
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional
import argparse

import jax
import jax.numpy as jnp

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))


def init_worker():
    """Initialize JAX in each worker process."""
    # JAX is initialized automatically, but we ensure consistent config
    pass


def create_sample_args(sample_args_code: str, arg_names: list):
    """Create sample arguments from code string."""
    local_vars = {}
    exec(sample_args_code, {"jnp": jnp, "jax": jax}, local_vars)
    return tuple(local_vars[name] for name in arg_names)


def generate_single_pair(args):
    """Generate a single training pair. Designed for multiprocessing."""
    seed, output_dir, pair_id = args
    
    import random
    import re
    
    from generator import GENERATORS
    
    try:
        # Generate kernel
        random.seed(seed)
        generator = random.choice(GENERATORS)
        kernel_source, metadata = generator(seed)
        
        # Extract kernel name
        match = re.search(r'def (\w+)\(', kernel_source)
        kernel_name = match.group(1) if match else f"kernel_{pair_id}"
        
        # Create the function from source
        local_vars = {"jnp": jnp, "jax": jax}
        exec(kernel_source, local_vars)
        
        # Find the function
        func = None
        for name, obj in local_vars.items():
            if callable(obj) and not name.startswith('_') and name not in ['jnp', 'jax']:
                func = obj
                break
        
        if func is None:
            raise ValueError("No function found")
        
        # Create sample arguments
        sample_args = create_sample_args(metadata["sample_args_code"], metadata["arg_names"])
        
        # Get Jaxpr
        try:
            jaxpr = jax.make_jaxpr(func)(*sample_args)
            jaxpr_text = str(jaxpr)
        except Exception as e:
            jaxpr_text = f"# Jaxpr extraction failed: {e}"
        
        # Get HLO text
        try:
            lowered = jax.jit(func).lower(*sample_args)
            hlo_text = lowered.as_text()
        except Exception as e:
            hlo_text = f"# HLO extraction failed: {e}"
        
        # Get optimized HLO
        optimized_hlo = None
        try:
            lowered = jax.jit(func).lower(*sample_args)
            compiled = lowered.compile()
            optimized_hlo = compiled.as_text()
        except Exception:
            pass
        
        # Add backward pass
        try:
            def scalar_loss(*args):
                result = func(*args)
                if isinstance(result, jnp.ndarray):
                    return jnp.sum(result)
                return result
            
            grad_fn = jax.grad(scalar_loss)
            grad_jaxpr = jax.make_jaxpr(grad_fn)(*sample_args)
            jaxpr_text += "\n\n# === BACKWARD (GRADIENT) JAXPR ===\n"
            jaxpr_text += str(grad_jaxpr)
            
            grad_lowered = jax.jit(grad_fn).lower(*sample_args)
            grad_hlo = grad_lowered.as_text()
            hlo_text += "\n\n// === BACKWARD (GRADIENT) HLO ===\n"
            hlo_text += grad_hlo
            
            if optimized_hlo is not None:
                try:
                    grad_compiled = grad_lowered.compile()
                    grad_opt_hlo = grad_compiled.as_text()
                    optimized_hlo += "\n\n// === BACKWARD (GRADIENT) OPTIMIZED HLO ===\n"
                    optimized_hlo += grad_opt_hlo
                except Exception:
                    pass
        except Exception as e:
            jaxpr_text += f"\n\n# Backward extraction failed: {e}"
        
        result = {
            "id": pair_id,
            "kernel_name": kernel_name,
            "python": kernel_source.strip(),
            "hlo": hlo_text,
            "jaxpr": jaxpr_text,
            "type": generator.__name__
        }
        
        if optimized_hlo:
            result["optimized_hlo"] = optimized_hlo
        
        return result
            
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
                continue
            
            f.write(json.dumps(result) + '\n')
            f.flush()
            generated += 1
            
            if generated % checkpoint_every == 0:
                elapsed = time.time() - start_time
                rate = (generated - start_id) / elapsed
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
    print(f"Rate: {generated/elapsed:.2f} pairs/second")
    
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
                rate = (i + 1) / elapsed
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000, help="Number of pairs")
    parser.add_argument("--output", type=str, default="data/training_pairs.jsonl")
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
