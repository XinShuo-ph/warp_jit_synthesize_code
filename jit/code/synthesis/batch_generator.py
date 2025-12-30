"""Batch Generator - Efficient large-scale function pair generation for JAX."""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
import argparse
import random

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))


def generate_single_pair(args):
    """Generate a single training pair."""
    seed, pair_id, ir_type = args
    
    try:
        from generator import GENERATORS
        from ir_extractor import extract_ir
        
        # Generate function
        random.seed(seed)
        generator = random.choice(GENERATORS)
        func_spec = generator(seed)
        
        # Extract IR
        ir = extract_ir(func_spec.func, func_spec.sample_inputs, enable_backward=True)
        
        result = {
            "id": pair_id,
            "function_name": func_spec.name,
            "python": func_spec.source.strip(),
            "type": generator.__name__
        }
        
        if ir_type == "jaxpr":
            result["jaxpr"] = ir.jaxpr
        elif ir_type == "hlo":
            result["hlo"] = ir.hlo
        elif ir_type == "stablehlo" and ir.stablehlo:
            result["stablehlo"] = ir.stablehlo
        else:  # both
            result["jaxpr"] = ir.jaxpr
            result["hlo"] = ir.hlo
            if ir.stablehlo:
                result["stablehlo"] = ir.stablehlo
        
        return result
        
    except Exception as e:
        return {"id": pair_id, "error": str(e)}


def batch_generate_sequential(
    count: int,
    output_path: str,
    base_seed: int = 42,
    checkpoint_every: int = 100,
    ir_type: str = "both"
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
            result = generate_single_pair((base_seed + i, i, ir_type))
            
            if "error" in result:
                failed += 1
                print(f"  Failed pair {i}: {result['error']}")
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
    print(f"Rate: {generated/elapsed:.2f} pairs/second" if elapsed > 0 else "")
    
    return generated


def batch_generate_multiprocess(
    count: int,
    output_path: str,
    base_seed: int = 42,
    num_workers: int = None,
    ir_type: str = "both"
):
    """Generate pairs using multiprocessing."""
    from multiprocessing import Pool, cpu_count
    
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # Limit workers
    
    print(f"Starting parallel generation with {num_workers} workers...")
    start_time = time.time()
    
    # Prepare arguments for each pair
    args = [(base_seed + i, i, ir_type) for i in range(count)]
    
    results = []
    with Pool(num_workers) as pool:
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
    parser = argparse.ArgumentParser(description="Batch generate JAX training pairs")
    parser.add_argument("--count", type=int, default=1000, help="Number of pairs")
    parser.add_argument("--output", type=str, default="data/training_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", action="store_true", help="Use parallel generation")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ir-type", type=str, default="both",
                        choices=["jaxpr", "hlo", "stablehlo", "both"],
                        help="IR type(s) to include")
    
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent.parent.parent)  # cd to jit/
    os.makedirs(Path(args.output).parent, exist_ok=True)
    
    if args.parallel:
        batch_generate_multiprocess(
            args.count, args.output, args.seed, args.workers, args.ir_type
        )
    else:
        batch_generate_sequential(
            args.count, args.output, args.seed, ir_type=args.ir_type
        )
