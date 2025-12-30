"""Synthesis Pipeline - Generates Python→HLO pairs for LLM training."""
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from generator import generate_kernel_batch, GENERATORS


@dataclass
class TrainingPair:
    """A Python→HLO training pair."""
    id: int
    kernel_name: str
    python_source: str
    hlo_code: str  # HLO representation (similar to CPU code in Warp)
    jaxpr_code: str  # Jaxpr representation (JAX's IR)
    optimized_hlo: Optional[str]  # Optimized HLO (similar to CUDA code)
    generator_type: str


def create_sample_args(sample_args_code: str, arg_names: List[str]):
    """Create sample arguments from code string."""
    # Execute the sample args code to get the variables
    local_vars = {}
    exec(sample_args_code, {"jnp": jnp, "jax": jax}, local_vars)
    
    # Return the arguments in order
    return tuple(local_vars[name] for name in arg_names)


def extract_ir_from_source(kernel_source: str, sample_args_code: str, arg_names: List[str]):
    """Extract HLO and Jaxpr IR from kernel source code."""
    # Create the function from source
    local_vars = {"jnp": jnp, "jax": jax}
    exec(kernel_source, local_vars)
    
    # Find the function (it's the last defined function in local_vars)
    func = None
    func_name = None
    for name, obj in local_vars.items():
        if callable(obj) and not name.startswith('_') and name not in ['jnp', 'jax']:
            func = obj
            func_name = name
    
    if func is None:
        raise ValueError("No function found in generated source")
    
    # Create sample arguments
    sample_args = create_sample_args(sample_args_code, arg_names)
    
    # Get Jaxpr
    try:
        jaxpr = jax.make_jaxpr(func)(*sample_args)
        jaxpr_text = str(jaxpr)
    except Exception as e:
        jaxpr_text = f"# Jaxpr extraction failed: {e}"
    
    # Get HLO text (lowered representation)
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
    
    # Add backward pass (gradient) IR
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
    
    return hlo_text, jaxpr_text, optimized_hlo


def generate_training_pairs(
    count: int,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> List[TrainingPair]:
    """Generate training pairs with HLO and Jaxpr code."""
    import random
    
    pairs = []
    failed = 0
    
    for i in range(count):
        seed = base_seed + i
        
        # Pick a generator and generate kernel
        random.seed(seed)
        generator = random.choice(GENERATORS)
        kernel_source, metadata = generator(seed)
        
        # Extract kernel name from source
        match = re.search(r'def (\w+)\(', kernel_source)
        kernel_name = match.group(1) if match else f"kernel_{i}"
        
        try:
            hlo_code, jaxpr_code, optimized_hlo = extract_ir_from_source(
                kernel_source, 
                metadata["sample_args_code"],
                metadata["arg_names"]
            )
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                hlo_code=hlo_code,
                jaxpr_code=jaxpr_code,
                optimized_hlo=optimized_hlo,
                generator_type=generator.__name__
            )
            pairs.append(pair)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} pairs ({failed} failed)")
                
        except Exception as e:
            failed += 1
            print(f"Failed to generate pair {i}: {e}")
            continue
    
    print(f"\nTotal: {len(pairs)} pairs generated, {failed} failed")
    
    # Save to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / "training_pairs.json"
        
        with open(output_path, 'w') as f:
            json.dump([asdict(p) for p in pairs], f, indent=2)
        
        print(f"Saved to {output_path}")
    
    return pairs


def generate_batch_to_jsonl(
    count: int,
    output_path: str,
    base_seed: int = 42,
    ir_type: str = "both"
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        ir_type: "hlo", "jaxpr", "optimized", or "both" (hlo + optimized)
    """
    import random
    
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            random.seed(seed)
            generator = random.choice(GENERATORS)
            kernel_source, metadata = generator(seed)
            
            match = re.search(r'def (\w+)\(', kernel_source)
            kernel_name = match.group(1) if match else f"kernel_{i}"
            
            try:
                hlo_code, jaxpr_code, optimized_hlo = extract_ir_from_source(
                    kernel_source,
                    metadata["sample_args_code"],
                    metadata["arg_names"]
                )
                
                pair = {
                    "id": i,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "type": generator.__name__
                }
                
                if ir_type == "hlo":
                    pair["hlo"] = hlo_code
                elif ir_type == "jaxpr":
                    pair["jaxpr"] = jaxpr_code
                elif ir_type == "optimized":
                    if optimized_hlo:
                        pair["optimized_hlo"] = optimized_hlo
                    else:
                        continue  # Skip if no optimized HLO
                else:  # both
                    pair["hlo"] = hlo_code
                    pair["jaxpr"] = jaxpr_code
                    if optimized_hlo:
                        pair["optimized_hlo"] = optimized_hlo
                
                f.write(json.dumps(pair) + '\n')
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated {generated} pairs...")
                    
            except Exception as e:
                failed += 1
                continue
    
    print(f"\nTotal: {generated} pairs saved to {output_path}, {failed} failed")
    return generated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→HLO training pairs")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--ir-type", type=str, default="both", 
                        choices=["hlo", "jaxpr", "optimized", "both"],
                        help="Type of IR to include in output")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.ir_type)
    elif args.output:
        generate_training_pairs(args.count, args.seed, args.output)
    else:
        # Demo mode
        pairs = generate_training_pairs(args.count, args.seed)
        
        print("\n=== Sample Pairs ===")
        for pair in pairs[:3]:
            print(f"\n--- {pair.kernel_name} ({pair.generator_type}) ---")
            print("Python:")
            print(pair.python_source)
            print(f"\nHLO ({len(pair.hlo_code)} chars):")
            print(pair.hlo_code[:500] + "..." if len(pair.hlo_code) > 500 else pair.hlo_code)
            print(f"\nJaxpr ({len(pair.jaxpr_code)} chars):")
            print(pair.jaxpr_code[:500] + "..." if len(pair.jaxpr_code) > 500 else pair.jaxpr_code)
            if pair.optimized_hlo:
                print(f"\nOptimized HLO ({len(pair.optimized_hlo)} chars):")
                print(pair.optimized_hlo[:500] + "..." if len(pair.optimized_hlo) > 500 else pair.optimized_hlo)
