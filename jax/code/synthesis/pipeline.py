"""Synthesis Pipeline - Generates Python→HLO pairs for LLM training."""
import json
import os
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

import jax
import jax.numpy as jnp
from generator import generate_kernel_batch, GENERATORS, EXTENDED_GENERATORS


@dataclass
class TrainingPair:
    """A Python→HLO training pair."""
    id: int
    kernel_name: str
    python_source: str
    hlo_forward: str
    hlo_backward: Optional[str]
    hlo_optimized: Optional[str]
    generator_type: str


def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    # Create a temp file to store the kernel
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n\n")
        f.write(source)
        temp_path = f.name
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, temp_path
    except Exception as e:
        os.unlink(temp_path)
        raise e


def get_inputs_for_function(fn, generator_name: str) -> tuple:
    """Get appropriate inputs based on function signature and generator type."""
    import inspect
    
    # Get the function signature
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        num_params = len(params)
    except (ValueError, TypeError):
        num_params = 2  # Default fallback
        params = []
    
    # Check for special parameter names
    has_key = any(p in ['key', 'rng_key'] for p in params)
    has_training = any(p in ['training', 'train'] for p in params)
    
    # Handle specific generator types
    if "attention" in generator_name:
        return (jnp.ones((4, 8, 16)), jnp.ones((4, 8, 16)), jnp.ones((4, 8, 16)))
    elif "matmul" in generator_name:
        return (jnp.ones((8, 8)), jnp.ones((8, 8)))
    elif "conv" in generator_name:
        return (jnp.ones((16,)), jnp.ones((3,)))
    elif "layernorm" in generator_name or "batchnorm" in generator_name:
        return (jnp.ones((4, 16)), jnp.ones((16,)), jnp.zeros((16,)))
    elif "dropout" in generator_name:
        return (jnp.ones((10,)), jax.random.PRNGKey(0), True)
    elif "softmax" in generator_name or "gelu" in generator_name:
        return (jnp.ones((4, 10)),)
    elif "loop" in generator_name:
        return (jnp.ones((10,)), 5)
    
    # For other cases, generate inputs based on signature
    inputs = []
    for param in params:
        if param in ['key', 'rng_key']:
            inputs.append(jax.random.PRNGKey(0))
        elif param in ['training', 'train']:
            inputs.append(True)
        elif param in ['n', 'num', 'count', 'size']:
            inputs.append(5)
        elif param in ['alpha', 'beta', 'gamma', 'scale', 'rate']:
            inputs.append(jnp.array(1.0))
        elif param in ['q', 'k', 'v']:
            inputs.append(jnp.ones((4, 8, 16)))
        elif param == 'kernel':
            inputs.append(jnp.ones((3,)))
        else:
            # Default to 1D float array
            inputs.append(jnp.ones((10,)))
    
    return tuple(inputs) if inputs else (jnp.ones((10,)),)


def extract_ir_from_source(kernel_source: str, kernel_name: str, module_id: int, generator_name: str):
    """Extract HLO IR from kernel source code."""
    from ir_extractor import extract_ir
    
    module_name = f"synth_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(kernel_source, module_name)
        
        # Find the kernel function in the module
        kernel_fn = None
        for name in dir(module):
            if not name.startswith('_'):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith('jnp') and not name.startswith('jax'):
                    kernel_fn = obj
                    break
        
        if kernel_fn is None:
            raise ValueError(f"No function found in generated source")
        
        # Get appropriate inputs based on function signature
        inputs = get_inputs_for_function(kernel_fn, generator_name)
        
        # Extract IR
        ir = extract_ir(kernel_fn, inputs)
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir.hlo_text, ir.hlo_backward, ir.hlo_optimized
        
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise e


def generate_training_pairs(
    count: int,
    base_seed: int = 42,
    output_dir: Optional[str] = None,
    extended: bool = False
) -> List[TrainingPair]:
    """Generate training pairs with HLO code."""
    
    pairs = []
    failed = 0
    generators = EXTENDED_GENERATORS if extended else GENERATORS
    
    for i in range(count):
        seed = base_seed + i
        
        # Pick a generator and generate kernel
        import random
        random.seed(seed)
        generator = random.choice(generators)
        kernel_source = generator(seed)
        
        # Extract kernel name from source
        import re
        match = re.search(r'def (\w+)\(', kernel_source)
        kernel_name = match.group(1) if match else f"kernel_{i}"
        
        try:
            hlo_forward, hlo_backward, hlo_optimized = extract_ir_from_source(
                kernel_source, kernel_name, i, generator.__name__
            )
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                hlo_forward=hlo_forward,
                hlo_backward=hlo_backward,
                hlo_optimized=hlo_optimized,
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
    mode: str = "both",
    extended: bool = False
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        mode: "forward", "backward", or "both"
        extended: Include extended (ML-focused) generators
    """
    generators = EXTENDED_GENERATORS if extended else GENERATORS
    
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            import random
            random.seed(seed)
            generator = random.choice(generators)
            kernel_source = generator(seed)
            
            import re
            match = re.search(r'def (\w+)\(', kernel_source)
            kernel_name = match.group(1) if match else f"kernel_{i}"
            
            try:
                hlo_forward, hlo_backward, hlo_optimized = extract_ir_from_source(
                    kernel_source, kernel_name, i, generator.__name__
                )
                
                pair = {
                    "id": generated,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "type": generator.__name__
                }
                
                if mode == "forward":
                    pair["hlo"] = hlo_forward
                elif mode == "backward":
                    if hlo_backward:
                        pair["hlo"] = hlo_backward
                    else:
                        continue  # Skip if no backward pass
                else:  # both
                    pair["hlo_forward"] = hlo_forward
                    if hlo_backward:
                        pair["hlo_backward"] = hlo_backward
                    if hlo_optimized:
                        pair["hlo_optimized"] = hlo_optimized
                
                f.write(json.dumps(pair) + '\n')
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated {generated} pairs...")
                    
            except Exception as e:
                failed += 1
                print(f"Failed pair {i}: {e}")
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
    parser.add_argument("--mode", type=str, default="both", choices=["forward", "backward", "both"],
                        help="Which HLO to include")
    parser.add_argument("--extended", action="store_true", help="Include ML-focused generators")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.mode, args.extended)
    elif args.output:
        generate_training_pairs(args.count, args.seed, args.output, args.extended)
    else:
        # Demo mode
        pairs = generate_training_pairs(args.count, args.seed, extended=args.extended)
        
        print("\n=== Sample Pairs ===")
        for pair in pairs[:3]:
            print(f"\n--- {pair.kernel_name} ({pair.generator_type}) ---")
            print("Python:")
            print(pair.python_source)
            print(f"\nHLO Forward ({len(pair.hlo_forward)} chars):")
            print(pair.hlo_forward[:500] + "..." if len(pair.hlo_forward) > 500 else pair.hlo_forward)
            if pair.hlo_backward:
                print(f"\nHLO Backward ({len(pair.hlo_backward)} chars):")
                print(pair.hlo_backward[:500] + "..." if len(pair.hlo_backward) > 500 else pair.hlo_backward)
