"""JAX Synthesis Pipeline - Generates Python→HLO pairs for LLM training."""
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
from jax_generator import generate_kernel_batch, GENERATORS


@dataclass
class TrainingPair:
    """A Python→HLO training pair."""
    id: int
    kernel_name: str
    python_source: str
    hlo_code: str
    hlo_optimized: Optional[str]
    generator_type: str
    

def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    # Create a temp file to store the function
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n")
        f.write("import jax.lax\n\n")
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


def get_sample_inputs_for_generator(generator_name: str, seed: int = 42) -> tuple:
    """Get appropriate sample inputs based on generator type."""
    key = jax.random.PRNGKey(seed)
    
    if 'elementwise' in generator_name:
        a = jax.random.normal(key, (16,))
        b = jax.random.normal(jax.random.PRNGKey(seed + 1), (16,))
        return (a, b)
    elif 'scalar_arr' in generator_name:
        x = jax.random.normal(key, (16,))
        y = jax.random.normal(jax.random.PRNGKey(seed + 1), (16,))
        return (2.0, x, y)
    elif 'unary' in generator_name:
        # Use positive values for sqrt
        a = jnp.abs(jax.random.normal(key, (16,))) + 0.1
        return (a,)
    elif 'branch' in generator_name or 'nested' in generator_name:
        a = jax.random.normal(key, (16,))
        return (a,)
    elif 'loop' in generator_name:
        a = jax.random.normal(key, (16,))
        return (a, 5)
    elif 'reduce' in generator_name:
        a = jax.random.normal(key, (16,))
        return (a,)
    elif 'vec' in generator_name:
        # Vector kernel can be 1-arg (norm) or 2-arg (dot)
        # We need to inspect the source to determine which
        a = jax.random.normal(key, (8, 3))
        b = jax.random.normal(jax.random.PRNGKey(seed + 1), (8, 3))
        return (a, b)  # Will be handled specially in extract_ir_from_source
    elif 'multi' in generator_name:
        a = jnp.abs(jax.random.normal(key, (16,))) + 0.1
        b = jnp.abs(jax.random.normal(jax.random.PRNGKey(seed + 1), (16,))) + 0.1
        return (a, b)
    elif 'compound' in generator_name:
        a = jax.random.normal(key, (16,))
        b = jax.random.normal(jax.random.PRNGKey(seed + 1), (16,))
        return (a, b, 2.5)
    elif 'matmul' in generator_name:
        a = jax.random.normal(key, (8, 8))
        b = jax.random.normal(jax.random.PRNGKey(seed + 1), (8, 8))
        return (a, b)
    elif 'softmax' in generator_name:
        x = jax.random.normal(key, (8, 10))
        return (x,)
    elif 'scan' in generator_name:
        x = jax.random.normal(key, (16,))
        return (x,)
    elif 'vmap' in generator_name:
        a = jax.random.normal(key, (8, 4))
        b = jax.random.normal(jax.random.PRNGKey(seed + 1), (8, 4))
        return (a, b)
    elif 'grad' in generator_name:
        x = jax.random.normal(key, (16,))
        return (x,)
    else:
        # Default fallback
        a = jax.random.normal(key, (16,))
        return (a,)


def extract_ir_from_source(kernel_source: str, kernel_name: str, module_id: int, 
                           generator_name: str = ""):
    """Extract HLO IR from JAX function source code."""
    from jax_ir_extractor import extract_ir, extract_ir_with_grad
    
    module_name = f"synth_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(kernel_source, module_name)
        
        # Find the main function in the module (the one that starts with the expected prefix)
        func = None
        func_name = kernel_name
        
        for name in dir(module):
            if name.startswith('_'):
                continue
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_'):
                # Prefer the decorated function (usually the main one)
                if hasattr(obj, '__wrapped__') or (hasattr(obj, 'lower')):
                    func = obj
                    func_name = name
                    break
        
        # If no jit-decorated function found, look for the kernel name
        if func is None:
            for name in dir(module):
                if name.startswith('_'):
                    continue
                obj = getattr(module, name)
                if callable(obj) and kernel_name in name:
                    func = obj
                    func_name = name
                    break
        
        # Last resort: find any callable
        if func is None:
            for name in dir(module):
                if name.startswith('_'):
                    continue
                obj = getattr(module, name)
                if callable(obj) and name not in ('jax', 'jnp'):
                    func = obj
                    func_name = name
                    break
        
        if func is None:
            raise ValueError(f"No function found in generated source")
        
        # Get sample inputs based on generator type
        sample_inputs = get_sample_inputs_for_generator(generator_name)
        
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
        
        # Try to get gradient IR if function returns scalar-like output
        hlo_optimized = ir.hlo_optimized
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir.hlo_text, hlo_optimized
        
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
    output_dir: Optional[str] = None
) -> List[TrainingPair]:
    """Generate training pairs with HLO code."""
    pairs = []
    failed = 0
    
    for i in range(count):
        seed = base_seed + i
        
        # Pick a generator and generate kernel
        import random
        random.seed(seed)
        generator = random.choice(GENERATORS)
        kernel_source = generator(seed)
        
        # Extract kernel name from source
        import re
        match = re.search(r'def (\w+)\(', kernel_source)
        kernel_name = match.group(1) if match else f"kernel_{i}"
        
        try:
            hlo_code, hlo_optimized = extract_ir_from_source(
                kernel_source, kernel_name, i, generator.__name__
            )
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                hlo_code=hlo_code,
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
    output_type: str = "hlo"
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        output_type: "hlo", "optimized", or "both"
    """
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            import random
            random.seed(seed)
            generator = random.choice(GENERATORS)
            kernel_source = generator(seed)
            
            import re
            match = re.search(r'def (\w+)\(', kernel_source)
            kernel_name = match.group(1) if match else f"kernel_{i}"
            
            try:
                hlo_code, hlo_optimized = extract_ir_from_source(
                    kernel_source, kernel_name, i, generator.__name__
                )
                
                pair = {
                    "id": i,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "type": generator.__name__
                }
                
                if output_type == "hlo":
                    pair["hlo"] = hlo_code
                elif output_type == "optimized":
                    if hlo_optimized:
                        pair["hlo_optimized"] = hlo_optimized
                    else:
                        pair["hlo"] = hlo_code
                else:  # both
                    pair["hlo"] = hlo_code
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
    
    parser = argparse.ArgumentParser(description="Generate Python→HLO training pairs using JAX")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--output-type", type=str, default="both", 
                        choices=["hlo", "optimized", "both"],
                        help="Type of HLO output")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.output_type)
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
            print(pair.hlo_code[:800] + "..." if len(pair.hlo_code) > 800 else pair.hlo_code)
