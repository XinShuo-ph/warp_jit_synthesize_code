"""Synthesis Pipeline - Generates Python→XLA HLO pairs for LLM training."""
import json
import os
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass, asdict

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

import jax
import jax.numpy as jnp
from generator import generate_kernel_batch, GENERATORS


@dataclass
class TrainingPair:
    """A Python→XLA HLO training pair."""
    id: int
    kernel_name: str
    python_source: str
    hlo_code: str
    optimized_hlo: Optional[str]
    generator_type: str


def create_sample_inputs(kernel_source: str, n: int = 10):
    """
    Create sample inputs for a JAX kernel based on its signature.
    
    Args:
        kernel_source: Source code of the kernel
        n: Size of arrays to create
        
    Returns:
        Tuple of sample inputs
    """
    # Parse function signature to determine inputs
    import re
    
    # Count number of parameters
    match = re.search(r'def \w+\((.*?)\):', kernel_source)
    if not match:
        return (jnp.ones(n, dtype=jnp.float32),)
    
    params = match.group(1).split(',')
    params = [p.strip() for p in params if p.strip()]
    
    # Create sample inputs based on common parameter names
    sample_inputs = []
    for param in params:
        param_lower = param.lower()
        if 'alpha' in param_lower or 'scale' in param_lower:
            # Scalar parameter
            sample_inputs.append(jnp.float32(2.0))
        else:
            # Array parameter
            # Check if kernel uses vectors (3D operations)
            if 'vec' in kernel_source or 'dot' in kernel_source or 'norm' in kernel_source:
                # Create 2D array where last dimension is vector
                sample_inputs.append(jnp.ones((n, 3), dtype=jnp.float32))
            else:
                # Regular 1D array
                sample_inputs.append(jnp.ones(n, dtype=jnp.float32))
    
    return tuple(sample_inputs)


def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    # Create a temp file to store the kernel
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


def extract_ir_from_source(kernel_source: str, kernel_name: str, module_id: int):
    """Extract XLA HLO IR from kernel source code."""
    from ir_extractor import extract_ir
    
    module_name = f"synth_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(kernel_source, module_name)
        
        # Find the kernel function in the module
        kernel_func = None
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_') and hasattr(obj, '__module__'):
                if obj.__module__ == module_name:
                    kernel_func = obj
                    break
        
        if kernel_func is None:
            raise ValueError(f"No function found in generated source")
        
        # Create sample inputs
        sample_inputs = create_sample_inputs(kernel_source)
        
        # Extract IR
        ir = extract_ir(kernel_func, sample_inputs)
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir.hlo_text, ir.optimized_hlo
        
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
    """Generate training pairs with XLA HLO code."""
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
            hlo_code, optimized_hlo = extract_ir_from_source(kernel_source, kernel_name, i)
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                hlo_code=hlo_code,
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
    device: str = "both"
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        device: "cpu", "gpu", or "both" (JAX generates device-agnostic HLO)
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
                hlo_code, optimized_hlo = extract_ir_from_source(kernel_source, kernel_name, i)
                
                pair = {
                    "id": i,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "type": generator.__name__
                }
                
                # JAX generates HLO that can target both CPU and GPU
                # Store HLO and optionally optimized HLO
                if device == "cpu" or device == "both":
                    pair["hlo"] = hlo_code
                    if optimized_hlo:
                        pair["optimized_hlo"] = optimized_hlo
                
                # Note: JAX's HLO is device-agnostic until final compilation
                # so we don't separate CPU/GPU like Warp does
                
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
    
    parser = argparse.ArgumentParser(description="Generate Python→XLA HLO training pairs")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--device", type=str, default="both", choices=["cpu", "gpu", "both"],
                        help="Target device(s) for code generation")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.device)
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
            print(pair.hlo_code[:500] + "...")
            if pair.optimized_hlo:
                print(f"\nOptimized HLO ({len(pair.optimized_hlo)} chars):")
                print(pair.optimized_hlo[:500] + "...")
