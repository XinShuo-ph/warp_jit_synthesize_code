"""Synthesis Pipeline - Generates Python→C++ pairs for LLM training."""
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

import warp as wp
from generator import generate_kernel_batch, GENERATORS


@dataclass
class TrainingPair:
    """A Python→C++ training pair."""
    id: int
    kernel_name: str
    python_source: str
    cpp_code: str
    generator_type: str
    

def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    # Create a temp file to store the kernel
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import warp as wp\n\n")
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


def extract_ir_from_source(kernel_source: str, kernel_name: str, module_id: int) -> Optional[str]:
    """Extract C++ IR from kernel source code."""
    from ir_extractor import extract_ir
    
    module_name = f"synth_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(kernel_source, module_name)
        
        # Find the kernel in the module
        kernel = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, wp.Kernel):
                kernel = obj
                break
        
        if kernel is None:
            raise ValueError(f"No kernel found in generated source")
        
        # Extract IR
        ir = extract_ir(kernel)
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir.cpp_code
        
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
    """Generate training pairs."""
    wp.init()
    
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
            cpp_code = extract_ir_from_source(kernel_source, kernel_name, i)
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                cpp_code=cpp_code,
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
    base_seed: int = 42
):
    """Generate pairs and save as JSONL (one JSON per line)."""
    wp.init()
    
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
                cpp_code = extract_ir_from_source(kernel_source, kernel_name, i)
                
                pair = {
                    "id": i,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "cpp": cpp_code,
                    "type": generator.__name__
                }
                
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed)
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
            print(f"\nC++ ({len(pair.cpp_code)} chars):")
            print(pair.cpp_code[:500] + "...")
