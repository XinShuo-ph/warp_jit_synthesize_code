"""Synthesis Pipeline - Generates Python(JAX)→HLO pairs for LLM training."""
import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from generator import generate_kernel_batch, GENERATORS
from ir_extractor import extract_ir

@dataclass
class TrainingPair:
    """A Python(JAX)→HLO training pair."""
    id: int
    kernel_name: str
    python_source: str
    hlo_fwd: str
    hlo_bwd: Optional[str]
    generator_type: str


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
            ir = extract_ir(kernel_source, kernel_name, i)
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                hlo_fwd=ir.hlo_fwd,
                hlo_bwd=ir.hlo_bwd,
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
    device: str = "both" # ignored for JAX HLO usually, as we just get HLO
):
    """Generate pairs and save as JSONL."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

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
                ir = extract_ir(kernel_source, kernel_name, i)
                
                pair = {
                    "id": i,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "hlo_fwd": ir.hlo_fwd,
                    "hlo_bwd": ir.hlo_bwd,
                    "type": generator.__name__
                }
                
                f.write(json.dumps(pair) + '\n')
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated {generated} pairs...")
                    
            except Exception as e:
                failed += 1
                # print(f"Failed: {e}")
                continue
    
    print(f"\nTotal: {generated} pairs saved to {output_path}, {failed} failed")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Python(JAX)→HLO training pairs")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--device", type=str, default="both", choices=["cpu", "cuda", "both"],
                        help="Target device (ignored for JAX HLO)")
    
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
            print(f"\nHLO Fwd ({len(pair.hlo_fwd)} chars):")
            print(pair.hlo_fwd[:500] + "...")
            if pair.hlo_bwd:
                print(f"\nHLO Bwd ({len(pair.hlo_bwd)} chars):")
                print(pair.hlo_bwd[:500] + "...")
