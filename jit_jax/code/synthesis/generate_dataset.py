import os
import json
from pipeline import synthesize_pair
import tqdm

OUTPUT_DIR = "/workspace/jit_jax/data/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    count = 0
    target = 100
    seed = 0
    
    pbar = tqdm.tqdm(total=target)
    
    while count < target:
        try:
            res = synthesize_pair(seed)
            if res:
                # Save to file
                filename = os.path.join(OUTPUT_DIR, f"sample_{count:04d}.json")
                with open(filename, 'w') as f:
                    json.dump(res, f, indent=2)
                
                count += 1
                pbar.update(1)
            
            seed += 1
        except Exception as e:
            print(f"Error on seed {seed}: {e}")
            seed += 1
            
    pbar.close()
    print(f"Generated {count} samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
