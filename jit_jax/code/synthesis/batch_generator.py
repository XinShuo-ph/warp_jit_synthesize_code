import os
import json
from pipeline import synthesize_pair
import tqdm
import multiprocessing

OUTPUT_DIR = "/workspace/jit_jax/data/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_one(seed):
    try:
        return synthesize_pair(seed)
    except:
        return None

def main():
    target = 1000
    current_count = len([name for name in os.listdir(OUTPUT_DIR) if name.endswith('.json')])
    needed = target - current_count
    
    if needed <= 0:
        print(f"Already have {current_count} samples. Exiting.")
        return

    print(f"Generating {needed} more samples...")
    
    # Simple loop is fast enough for 1000
    pbar = tqdm.tqdm(total=needed)
    seed = current_count # Start seed where we left off (roughly)
    
    count = 0
    while count < needed:
        res = generate_one(seed)
        if res:
            filename = os.path.join(OUTPUT_DIR, f"sample_{current_count + count:05d}.json")
            with open(filename, 'w') as f:
                json.dump(res, f, indent=2)
            count += 1
            pbar.update(1)
        seed += 1
            
    pbar.close()
    print(f"Finished. Total samples: {current_count + count}")

if __name__ == "__main__":
    main()
