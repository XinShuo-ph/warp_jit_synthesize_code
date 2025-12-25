import warp as wp
import os
from pipeline import generate_pair
import time

def generate_dataset(count=10000):
    wp.init()
    
    print(f"Generating {count} samples...")
    start_time = time.time()
    
    success_count = 0
    for i in range(count):
        res = generate_pair()
        if res:
            success_count += 1
            if i % 100 == 0:
                print(f"Generated {i+1}/{count}")
        else:
            print(f"Failed at {i}")
            
    duration = time.time() - start_time
    print(f"Finished. Generated {success_count} samples in {duration:.2f}s")

if __name__ == "__main__":
    generate_dataset()
