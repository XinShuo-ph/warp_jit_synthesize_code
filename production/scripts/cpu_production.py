#!/usr/bin/env python3
"""
CPU Code Production Script
Target: 200MB of Python→IR training data

Based on branch agent-work-merge-9d9b
Generates ~84,195 samples at average 2.43 KB/sample
"""
import os
import sys
import json
import time
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpu_batch_generator import generate_batch


def get_dir_size_mb(directory: Path) -> float:
    """Get directory size in MB."""
    if not directory.exists():
        return 0.0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)


def count_files(directory: Path, extension: str = ".json") -> int:
    """Count files with given extension in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"*{extension}")))


def main():
    """Generate 200MB of CPU code data."""
    TARGET_MB = 200
    AVG_FILE_SIZE_KB = 2.43
    ESTIMATED_SAMPLES = int((TARGET_MB * 1024) / AVG_FILE_SIZE_KB)
    
    # Adjust for overhead and ensure we exceed target
    SAMPLES_TO_GENERATE = int(ESTIMATED_SAMPLES * 1.05)  # 5% buffer
    
    output_dir = Path("/workspace/production/cpu_code")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CPU CODE PRODUCTION - 200MB TARGET")
    print("=" * 70)
    print(f"Target size: {TARGET_MB} MB")
    print(f"Estimated samples needed: {ESTIMATED_SAMPLES:,}")
    print(f"Samples to generate (with buffer): {SAMPLES_TO_GENERATE:,}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if generation already in progress
    existing_files = count_files(output_dir)
    existing_size_mb = get_dir_size_mb(output_dir)
    
    if existing_files > 0:
        print(f"Existing progress detected:")
        print(f"  Files: {existing_files:,}")
        print(f"  Size: {existing_size_mb:.2f} MB")
        print()
        
        if existing_size_mb >= TARGET_MB:
            print(f"✓ Target already reached! ({existing_size_mb:.2f} MB >= {TARGET_MB} MB)")
            return
        
        print("Resuming generation...")
        start_index = existing_files
        remaining_samples = SAMPLES_TO_GENERATE - existing_files
    else:
        print("Starting fresh generation...")
        start_index = 0
        remaining_samples = SAMPLES_TO_GENERATE
    
    print()
    print("=" * 70)
    print("GENERATION IN PROGRESS")
    print("=" * 70)
    
    # Generate in chunks to allow monitoring
    CHUNK_SIZE = 1000
    total_start_time = time.time()
    
    generated_so_far = existing_files
    
    while True:
        current_size_mb = get_dir_size_mb(output_dir)
        
        if current_size_mb >= TARGET_MB:
            print()
            print("=" * 70)
            print("✓ TARGET REACHED!")
            print("=" * 70)
            print(f"Final size: {current_size_mb:.2f} MB")
            print(f"Total files: {count_files(output_dir):,}")
            break
        
        remaining_mb = TARGET_MB - current_size_mb
        
        # Generate next chunk
        print()
        print(f"Current: {current_size_mb:.2f} MB / {TARGET_MB} MB ({current_size_mb/TARGET_MB*100:.1f}%)")
        print(f"Remaining: {remaining_mb:.2f} MB")
        print(f"Generating next {CHUNK_SIZE} samples...")
        
        chunk_start = time.time()
        
        try:
            stats = generate_batch(
                n=CHUNK_SIZE,
                output_dir=output_dir,
                seed=42,
                chunk_size=CHUNK_SIZE,
                start_index=generated_so_far
            )
            
            generated_so_far += stats['total_pairs']
            
            chunk_time = time.time() - chunk_start
            chunk_rate = stats['total_pairs'] / chunk_time if chunk_time > 0 else 0
            
            print(f"  Chunk completed: {stats['total_pairs']} pairs in {chunk_time:.1f}s ({chunk_rate:.1f} pairs/sec)")
            
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user.")
            print(f"Current progress: {get_dir_size_mb(output_dir):.2f} MB")
            print(f"Files generated: {count_files(output_dir):,}")
            sys.exit(1)
        except Exception as e:
            print(f"\nError during generation: {e}")
            print("Continuing with next chunk...")
            continue
    
    total_time = time.time() - total_start_time
    total_files = count_files(output_dir)
    final_size = get_dir_size_mb(output_dir)
    avg_rate = total_files / total_time if total_time > 0 else 0
    
    print()
    print("=" * 70)
    print("GENERATION STATISTICS")
    print("=" * 70)
    print(f"Total size: {final_size:.2f} MB")
    print(f"Total files: {total_files:,}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Average rate: {avg_rate:.1f} samples/sec")
    print(f"Average file size: {final_size*1024/total_files:.2f} KB")
    print()
    
    # Save final statistics
    final_stats = {
        "target_mb": TARGET_MB,
        "actual_mb": final_size,
        "total_files": total_files,
        "total_time_sec": total_time,
        "avg_rate_per_sec": avg_rate,
        "avg_file_size_kb": final_size*1024/total_files,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    stats_file = output_dir / "final_production_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_file}")
    print()
    print("✓ CPU code production complete!")


if __name__ == "__main__":
    main()
