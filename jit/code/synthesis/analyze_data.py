import os
import glob
import statistics

DATA_DIR = "jit/data/samples"

def analyze():
    py_files = glob.glob(os.path.join(DATA_DIR, "*.py"))
    hlo_files = glob.glob(os.path.join(DATA_DIR, "*.hlo"))
    
    count = len(py_files)
    if count == 0:
        print("No data found.")
        return

    py_sizes = [os.path.getsize(f) for f in py_files]
    hlo_sizes = [os.path.getsize(f) for f in hlo_files]
    
    stats_md = f"""# Data Statistics

- **Total Samples**: {count}
- **Python Source Size**:
  - Avg: {statistics.mean(py_sizes):.1f} bytes
  - Max: {max(py_sizes)} bytes
- **HLO IR Size**:
  - Avg: {statistics.mean(hlo_sizes):.1f} bytes
  - Max: {max(hlo_sizes)} bytes

Generated on {os.popen('date').read().strip()}
"""
    
    with open("jit/notes/data_stats.md", "w") as f:
        f.write(stats_md)
    print("Stats written to jit/notes/data_stats.md")

if __name__ == "__main__":
    analyze()
