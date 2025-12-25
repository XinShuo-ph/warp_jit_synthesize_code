import json
import os
import sys

# Add workspace to path
sys.path.append(os.getcwd())

def analyze(filepath):
    print(f"Analyzing {filepath}...")
    
    total_samples = 0
    total_py_lines = 0
    total_cpp_fwd_lines = 0
    total_cpp_bwd_lines = 0
    
    op_counts = {}
    
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip(): continue
            sample = json.loads(line)
            total_samples += 1
            
            py_src = sample["python_source"]
            cpp_fwd = sample["cpp_source_forward"]
            cpp_bwd = sample["cpp_source_backward"]
            
            total_py_lines += len(py_src.splitlines())
            total_cpp_fwd_lines += len(cpp_fwd.splitlines())
            if cpp_bwd:
                total_cpp_bwd_lines += len(cpp_bwd.splitlines())
            
            # Rough op count check in python source
            for op in ["wp.add", "wp.sub", "wp.mul", "wp.min", "wp.max", "float(", "int("]:
                if op in py_src:
                    op_counts[op] = op_counts.get(op, 0) + 1
                    
    print(f"Total Samples: {total_samples}")
    print(f"Avg Python Lines: {total_py_lines/total_samples:.2f}")
    print(f"Avg C++ Forward Lines: {total_cpp_fwd_lines/total_samples:.2f}")
    print(f"Avg C++ Backward Lines: {total_cpp_bwd_lines/total_samples:.2f}")
    print("Operation Frequency (samples containing op):")
    for op, count in op_counts.items():
        print(f"  {op}: {count} ({count/total_samples*100:.1f}%)")

if __name__ == "__main__":
    analyze("jit/data/large_scale/dataset_large.jsonl")
