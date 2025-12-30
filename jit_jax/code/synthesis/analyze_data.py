import os
import json
import collections
import statistics

DATA_DIR = "/workspace/jit_jax/data/samples"

def analyze():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"Analyzing {len(files)} files...")
    
    code_lengths = []
    jaxpr_lengths = []
    hlo_lengths = []
    op_counts = collections.Counter()
    
    for fname in files:
        with open(os.path.join(DATA_DIR, fname), 'r') as f:
            data = json.load(f)
            
        code = data['code']
        jaxpr = data['jaxpr']
        hlo = data['hlo']
        
        code_lengths.append(len(code.splitlines()))
        jaxpr_lengths.append(len(jaxpr.splitlines()))
        hlo_lengths.append(len(hlo.splitlines()))
        
        # Simple string search for ops
        for line in code.splitlines():
            if "jnp." in line:
                op = line.split("jnp.")[1].split("(")[0]
                op_counts[op] += 1
                
    print("\n--- Statistics ---")
    print(f"Avg Code Lines: {statistics.mean(code_lengths):.2f}")
    print(f"Avg Jaxpr Lines: {statistics.mean(jaxpr_lengths):.2f}")
    print(f"Avg HLO Lines: {statistics.mean(hlo_lengths):.2f}")
    
    print("\n--- Top Operations ---")
    for op, count in op_counts.most_common(10):
        print(f"{op}: {count}")
        
    # Write to notes
    with open("/workspace/jit_jax/notes/data_stats.md", 'w') as f:
        f.write("# Dataset Statistics\n\n")
        f.write(f"- **Total Samples**: {len(files)}\n")
        f.write(f"- **Avg Code Lines**: {statistics.mean(code_lengths):.2f}\n")
        f.write(f"- **Avg Jaxpr Lines**: {statistics.mean(jaxpr_lengths):.2f}\n")
        f.write(f"- **Avg HLO Lines**: {statistics.mean(hlo_lengths):.2f}\n\n")
        f.write("## Top Operations\n")
        for op, count in op_counts.most_common(10):
            f.write(f"- {op}: {count}\n")

if __name__ == "__main__":
    analyze()
