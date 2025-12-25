import json
import os
import re
from collections import Counter

dataset_path = "jit/data/large_dataset/dataset.jsonl"

stats = {
    "total_samples": 0,
    "strategies": Counter(),
    "avg_ir_lines": 0
}

total_ir_lines = 0

with open(dataset_path, "r") as f:
    for line in f:
        data = json.loads(line)
        stats["total_samples"] += 1
        
        code = data["python_code"]
        ir = data["ir_code"]
        
        total_ir_lines += len(ir.splitlines())
        
        # Heuristic strategy detection
        if "wp.sin" in code and "wp.sqrt" in code:
            stats["strategies"]["complex_math"] += 1
        elif "atomic_add" in code:
            stats["strategies"]["atomic_accumulate"] += 1
        elif "for i" in code and "for j" in code:
            stats["strategies"]["nested_loop"] += 1
        elif "for i" in code:
            stats["strategies"]["loop"] += 1
        elif "wp.vec3" in code:
            stats["strategies"]["vec3_op"] += 1
        elif "if val" in code:
            stats["strategies"]["conditional"] += 1
        else:
            stats["strategies"]["elementwise"] += 1

stats["avg_ir_lines"] = total_ir_lines / stats["total_samples"]

print("## Dataset Statistics")
print(f"- **Total Samples**: {stats['total_samples']}")
print(f"- **Average IR Lines**: {stats['avg_ir_lines']:.1f}")
print("### Strategy Distribution")
for strat, count in stats["strategies"].most_common():
    print(f"- {strat}: {count} ({count/stats['total_samples']*100:.1f}%)")
