import random
import warp as wp

class KernelGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        
    def generate_kernel(self, name="generated_kernel"):
        # 1. Inputs
        num_inputs = self.rng.randint(1, 5)
        inputs = []
        input_vars = []
        input_types = {} # name -> 'float' or 'int' or 'array_float'
        
        # Always have at least one array for output/processing
        inputs.append(f"data: wp.array(dtype=float)")
        input_vars.append("data")
        input_types["data"] = "array_float"
        
        for i in range(num_inputs):
            arg_name = f"arg_{i}"
            is_float = self.rng.choice([True, False])
            is_array = self.rng.choice([True, False])
            
            if is_array:
                dtype = "float" if is_float else "int"
                inputs.append(f"{arg_name}: wp.array(dtype={dtype})")
                input_types[arg_name] = f"array_{dtype}"
            else:
                dtype = "float" if is_float else "int"
                inputs.append(f"{arg_name}: {dtype}")
                input_types[arg_name] = dtype
                
            input_vars.append(arg_name)
            
        input_str = ", ".join(inputs)
        
        # 2. Body
        lines = []
        lines.append("    tid = wp.tid()")
        
        # Local vars: name -> type
        local_vars = {"tid": "int"}
        
        # Generate some math ops
        num_ops = self.rng.randint(5, 15)
        
        for i in range(num_ops):
            target = f"t{i}"
            
            # Weighted choice of operations
            op_choices = ["assign", "math", "read_array", "write_array", "cast"]
            weights = [10, 40, 20, 20, 10]
            op_type = self.rng.choices(op_choices, weights=weights, k=1)[0]
            
            if op_type == "assign":
                dtype = self.rng.choice(["float", "int"])
                if dtype == "float":
                    val = self.rng.uniform(-10, 10)
                    lines.append(f"    {target} = float({val:.2f})")
                else:
                    val = self.rng.randint(0, 10)
                    lines.append(f"    {target} = int({val})")
                local_vars[target] = dtype
                
            elif op_type == "math":
                if not local_vars: continue
                # Pick two vars
                v1_name = self.rng.choice(list(local_vars.keys()))
                v2_name = self.rng.choice(list(local_vars.keys()))
                t1 = local_vars[v1_name]
                t2 = local_vars[v2_name]
                
                # Only operate if types match
                if t1 != t2:
                    # Try to pick compatible
                    candidates = [v for v, t in local_vars.items() if t == t1]
                    if candidates:
                        v2_name = self.rng.choice(candidates)
                        t2 = t1
                    else:
                        continue
                
                op = self.rng.choice(["+", "-", "*", "max", "min"])
                
                if op in ["max", "min"]:
                    lines.append(f"    {target} = wp.{op}({v1_name}, {v2_name})")
                else:
                    lines.append(f"    {target} = {v1_name} {op} {v2_name}")
                local_vars[target] = t1
                
            elif op_type == "cast":
                if not local_vars: continue
                v_name = self.rng.choice(list(local_vars.keys()))
                t_in = local_vars[v_name]
                if t_in == "int":
                    lines.append(f"    {target} = float({v_name})")
                    local_vars[target] = "float"
                else:
                    lines.append(f"    {target} = int({v_name})")
                    local_vars[target] = "int"

            elif op_type == "read_array":
                # Find an array input
                arrays = [v for v, t in input_types.items() if "array" in t]
                if arrays:
                    arr = self.rng.choice(arrays)
                    arr_type = input_types[arr]
                    elem_type = arr_type.split("_")[1] # float or int
                    
                    # Index
                    idx_vars = [v for v, t in local_vars.items() if t == "int"]
                    if not idx_vars: continue
                    idx = self.rng.choice(idx_vars)
                    
                    lines.append(f"    {target} = {arr}[{idx}]")
                    local_vars[target] = elem_type
            
            elif op_type == "write_array":
                 # Find an array input
                arrays = [v for v, t in input_types.items() if "array" in t]
                if arrays:
                    arr = self.rng.choice(arrays)
                    arr_type = input_types[arr]
                    elem_type = arr_type.split("_")[1]
                    
                    # Value to write
                    val_candidates = [v for v, t in local_vars.items() if t == elem_type]
                    if not val_candidates: continue
                    val = self.rng.choice(val_candidates)
                    
                    # Index
                    idx_vars = [v for v, t in local_vars.items() if t == "int"]
                    if not idx_vars: continue
                    idx = self.rng.choice(idx_vars)
                    
                    lines.append(f"    {arr}[{idx}] = {val}")

        body_str = "\n".join(lines)
        
        source = f"""
@wp.kernel
def {name}({input_str}):
{body_str}
"""
        return source

if __name__ == "__main__":
    gen = KernelGenerator(seed=42)
    print(gen.generate_kernel())
