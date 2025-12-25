import random
import textwrap

class KernelGenerator:
    def __init__(self):
        self.ops = ['+', '-', '*', '/']
        self.funcs = ['wp.sin', 'wp.cos', 'wp.exp', 'wp.abs']
        self.vars = ['v0', 'v1', 'v2', 'tmp']

    def generate_expression(self, depth=0):
        if depth > 2:
            return f"{random.choice(self.vars)} + {random.random():.2f}"
        
        choice = random.random()
        if choice < 0.4:
            # Binary op
            op = random.choice(self.ops)
            left = self.generate_expression(depth + 1)
            right = self.generate_expression(depth + 1)
            return f"({left} {op} {right})"
        elif choice < 0.7:
            # Func
            func = random.choice(self.funcs)
            arg = self.generate_expression(depth + 1)
            return f"{func}({arg})"
        else:
            # Variable or const
            if random.random() < 0.5:
                return random.choice(self.vars)
            else:
                return f"{random.random():.2f}"

    def generate_kernel(self, name="random_kernel"):
        # Fixed signature for simplicity for now: (data: wp.array(dtype=float))
        
        lines = []
        lines.append("import warp as wp")
        lines.append("")
        lines.append("@wp.kernel")
        lines.append(f"def {name}(data: wp.array(dtype=float)):")
        lines.append("    tid = wp.tid()")
        
        # Init vars
        lines.append("    v0 = data[tid]")
        lines.append("    v1 = 0.0")
        lines.append("    v2 = 1.0")
        lines.append("    tmp = 0.0")
        
        # Generate random statements
        num_stmts = random.randint(3, 8)
        for _ in range(num_stmts):
            target = random.choice(self.vars)
            expr = self.generate_expression()
            lines.append(f"    {target} = {expr}")
            
        # Store result
        lines.append("    data[tid] = v0")
        
        return "\n".join(lines)

if __name__ == "__main__":
    gen = KernelGenerator()
    print(gen.generate_kernel())
