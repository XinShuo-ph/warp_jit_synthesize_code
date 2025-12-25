import warp as wp
from warp.examples.fem.example_diffusion import Example
import sys

# Suppress graphical output
sys.argv.append("--headless")

def run_test():
    print("Initializing Warp...")
    wp.init()
    
    print("Creating Example with low resolution...")
    # Lower resolution for speed
    example = Example(resolution=10, quiet=True)
    
    print("Computing...")
    # The example class usually has a method to run the simulation
    # Inspecting the file, it has a 'compute' or similar?
    # Let's check the code I read earlier or assume standard structure.
    # Looking at lines 72+, it has __init__.
    # I didn't read the whole file. Let's assume there is a method to run it.
    
    # Typically examples have a render() or step() or compute()
    # Let's inspect the object
    if hasattr(example, 'step'):
        example.step()
    elif hasattr(example, 'render'):
        example.render()
    else:
        # Check for other methods
        print(dir(example))

if __name__ == "__main__":
    try:
        run_test()
        print("Test complete.")
    except Exception as e:
        print(f"Test failed: {e}")
