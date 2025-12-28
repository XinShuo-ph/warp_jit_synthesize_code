"""
Example 3: Kernel with control flow and conditionals
"""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def classify_and_process(values: wp.array(dtype=float),
                         categories: wp.array(dtype=int),
                         processed: wp.array(dtype=float)):
    """
    Classify values and apply different operations based on category
    Category 0: negative numbers -> square
    Category 1: zero -> set to 1.0
    Category 2: positive numbers -> square root
    """
    tid = wp.tid()
    val = values[tid]
    
    if val < 0.0:
        categories[tid] = 0
        processed[tid] = val * val
    elif val == 0.0:
        categories[tid] = 1
        processed[tid] = 1.0
    else:
        categories[tid] = 2
        processed[tid] = wp.sqrt(val)

@wp.kernel
def conditional_sum(values: wp.array(dtype=float),
                    threshold: float,
                    result: wp.array(dtype=float)):
    """Sum only values above threshold using atomic operations"""
    tid = wp.tid()
    val = values[tid]
    
    if val > threshold:
        wp.atomic_add(result, 0, val)

def run_example():
    # Test data with negative, zero, and positive values
    values_np = np.array([-4.0, -2.0, 0.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    n = len(values_np)
    
    # Arrays for classification example
    values = wp.array(values_np, dtype=wp.float32)
    categories = wp.zeros(n, dtype=wp.int32)
    processed = wp.zeros(n, dtype=wp.float32)
    
    # Launch classification kernel
    wp.launch(classify_and_process, dim=n, inputs=[values, categories, processed])
    
    # Get results
    categories_result = categories.numpy()
    processed_result = processed.numpy()
    
    print("Example 3: Control Flow and Conditionals")
    print("\nPart 1: Classification and Processing")
    print(f"{'Value':<10} {'Category':<10} {'Processed':<10} {'Expected':<10}")
    print("-" * 40)
    
    for i, val in enumerate(values_np):
        cat = categories_result[i]
        proc = processed_result[i]
        if val < 0:
            expected = val * val
        elif val == 0:
            expected = 1.0
        else:
            expected = np.sqrt(val)
        print(f"{val:<10.2f} {cat:<10} {proc:<10.4f} {expected:<10.4f}")
    
    # Test conditional sum
    threshold = 5.0
    sum_result = wp.zeros(1, dtype=wp.float32)
    values2 = wp.array(values_np, dtype=wp.float32)
    
    wp.launch(conditional_sum, dim=n, inputs=[values2, threshold, sum_result])
    
    sum_value = sum_result.numpy()[0]
    expected_sum = np.sum(values_np[values_np > threshold])
    
    print(f"\nPart 2: Conditional Sum (threshold > {threshold})")
    print(f"Values above threshold: {values_np[values_np > threshold]}")
    print(f"Computed sum: {sum_value:.4f}")
    print(f"Expected sum: {expected_sum:.4f}")
    print(f"Match: {np.isclose(sum_value, expected_sum)}")
    
    return np.isclose(sum_value, expected_sum)

if __name__ == "__main__":
    success = run_example()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
