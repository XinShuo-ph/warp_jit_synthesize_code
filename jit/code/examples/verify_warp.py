import warp as wp

try:
    wp.init()
    print("Warp initialized successfully!")
    print(f"Warp version: {wp.config.version}")
    
    # Check if we are running on CPU or GPU (just for info)
    device = wp.get_device("cpu")
    print(f"Default device: {device}")
    
except Exception as e:
    print(f"Failed to initialize Warp: {e}")
    exit(1)
