#!/bin/bash
# test_on_gpu.sh - Script for testing CUDA dataset generation on GPU-enabled machine

echo "=================================================="
echo "CUDA Dataset Generation Test"
echo "=================================================="

# Check if CUDA is available
python3 -c "import warp as wp; wp.init(); print(f'CUDA Available: {wp.is_cuda_available()}'); print(f'Devices: {wp.get_devices()}')"

if [ $? -ne 0 ]; then
    echo "❌ Error: Warp not installed or CUDA not available"
    exit 1
fi

# Test small batch generation
echo ""
echo "Generating 10 test samples..."
python3 code/pipeline.py --count 10 --output data/test_cuda --seed 1000 --device cuda

if [ $? -eq 0 ]; then
    echo "✅ Test generation successful!"
    echo ""
    echo "Sample statistics:"
    find data/test_cuda -name "*.json" | wc -l
    du -sh data/test_cuda
    
    echo ""
    echo "To generate full 200MB dataset, run:"
    echo "python3 code/pipeline.py --count 30000 --output data/large_cuda --seed 2000"
else
    echo "❌ Test generation failed"
    exit 1
fi
