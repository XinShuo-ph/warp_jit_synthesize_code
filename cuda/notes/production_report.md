================================================================================
CUDA KERNEL DATASET ANALYSIS
================================================================================
Dataset: /workspace/cuda/data/production
Total kernels: 500

CATEGORY DISTRIBUTION
--------------------------------------------------------------------------------
  arithmetic          :   46 (  9.2%)
  atomic              :   46 (  9.2%)
  transform           :   46 (  9.2%)
  stencil             :   46 (  9.2%)
  reduction           :   46 (  9.2%)
  grid_3d             :   45 (  9.0%)
  matrix              :   45 (  9.0%)
  vector              :   45 (  9.0%)
  grid_2d             :   45 (  9.0%)
  control_flow        :   45 (  9.0%)
  math                :   45 (  9.0%)

BACKWARD PASS COVERAGE
--------------------------------------------------------------------------------
  Total kernels:        500
  With backward pass:   500
  Coverage:             100.0%

CUDA PATTERN USAGE
--------------------------------------------------------------------------------
  blockDim            :  500 (100.0%)
  blockIdx            :  500 (100.0%)
  threadIdx           :  500 (100.0%)
  gridDim             :  500 (100.0%)
  shared_memory       :  500 (100.0%)
  grid_stride         :  500 (100.0%)
  atomic_ops          :   92 ( 18.4%)
  syncthreads         :    0 (  0.0%)

DEVICE DISTRIBUTION
--------------------------------------------------------------------------------
  cuda                :  500 (100.0%)

COMPLEXITY METRICS
--------------------------------------------------------------------------------
  Python source length (chars):
    min       :      124
    max       :      612
    mean      :      252
    median    :      180
  IR forward length (chars):
    min       :     1219
    max       :    12961
    mean      :     2569
    median    :     1552

TOP 10 OPERATIONS
--------------------------------------------------------------------------------
   1. wp.sqrt                  :   48
   2. wp.abs                   :   45
   3. wp.atomic_add            :   37
   4. wp.atomic_min            :   29
   5. wp.atomic_max            :   26
   6. wp.sin                   :   21
   7. wp.cos                   :   20
   8. wp.normalize             :   16
   9. wp.min                   :   13
  10. wp.length                :   13

================================================================================