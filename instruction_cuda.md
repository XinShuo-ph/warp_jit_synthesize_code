# Objective
Adapt the current production code to use a CUDA-backed JAX runtime (i.e., CUDA-enabled `jaxlib`).
Currently no GPU device is available for the agent to use. Provide concise code changes and commands for you to test on a GPU device yourself.

# Milestones

1. Reproduce the current production code using JAX on CPU. Study the `cursor/agent-work-merge-process-....` branches and pick the best branch to use as base.

2. Adapt the code to run on CUDA via JAX. Do this iteratively, iterate over
- All program/kernel pattern types
- both forward and backward pass
- batch generation pipeline
- Validation tools
- Test suite for me to run later on actual GPU device.
Since there are ~10 pattern types that I can see currently, I expect ~20 iterations to complete this milestone. Focus on one iteration, one task at a time.


