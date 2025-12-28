# Objective
Adapt the current production code to use CUDA backend.
Currently no GPU device is available for agent to use. Provide concise code and command for me to test on GPU device by myself.

# Milestones

1. Reproduce the current production code using cpu backend. Study the `cursor/agent-work-merge-process-....` branches and pick one best branch to use as base.

2. Adapt the code to use CUDA backend. Do this iteratively, iterate over
- All kernel types
- both forward and backward pass
- batch generation pipeline
- Validation tools
- Test suite for me to run later on actual GPU device.
Since there are ~10 kernel types that I can see currently, I expect ~ 20 iterations to complete this milestone. Focus on one iteration, one task at a time.


