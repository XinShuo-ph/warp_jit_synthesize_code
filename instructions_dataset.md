# Dataset Generation and Report Instructions

## Objective
Produce 200MB of CPU code dataset and 200MB of CUDA code dataset. Write a simple report in markdown format for the chief scientist to review, introducing JIT, IR, Nvidia Warp, and the current dataset.

## Workflow

### Phase 1: CPU Code Production (200MB)
1.  **Study** `cursor/agent-work-merge-` branches.
2.  **Reproduce** code from these branches.
3.  **Pick** the best branch for production.
4.  **Produce** 200MB of data (gradually).
5.  **Push** to remote.

### Phase 2: CUDA Code Production (200MB)
1.  **Study** `cursor/cuda...` branches (or relevant branches implementing CUDA support).
2.  **Reproduce** code.
3.  **Pick** best for production.
4.  **Produce** 200MB of data (gradually).
5.  **Push** to remote.

### Phase 3: Report
1.  **Write** a markdown report (`REPORT.md`).
2.  **Content**: Introduce JIT, IR, Nvidia Warp, and the current dataset statistics.
