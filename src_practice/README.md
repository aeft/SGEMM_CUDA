# SGEMM CUDA Practice

This directory contains my practice implementations of SGEMM kernels.

## Directory Structure

```
src_practice/
├── kernels/           # Your kernel implementations
│   ├── 1_naive.cuh
│   └── ...
├── kernels.cuh        # Include all kernels (enable one by one)
├── runner.cu          # Copied from src/ (same as original)
└── runner.cuh
```

## How to Use

### 1. Build

```bash
# In project root directory
make                    # Build both sgemm and sgemm_practice
```

### 2. Run

```bash
# Quick test (build + run)
make practice KERNEL=1     # Test your practice version
make test KERNEL=1         # Test original (for comparison)
```

### 3. Implement a New Kernel

1. Create or edit `.cuh` file in `kernels/` directory
2. Uncomment corresponding `#include` in `kernels.cuh`
3. Reference original implementation in `../src/kernels/`
4. Build: `make`
5. Test: `./build/sgemm_practice <kernel_num>`

### 4. Compare with Original

```bash
# Performance comparison (quick)
make compare KERNEL=1              # Compare both versions
make compare KERNEL="1 2 3"        # Compare multiple kernels
make compare KERNEL="1 2" PRACTICE_ONLY=1  # Only run practice version
```

### 5. Full benchmark
```bash
make bench
```

## Kernel Progress

- [ ] Kernel 1: Naive
- [ ] Kernel 2: Global Memory Coalescing
- [ ] Kernel 3: Shared Memory Blocking
- [ ] Kernel 4: 1D Blocktiling
- [ ] Kernel 5: 2D Blocktiling
- [ ] Kernel 6: Vectorized Memory Access
- [ ] Kernel 7: Resolve Bank Conflicts (Linearize)
- [ ] Kernel 8: Resolve Bank Conflicts (Offset)
- [ ] Kernel 9: Autotuning
- [ ] Kernel 10: Warptiling
- [ ] Kernel 11: Double Buffering

## Learning Notes

### Kernel 1: Naive Implementation
...