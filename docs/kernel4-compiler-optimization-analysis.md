# Compiler Optimization Analysis for Kernel 4

## Methodology

This document describes the commands and approach used to analyze compiler optimizations for the 1D blocktiling SGEMM kernel.

## Commands Used

### 1. Generate PTX Intermediate Representation

```bash
module load cuda/12.1
nvcc -ptx -arch=sm_80 --ptxas-options=-v src_practice/runner.cu -o /tmp/kernel4.ptx
```

This generates PTX (Parallel Thread Execution) code, NVIDIA's intermediate representation before final compilation.

### 2. Generate SASS Assembly

```bash
module load cuda/12.1
nvcc -cubin -arch=sm_80 src_practice/runner.cu -o /tmp/kernel4.cubin
cuobjdump -sass /tmp/kernel4.cubin > /tmp/kernel4.sass
```

This produces SASS (Shader Assembly), the actual machine code executed by the GPU.

### 3. Extract Kernel-Specific Information

```bash
# Find kernel function in PTX
grep -n "Sgemm1DBlocktiling" /tmp/kernel4.ptx

# Extract register declarations
sed -n '471,665p' /tmp/kernel4.ptx | grep "\.reg"

# Find FMA (fused multiply-add) instructions
sed -n '471,665p' /tmp/kernel4.ptx | grep -B 5 -A 10 "fma.rn.f32"

# Extract SASS computation loop
cuobjdump -sass /tmp/kernel4.cubin | grep -A 200 "Sgemm1DBlocktiling"
```

## Analysis Findings

### Register Usage

From PTX register declarations:
```
.reg .pred   %p<4>;      // 4 predicate registers
.reg .f32    %f<127>;    // 127 float registers (virtual, reused)
.reg .b32    %r<50>;     // 50 32-bit integer registers
.reg .b64    %rd<21>;    // 21 64-bit registers
```

Actual register usage per thread:
- localC[8]: 8 registers (persistent throughout computation)
- tmpB values: 4-8 registers (loaded on-demand, reused)
- sharedA temporary values: 4-8 registers (loaded, used, released)
- Total: approximately 20-25 registers per thread

### Compiler Optimizations Identified

#### 1. Full Loop Unrolling

Both loops are completely unrolled: 8x8 = 64 iterations explicitly generated.

Evidence: Sequential FMA instructions without loop control (ISETP/BRA) patterns.

#### 2. Instruction Reordering

The compiler reorders memory accesses based on data dependencies, not source code order.

Key insight: `sharedA[(innerARow * BK + im) * BK + it]` appears to have stride BK in source order, but across all (it, im) pairs, accessed addresses are sequential: [0, 1, 2, ..., 63].

The compiler recognizes this and reorders instructions to access memory sequentially.

#### 3. Vectorized Memory Access

Sequential access enables vectorized loads:

```sass
LDS.128 R4, [R2]        // Load sharedA[0,1,2,3]
LDS.128 R8, [R2+0x10]   // Load sharedA[4,5,6,7]
LDS.128 R16, [R2+0x20]  // Load sharedA[8,9,10,11]
```

This reduces shared memory transactions from 64 scalar loads to 16 vectorized loads.

However, vectorized loads of sharedA require multiple tmpB values to be available simultaneously:

```sass
LDS R23, [sharedB + 0x800]  // tmpB for it=0
LDS R26, [sharedB + 0x900]  // tmpB for it=1
LDS R25, [sharedB + 0xa00]  // tmpB for it=2
LDS R24, [sharedB + 0xb00]  // tmpB for it=3
// Use R23, R26, R25, R24 with vectorized sharedA loads
```

#### 4. Instruction Prefetching and Pipelining

The compiler interleaves loads and computations to hide latency:

```sass
LDS R23, [...]          // Load tmpB for it=0
LDS.128 R4, [...]       // Load sharedA (parallel)
LDS R26, [...]          // Prefetch tmpB for it=1 (hides latency)
FFMA R4, R4, R23, ...   // Compute using tmpB it=0
FFMA R5, R5, R26, ...   // tmpB it=1 already loaded
```

This software pipelining overlaps memory access with computation.

## Performance Impact Analysis

### Register Pressure Comparison

tmpB approach:
- localC[8]: 8 persistent registers
- tmpB: 4-8 on-demand registers
- sharedA temps: 4-8 transient registers
- Total: ~20-25 registers

localB[8] approach:
- localC[8]: 8 persistent registers
- localB[8]: 8 persistent registers
- sharedA temps: 4-8 transient registers
- Total: ~25-30 registers

### Occupancy Analysis

For SM80 with 65536 registers and 512 threads/block:
- 20 regs/thread: 6 blocks/SM, 96 warps/SM (100% occupancy)
- 25 regs/thread: 5 blocks/SM, 80 warps/SM (100% occupancy)
- 30 regs/thread: 4 blocks/SM, 64 warps/SM (100% occupancy)

Both approaches achieve 100% occupancy, so occupancy is not the primary performance differentiator.

### True Sources of 10% Performance Improvement

1. **Instruction Scheduling Flexibility** (primary factor)
   - tmpB: Short lifetime allows aggressive instruction reordering
   - localB: 8 values must persist, constraining scheduler

2. **Memory Latency Hiding** (primary factor)
   - tmpB: Load-use distance is short, prefetching is flexible
   - localB: Separate load phase cannot overlap with computation

3. **Pipeline Efficiency**
   - tmpB: Loads and computations can be interleaved
   - localB: Must complete all loads before computation begins

4. **Register Allocator Freedom** (secondary factor)
   - tmpB: More registers available for prefetching and intermediate values
   - localB: 16 persistent registers reduce optimization headroom

The 10% improvement comes from microarchitectural optimizations (instruction scheduling, latency hiding, pipelining) rather than occupancy differences.

## Key Insights

1. Loop unrolling does not require all accessed data to reside in registers simultaneously. The compiler manages register lifetimes dynamically.

2. The compiler identifies that memory accesses across all loop iterations are sequential, enabling vectorization despite apparent stride-BK access pattern.

3. Vectorized loads require multiple tmpB values to be available, but these are loaded on-demand rather than pre-loaded like localB.

4. Register pressure matters not just for occupancy, but for giving the compiler freedom to apply advanced microarchitectural optimizations.

5. tmpB approach allows better instruction scheduling and latency hiding compared to localB, even when both achieve the same occupancy.
