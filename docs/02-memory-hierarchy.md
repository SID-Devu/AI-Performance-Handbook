# Memory Hierarchy Deep Dive

Understanding GPU memory hierarchy is critical for AI performance optimization.

## Memory Types

### Registers
- **Location**: On-chip, per-thread
- **Latency**: 0 cycles
- **Size**: ~256 32-bit registers per thread (architecture dependent)
- **Usage**: Automatic for local variables

### Shared Memory / LDS
- **Location**: On-chip, per-block (compute unit)
- **Latency**: ~20 cycles
- **Size**: 64KB-128KB per CU
- **Usage**: Explicit `__shared__` declaration

### L1 Cache
- **Location**: On-chip, per-CU
- **Latency**: ~80 cycles
- **Size**: 16-128KB
- **Usage**: Automatic caching of global memory

### L2 Cache
- **Location**: On-chip, shared across all CUs
- **Latency**: ~200 cycles
- **Size**: 4-50MB
- **Usage**: Automatic, coherency point

### Global Memory (HBM/GDDR)
- **Location**: Off-chip
- **Latency**: ~400 cycles
- **Size**: 8-192GB
- **Bandwidth**: 500GB/s - 5TB/s

## Memory Access Patterns

### Coalesced Access
```cpp
// GOOD: Threads access consecutive addresses
data[threadIdx.x]

// BAD: Strided access
data[threadIdx.x * stride]
```

### Bank Conflicts
Shared memory has 32 banks. Conflicts occur when multiple threads access the same bank:

```cpp
// NO CONFLICT: Different banks
shared[threadIdx.x]

// 2-WAY CONFLICT: Bank collision
shared[threadIdx.x * 2]  // Every other bank

// SOLUTION: Pad to avoid conflicts
__shared__ float data[32 + 1];  // 33 instead of 32
```

## Optimization Strategies

1. **Maximize L1/L2 hit rate**: Keep working set in cache
2. **Use shared memory**: For data reused within a block
3. **Coalesce memory access**: Consecutive threads → consecutive addresses
4. **Avoid bank conflicts**: Pad shared memory arrays
5. **Overlap compute and memory**: Use prefetching
