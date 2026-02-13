# Chapter 2: Profiling and Analysis

This chapter covers techniques for profiling AI workloads and analyzing performance bottlenecks.

## 2.1 Profiling Fundamentals

### What to Measure

1. **Timing**: End-to-end latency, kernel duration
2. **Throughput**: Samples/second, tokens/second
3. **Resource Usage**: GPU utilization, memory bandwidth
4. **Efficiency**: Achieved vs. theoretical performance

### Profiling Tools

| Tool | Platform | Purpose |
|------|----------|---------|
| rocprof | ROCm | Kernel timing, counters |
| nsys | CUDA | System-wide tracing |
| PyTorch Profiler | PyTorch | High-level profiling |
| perfetto | Cross-platform | Trace visualization |

## 2.2 Using rocprof

### Basic Usage

```bash
# Simple kernel timing
rocprof ./my_app

# With performance counters
rocprof -i metrics.txt -o results.csv ./my_app

# With HIP API tracing
rocprof --hip-trace ./my_app
```

### Key Metrics

```
pmc: VALUUtilization    # Vector ALU usage
pmc: VALUBusy           # ALU busy cycles
pmc: MemUnitBusy        # Memory controller busy
pmc: FetchSize          # Bytes read from memory
pmc: WriteSize          # Bytes written to memory
pmc: L2CacheHit         # L2 cache hit rate
```

## 2.3 Analyzing Results

### Identifying Bottlenecks

| Indicator | Bottleneck | Action |
|-----------|------------|--------|
| High VALUUtil, Low MemBusy | Compute | Use faster math |
| Low VALUUtil, High MemBusy | Memory | Optimize accesses |
| Low everything | Latency | Increase parallelism |

### Hotspot Analysis

1. Find kernels with highest total time
2. Check call frequency
3. Calculate percentage of total runtime

## 2.4 Memory Analysis

### Access Patterns

```
Good: Coalesced (consecutive addresses)
[T0][T1][T2][T3] → [A0][A1][A2][A3]

Bad: Strided (gaps between addresses)
[T0][T1][T2][T3] → [A0][A8][A16][A24]
```

### Bandwidth Calculation

```
Achieved_BW = (FetchSize + WriteSize) / Duration
Efficiency = Achieved_BW / Peak_BW
```

## 2.5 Optimization Workflow

1. **Profile** - Get baseline measurements
2. **Identify** - Find the bottleneck
3. **Hypothesize** - Predict improvement
4. **Optimize** - Make targeted changes
5. **Verify** - Confirm improvement
6. **Repeat** - Until target achieved

## 2.6 Case Study: GEMM Optimization

### Initial Profile

```
Kernel: sgemm_naive
Duration: 12.5 ms
VALUUtil: 35%
MemBusy: 85%
Diagnosis: Memory bound
```

### After Tiling

```
Kernel: sgemm_tiled
Duration: 4.2 ms
VALUUtil: 72%
MemBusy: 60%
Improvement: 3x faster
```

## Exercises

1. Profile a matrix multiplication and identify the bottleneck
2. Calculate achieved bandwidth for a vector addition
3. Use rocprof to compare naive vs. optimized implementations
