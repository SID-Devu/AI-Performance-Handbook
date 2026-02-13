# Chapter 1: Fundamentals of AI Acceleration

This chapter introduces the fundamental concepts of AI acceleration on modern hardware.

## 1.1 The AI Compute Landscape

Modern AI workloads require massive computational power. Training large language models can take thousands of GPU-hours, while inference must meet strict latency requirements.

### Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Latency | Time to complete one inference | <100ms for interactive |
| Throughput | Inferences per second | As high as possible |
| Efficiency | Performance per watt | Critical for deployment |

## 1.2 GPU Architecture Basics

### Execution Model

GPUs use SIMD (Single Instruction Multiple Data) execution:
- **Wavefront/Warp**: Group of threads executing same instruction
- **Compute Unit/SM**: Contains multiple SIMD units
- **Device**: Contains many CUs/SMs

### Memory Hierarchy

```
Registers (fastest, smallest)
    ↓
L1 Cache / Shared Memory
    ↓
L2 Cache  
    ↓
Global Memory / HBM (slowest, largest)
```

### Key Concepts

1. **Occupancy**: Ratio of active to maximum waves per CU
2. **Memory Coalescing**: Combining memory accesses
3. **Bank Conflicts**: Contention for shared memory banks

## 1.3 AI Workload Characteristics

### Common Operations

| Operation | Compute Intensity | Memory Pattern |
|-----------|-------------------|----------------|
| GEMM | High | Regular |
| Convolution | High | Strided |
| Attention | Variable | Complex |
| LayerNorm | Low | Sequential |
| Softmax | Low | Reduction |

### Bottleneck Classification

1. **Compute-Bound**: Limited by ALU throughput
   - Dense matrix multiplication
   - High arithmetic intensity

2. **Memory-Bound**: Limited by bandwidth
   - Batch normalization
   - Element-wise operations

3. **Latency-Bound**: Limited by dependencies
   - Serial operations
   - Small batch sizes

## 1.4 Performance Analysis Framework

### Roofline Model

The roofline model relates:
- **Arithmetic Intensity** (FLOPS/byte)
- **Peak Compute** (TFLOPS)
- **Peak Bandwidth** (TB/s)

```
Performance = min(Peak_Compute, AI × Bandwidth)
```

### Key Questions

1. Am I compute or memory bound?
2. What is my achieved vs. theoretical performance?
3. Where are the bottlenecks?

## 1.5 Summary

Understanding these fundamentals is crucial for:
- Choosing the right optimization strategy
- Interpreting profiling results
- Making architecture decisions

## Exercises

1. Calculate the arithmetic intensity of GEMM (M×K × K×N)
2. Determine if a 256×256 convolution is compute or memory bound
3. Estimate the theoretical peak throughput for your GPU
