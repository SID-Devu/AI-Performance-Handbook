# Compute vs Memory Bound

Understanding whether your kernel is compute-bound or memory-bound is fundamental to optimization.

## The Roofline Model

```
Performance (FLOPS/s)
    │
    │                    Compute Bound Region
    │                           ┌──────────────────────
    │                          /│
    │                         / │
Peak─┼─────────────────────────/─│─────────────────────
    │                       /   │
    │                      /    │                       
    │                     /     │
    │  Memory Bound      /      │
    │    Region         /       │
    │                  /        │
    │                 /         │
    │                /          │
    └───────────────/───────────┴───────────────────────
                  Ridge        Arithmetic
                  Point        Intensity (FLOP/byte)
```

## Arithmetic Intensity

```
AI = Operations / Bytes Accessed

Low AI (<10): Memory bound
High AI (>10): Compute bound
```

### Examples

| Operation | Typical AI | Bound Type |
|-----------|-----------|------------|
| Vector Add | 0.25 | Memory |
| GEMM (large) | 100+ | Compute |
| Conv2D | 10-50 | Varies |
| Batch Norm | 1 | Memory |
| Softmax | 2-5 | Memory |
| Attention | 5-20 | Varies |

## Identifying Your Bottleneck

### Memory Bound Symptoms
- Low compute utilization
- High memory throughput
- Bandwidth near peak

### Compute Bound Symptoms  
- High ALU utilization
- Low memory traffic
- High occupancy helpful

## Optimization Strategies

### For Memory Bound
1. Reduce data movement (fusion)
2. Use lower precision (FP16/INT8)
3. Improve cache utilization
4. Optimize memory access patterns

### For Compute Bound
1. Increase parallelism
2. Use specialized units (tensor cores)
3. Algorithmic improvements
4. Reduce instruction count
