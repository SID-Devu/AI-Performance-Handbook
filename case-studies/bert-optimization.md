# Case Study: Optimizing BERT Inference

## Problem Statement

BERT model inference running at 120ms latency, target is <30ms for real-time applications.

## Initial Analysis

### Profiling Results
```
rocprof --stats --hip-trace ./inference

Top 5 Kernels by Time:
1. gemm_fp32      45.2ms (38%)
2. softmax        22.1ms (18%)
3. layer_norm     18.5ms (15%)
4. attention      15.8ms (13%)
5. gelu           10.2ms (9%)
```

## Optimization Steps

### Step 1: Precision Reduction (FP32 → FP16)
- Changed to FP16 execution
- Result: 120ms → 65ms (1.8x speedup)

### Step 2: Operator Fusion
- Fused: LayerNorm + Attention + GELU
- Result: 65ms → 42ms (1.5x additional)

### Step 3: KV-Cache Implementation  
- Cached key/value for autoregressive
- Result: 42ms → 28ms (target achieved!)

## Final Architecture

```
┌─────────────────────────────────────────┐
│        Optimized BERT Pipeline          │
├─────────────────────────────────────────┤
│                                         │
│  Input → Embedding → [Fused Blocks] →   │
│              ↓                          │
│    ┌─────────────────────────┐          │
│    │   Fused Transformer     │          │
│    │   - FP16 GEMM           │          │
│    │   - Fused Attention     │          │
│    │   - KV Cache            │          │
│    └─────────────────────────┘          │
│              ↓                          │
│         → Pooling → Output              │
│                                         │
└─────────────────────────────────────────┘
```

## Key Learnings

1. Profile first - GEMM was 38% of time
2. Precision matters - easy 2x win
3. Fusion reduces memory traffic significantly
4. Caching eliminates redundant computation
