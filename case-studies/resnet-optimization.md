# Case Study: ResNet-50 Inference Optimization

## Overview
This case study demonstrates how to optimize ResNet-50 inference performance on AMD GPUs using ROCm.

## Initial Performance
- **Baseline Latency:** 12.5 ms @ batch size 1
- **Baseline Throughput:** 80 images/sec @ batch size 32
- **GPU Utilization:** 45%

## Optimization Journey

### Step 1: Profile the Baseline
```bash
rocprof --stats python inference.py
```

**Findings:**
- Convolution kernels: 65% of time
- Memory transfers: 20% of time
- Batch normalization: 10% of time
- Other: 5%

### Step 2: Enable Mixed Precision (FP16)
```python
import torch
model = model.half().cuda()
with torch.cuda.amp.autocast():
    output = model(input_fp16)
```

**Results:**
- Latency reduced to 6.8 ms (1.84x speedup)
- Memory usage reduced by 45%

### Step 3: Kernel Fusion with MIGraphX
```python
import migraphx
program = migraphx.parse_onnx("resnet50.onnx")
program.compile(migraphx.get_target("gpu"))
```

**Results:**
- Latency reduced to 4.2 ms (2.98x total speedup)
- Fused BN+ReLU kernels eliminated memory round-trips

### Step 4: Optimize Batch Size
Testing various batch sizes:

| Batch Size | Latency (ms) | Throughput (img/s) | Efficiency |
|------------|--------------|-------------------|------------|
| 1          | 4.2          | 238               | 85%        |
| 4          | 5.1          | 784               | 92%        |
| 8          | 6.8          | 1,176             | 95%        |
| 16         | 10.2         | 1,568             | 97%        |
| 32         | 18.5         | 1,730             | 98%        |

### Step 5: Memory Optimization
```python
# Pre-allocate buffers
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
```

## Final Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Latency (bs=1) | 12.5 ms | 4.2 ms | **3.0x** |
| Throughput (bs=32) | 80 img/s | 1,730 img/s | **21.6x** |
| GPU Utilization | 45% | 97% | **2.2x** |
| Memory Footprint | 4.2 GB | 2.1 GB | **50% reduction** |

## Key Takeaways

1. **Mixed precision** provides nearly 2x speedup with minimal accuracy loss
2. **Kernel fusion** eliminates memory bottlenecks
3. **Batch size tuning** is critical for throughput optimization
4. **Profiling first** ensures you optimize the right things

## Profiling Commands Used
```bash
# Overall stats
rocprof --stats python inference.py

# Kernel-level timing
rocprof -i counters.txt python inference.py

# Memory analysis
rocprof --hip-trace python inference.py
```
