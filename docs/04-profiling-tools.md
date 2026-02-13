# Profiling Tools Overview

Comprehensive guide to AI/ML profiling tools.

## Tool Categories

### GPU Profilers
- **rocprof** (AMD): Hardware counters, tracing
- **Nsight Systems** (NVIDIA): System-wide timeline
- **Nsight Compute** (NVIDIA): Kernel analysis
- **Intel VTune**: CPU + GPU profiling

### Framework Profilers
- **PyTorch Profiler**: Operator-level profiling
- **TensorFlow Profiler**: Graph execution analysis
- **ONNX Runtime Profiling**: EP-level stats

### System Profilers
- **perf** (Linux): CPU profiling
- **top/htop**: Resource monitoring
- **nvidia-smi/rocm-smi**: GPU monitoring

## Quick Reference

### rocprof Basics
```bash
# Kernel statistics
rocprof --stats ./app

# Hardware counters
rocprof -i counters.txt ./app

# Full trace
rocprof --hip-trace --hsa-trace -o trace.csv ./app
```

### Nsight Systems
```bash
# Basic profile
nsys profile ./app

# With CUDA/NVTX
nsys profile --trace=cuda,nvtx ./app
```

### PyTorch Profiler
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
) as prof:
    model(input)
print(prof.key_averages().table())
```

## When to Use What

| Goal | Tool |
|------|------|
| Kernel bottlenecks | rocprof/ncu |
| System overview | nsys/rocprof trace |
| Memory issues | rocprof counters |
| Framework ops | PyTorch/TF profiler |
| CPU/GPU interplay | nsys timeline |
