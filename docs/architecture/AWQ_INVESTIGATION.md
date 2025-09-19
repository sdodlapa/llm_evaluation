# AWQ Quantization Investigation Report

**Date**: September 17, 2025  
**Framework**: vLLM 0.10.2  
**Model**: Qwen/Qwen2.5-14B-Instruct  
**Hardware**: H100 80GB GPU

## 🔍 Investigation Summary

### Problem Statement
Initial attempt to use AWQ quantization with `Qwen/Qwen2.5-14B-Instruct` failed with the error:
```
Cannot find the config file for awq [type=value_error]
```

### Root Cause Analysis

**✅ Issue Identified**: The base model `Qwen/Qwen2.5-14B-Instruct` does not include AWQ quantization configuration files. AWQ quantization requires models that were specifically quantized during preparation.

**✅ Solution Applied**: Use the official AWQ-quantized model `Qwen/Qwen2.5-14B-Instruct-AWQ` instead of trying to apply AWQ to the base model.

## 📊 Performance Investigation Results

### Configuration Changes
```python
# BEFORE (Failed)
"qwen3_14b": ModelConfig(
    huggingface_id="Qwen/Qwen2.5-14B-Instruct",
    quantization_method="awq",  # ❌ AWQ config not available
)

# AFTER (Working)  
"qwen3_14b": ModelConfig(
    huggingface_id="Qwen/Qwen2.5-14B-Instruct-AWQ",  # ✅ Pre-quantized model
    quantization_method="awq",  # ✅ AWQ works correctly
)
```

### Performance Comparison

| Configuration | Model | Quantization | Performance | Memory Usage |
|---------------|-------|-------------|-------------|--------------|
| **Original** | Qwen2.5-14B-Instruct | None (fp16) | 71.5 tok/s | ~27.6 GiB |
| **AWQ Fixed** | Qwen2.5-14B-Instruct-AWQ | AWQ | 11.5 tok/s | ~9.4 GiB |
| **Performance Drop** | - | - | **-84%** | **-66%** |

### Memory Efficiency vs Speed Trade-off

**✅ Memory Benefits:**
- **66% memory reduction**: 27.6 GiB → 9.4 GiB
- **Allows larger batch sizes**: More concurrent requests possible
- **Better multi-model support**: Can load multiple models simultaneously

**⚠️ Performance Cost:**
- **84% speed reduction**: 71.5 → 11.5 tokens/second
- **Significantly slower inference**: May not be suitable for real-time applications
- **vLLM 0.10.2 optimization**: AWQ implementation not fully optimized yet

## 🔬 Technical Analysis

### vLLM AWQ Implementation Status

**Current State (vLLM 0.10.2):**
```
WARNING: awq quantization is not fully optimized yet. 
The speed can be slower than non-quantized models.
```

**Alternative Options Detected:**
```
INFO: Detected that the model can run with awq_marlin, however you specified 
quantization=awq explicitly, so forcing awq. Use quantization=awq_marlin for 
faster inference
```

### AWQ vs AWQ-Marlin Investigation

**AWQ-Marlin Testing**: Failed due to memory constraints during testing (previous model still loaded). However, logs suggest AWQ-Marlin is the optimized implementation that should provide better performance.

**Recommendations for Future Testing:**
1. Test `quantization="awq_marlin"` instead of `quantization="awq"`
2. Ensure clean GPU state between tests
3. Consider vLLM version upgrade for better AWQ optimization

## 📋 Available Quantized Models

### Official Qwen AWQ Models Found:
- ✅ `Qwen/Qwen2.5-14B-Instruct-AWQ` (Used in fix)
- ✅ `ibrahimkettaneh/Qwen2.5-14B-Instruct-abliterated-AWQ`
- ✅ `tadangkhoa1999/Qwen2.5-14B-Instruct-AWQ-trim-vocab`
- ✅ Several community variants available

### Other Quantization Options:
- 🔍 `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4` (GPTQ quantization available)
- 🔍 Various GGUF quantized versions for different use cases

## 🎯 Conclusions & Recommendations

### Short-term Recommendations

**1. Current Configuration (Implemented):**
- ✅ Use `Qwen/Qwen2.5-14B-Instruct-AWQ` for memory-constrained scenarios
- ✅ Accept 84% performance reduction for 66% memory savings
- ✅ Suitable for batch processing, research, or memory-limited environments

**2. Performance-Critical Scenarios:**
- 📋 Use unquantized `Qwen/Qwen2.5-14B-Instruct` for maximum speed
- 📋 71.5 tokens/second performance maintained
- 📋 Requires 27.6 GiB GPU memory

### Long-term Improvements

**1. vLLM Optimization:**
- 🔄 Monitor vLLM updates for improved AWQ performance
- 🔄 Test AWQ-Marlin quantization when memory allows
- 🔄 Consider vLLM version upgrade (current: 0.10.2)

**2. Alternative Quantization:**
- 🔍 Evaluate GPTQ quantization performance
- 🔍 Test mixed precision approaches
- 🔍 Consider model-specific optimizations

**3. Configuration Options:**
```python
# Option 1: Speed-optimized (current default)
"qwen3_14b": ModelConfig(
    huggingface_id="Qwen/Qwen2.5-14B-Instruct",
    quantization_method="none",
    # Fast but memory-intensive
)

# Option 2: Memory-optimized (AWQ)
"qwen3_14b_awq": ModelConfig(
    huggingface_id="Qwen/Qwen2.5-14B-Instruct-AWQ", 
    quantization_method="awq",
    # Memory-efficient but slower
)

# Option 3: Potential future optimization
"qwen3_14b_marlin": ModelConfig(
    huggingface_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
    quantization_method="awq_marlin",
    # May offer better speed/memory balance
)
```

## 🔧 Implementation Status

**✅ Completed:**
- AWQ quantization error resolved
- Official AWQ model successfully integrated
- Performance benchmarking completed
- Configuration documented in `ARCHITECTURE.md`

**📋 Future Work:**
- AWQ-Marlin performance testing
- GPTQ quantization evaluation  
- vLLM version compatibility testing
- Multi-quantization configuration support

---

**Final Assessment**: AWQ quantization works correctly but comes with significant performance trade-offs in the current vLLM implementation. The fix enables memory-efficient evaluation scenarios while maintaining functional correctness.