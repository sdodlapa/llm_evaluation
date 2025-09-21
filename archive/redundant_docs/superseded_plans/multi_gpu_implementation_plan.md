# Multi-GPU Implementation Plan for Large Open Source Models (40B-100B Parameters)

## Executive Summary

This document outlines the implementation strategy for adding multi-GPU support to our current evaluation pipeline to handle large language models (40B-100B parameters) that require distributed inference across multiple GPUs.

## Table of Contents
1. [Large Model Inventory](#large-model-inventory)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Implementation Strategy](#implementation-strategy)
4. [Code Changes Required](#code-changes-required)
5. [Infrastructure Requirements](#infrastructure-requirements)
6. [Testing and Validation Plan](#testing-and-validation-plan)
7. [Resource Requirements](#resource-requirements)
8. [Implementation Timeline](#implementation-timeline)

---

## Large Model Inventory

### 40B Parameter Models
1. **Falcon-40B** (tiiuae/falcon-40b)
   - Parameters: 40B
   - Memory requirement: 85-100GB
   - License: Apache 2.0
   - Architecture: Causal decoder-only with FlashAttention and multiquery
   - Trained on: 384 A100 40GB GPUs using 3D parallelism (TP=8, PP=4, DP=12)

2. **ALIA-40B** (BSC-LT/ALIA-40b)
   - Parameters: 40B
   - Downloads: 1.73k
   - Multi-GPU inference required

3. **Evo2-40B** (arcinstitute/evo2_40b)
   - Parameters: 40B
   - Downloads: 599
   - Specialized for biological sequences

4. **GPT-SW3-40B** (AI-Sweden-Models/gpt-sw3-40b)
   - Parameters: 40B
   - Downloads: 1.83k
   - Swedish language focus

### 70B+ Parameter Models
1. **Llama 2 70B** (meta-llama/Llama-2-70b-hf)
   - Parameters: 70B
   - Memory requirement: ~140GB
   - License: Custom Llama 2 license
   - Widely used and benchmarked

2. **Code Llama 70B** (codellama/CodeLlama-70b-hf)
   - Parameters: 70B
   - Specialized for code generation
   - Memory requirement: ~140GB

3. **Falcon-180B** (tiiuae/falcon-180b)
   - Parameters: 180B
   - Memory requirement: ~360GB
   - Largest open-source model available
   - License: Apache 2.0

### Multi-GPU Memory Requirements
- **40B models**: 4-6 H100 GPUs (80GB each)
- **70B models**: 6-8 H100 GPUs
- **180B models**: 12-16 H100 GPUs

---

## Current Architecture Analysis

### Existing Infrastructure Capabilities
Our current pipeline already has multi-GPU support infrastructure in place but unused:

#### 1. Model Configuration System (`configs/model_registry.py`)
```python
def get_vllm_config(self) -> Dict[str, Any]:
    return {
        "model": self.model_path,
        "tensor_parallel_size": 1,  # ← Currently hardcoded to 1
        "pipeline_parallel_size": 1,  # ← Currently hardcoded to 1
        "max_model_len": self.max_length,
        "dtype": self.dtype,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9
    }
```

#### 2. Model Loading (`models/qwen_implementation.py`)
```python
def load_model(self):
    vllm_args = self.model_config.to_vllm_args()
    self.model = LLM(**vllm_args)  # ← Already supports tensor parallelism
```

#### 3. SLURM Integration (`slurm_jobs/*.slurm`)
```bash
#SBATCH --gpus=1  # ← Currently limited to single GPU
```

### Key Findings
- **vLLM backend**: Fully supports tensor and pipeline parallelism
- **Configuration system**: Parameters exist but set to 1
- **SLURM jobs**: Limited to single GPU allocation
- **GPU monitoring**: Needs extension for multi-GPU tracking

---

## Implementation Strategy

### 1. vLLM Tensor Parallelism Approach
vLLM provides two parallelism strategies:

#### Tensor Parallelism (Single-node, Multi-GPU)
- **Use case**: 40B-70B models on single node with multiple H100s
- **Configuration**: `tensor_parallel_size=4` for 4 GPUs
- **Memory distribution**: Model weights split across GPUs
- **Communication**: High-bandwidth GPU interconnect (NVLink)

#### Pipeline Parallelism (Multi-node)
- **Use case**: 180B models requiring multiple nodes
- **Configuration**: `pipeline_parallel_size=2` for 2 nodes
- **Memory distribution**: Model layers distributed across nodes
- **Communication**: Network-based inter-node communication

### 2. HuggingFace Device Map Approach (Alternative)
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-40b",
    device_map="auto",  # Automatic multi-GPU distribution
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
```

---

## Code Changes Required

### 1. Configuration System Updates

#### `configs/model_registry.py`
```python
class ModelConfig:
    def __init__(self, 
                 model_path: str,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 **kwargs):
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        # ... existing code ...
    
    def get_vllm_config(self) -> Dict[str, Any]:
        return {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_length,
            "dtype": self.dtype,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9
        }
    
    def estimate_gpu_requirements(self) -> Dict[str, Any]:
        """Estimate GPU memory and count requirements"""
        model_size_gb = self.get_model_size_gb()
        
        if model_size_gb <= 80:
            return {"gpu_count": 1, "memory_per_gpu": model_size_gb}
        elif model_size_gb <= 160:
            return {"gpu_count": 2, "memory_per_gpu": model_size_gb / 2}
        elif model_size_gb <= 320:
            return {"gpu_count": 4, "memory_per_gpu": model_size_gb / 4}
        else:
            return {"gpu_count": 8, "memory_per_gpu": model_size_gb / 8}
```

#### Large Model Preset Definitions
```python
# New preset configurations for large models
LARGE_MODEL_PRESETS = {
    "falcon_40b": ModelConfig(
        model_path="tiiuae/falcon-40b",
        tensor_parallel_size=4,
        max_length=2048,
        dtype="bfloat16"
    ),
    "llama2_70b": ModelConfig(
        model_path="meta-llama/Llama-2-70b-hf",
        tensor_parallel_size=6,
        max_length=4096,
        dtype="bfloat16"
    ),
    "falcon_180b": ModelConfig(
        model_path="tiiuae/falcon-180b",
        tensor_parallel_size=12,
        pipeline_parallel_size=2,
        max_length=2048,
        dtype="bfloat16"
    )
}
```

### 2. Model Implementation Updates

#### `models/qwen_implementation.py`
```python
class Qwen3Implementation:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.multi_gpu_setup = model_config.tensor_parallel_size > 1
        
    def load_model(self):
        """Enhanced model loading with multi-GPU support"""
        vllm_args = self.model_config.get_vllm_config()
        
        if self.multi_gpu_setup:
            # Ensure proper GPU visibility
            import os
            gpu_count = vllm_args["tensor_parallel_size"]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(gpu_count)))
            
        try:
            self.model = LLM(**vllm_args)
            self._log_gpu_allocation()
        except Exception as e:
            self._handle_multi_gpu_errors(e)
            
    def _log_gpu_allocation(self):
        """Log GPU memory allocation across devices"""
        if self.multi_gpu_setup:
            import torch
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB allocated")
```

### 3. SLURM Job Template Updates

#### `slurm_jobs/large_model_evaluation.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=large_model_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4                    # Request 4 GPUs for 40B models
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G                  # Increased memory for large models
#SBATCH --time=24:00:00
#SBATCH --output=logs/large_model_%j.out
#SBATCH --error=logs/large_model_%j.err

# Environment setup
source ~/.bashrc
conda activate llm_env

# GPU verification
echo "Available GPUs:"
nvidia-smi

# Set CUDA visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run evaluation with multi-GPU model
python category_evaluation.py \
    --model_preset falcon_40b \
    --tensor_parallel_size 4 \
    --categories reasoning,math,coding \
    --output_dir results/large_models/
```

### 4. GPU Monitoring Enhancement

#### `evaluation/performance_monitor.py`
```python
class MultiGPUMonitor(GPUMonitor):
    def __init__(self, gpu_count: int):
        super().__init__()
        self.gpu_count = gpu_count
        
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Enhanced GPU monitoring for multiple devices"""
        metrics = {}
        
        for gpu_id in range(self.gpu_count):
            try:
                gpu_info = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Memory usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_info)
                memory_used = mem_info.used / 1024**3
                memory_total = mem_info.total / 1024**3
                
                # Utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_info)
                
                metrics[f"gpu_{gpu_id}"] = {
                    "memory_used_gb": memory_used,
                    "memory_total_gb": memory_total,
                    "memory_utilization": (memory_used / memory_total) * 100,
                    "gpu_utilization": util_info.gpu,
                    "temperature": pynvml.nvmlDeviceGetTemperature(gpu_info, pynvml.NVML_TEMPERATURE_GPU)
                }
                
            except Exception as e:
                metrics[f"gpu_{gpu_id}"] = {"error": str(e)}
                
        return metrics
```

---

## Infrastructure Requirements

### 1. Hardware Requirements

#### For 40B Models (Falcon-40B, ALIA-40B)
- **GPUs**: 4x H100 80GB or 6x A100 40GB
- **CPU**: 32+ cores
- **RAM**: 256GB+
- **Storage**: 500GB+ NVMe SSD
- **Network**: High-bandwidth interconnect (NVLink/InfiniBand)

#### For 70B Models (Llama 2 70B)
- **GPUs**: 6x H100 80GB or 8x A100 40GB
- **CPU**: 48+ cores
- **RAM**: 384GB+
- **Storage**: 1TB+ NVMe SSD

#### For 180B Models (Falcon-180B)
- **GPUs**: 12-16x H100 80GB
- **CPU**: 64+ cores
- **RAM**: 512GB+
- **Storage**: 2TB+ NVMe SSD
- **Network**: Multi-node setup with InfiniBand

### 2. Software Dependencies

#### Updated Requirements
```txt
# Add to requirements.txt
vllm>=0.4.0                 # Latest version with enhanced parallelism
ray>=2.9.0                  # For distributed inference
torch>=2.1.0                # PyTorch 2.0+ required for Falcon models
transformers>=4.35.0        # Latest transformers with device_map improvements
accelerate>=0.24.0          # For automatic device mapping
bitsandbytes>=0.41.0        # For potential quantization support
```

### 3. Environment Configuration

#### SLURM Partition Setup
```bash
# New GPU partition configuration needed
PartitionName=large_gpu Nodes=gpu-node[01-04] Default=NO MaxTime=48:00:00 State=UP
```

---

## Testing and Validation Plan

### 1. Progressive Testing Strategy

#### Phase 1: Single Large Model (Falcon-40B)
1. **Memory validation**: Verify 4-GPU tensor parallelism loads successfully
2. **Inference testing**: Generate responses with various prompt lengths
3. **Performance benchmarking**: Compare against single-GPU smaller models
4. **Error handling**: Test OOM scenarios and recovery

#### Phase 2: Multiple Model Support
1. **Model switching**: Test loading different 40B+ models
2. **Configuration validation**: Verify auto-GPU allocation works
3. **Resource cleanup**: Ensure proper memory cleanup between models

#### Phase 3: Evaluation Pipeline Integration
1. **Category evaluation**: Run existing evaluation categories on large models
2. **Comparative analysis**: Compare large vs small model performance
3. **Resource utilization**: Monitor GPU memory and compute efficiency

### 2. Validation Commands

#### Memory Test Script
```python
# test_large_model_loading.py
import torch
from models.qwen_implementation import Qwen3Implementation
from configs.model_registry import LARGE_MODEL_PRESETS

def test_model_loading(model_name: str):
    """Test loading a large model with multi-GPU"""
    config = LARGE_MODEL_PRESETS[model_name]
    
    print(f"Testing {model_name}:")
    print(f"  Tensor parallel size: {config.tensor_parallel_size}")
    print(f"  Expected GPU memory: {config.estimate_gpu_requirements()}")
    
    # Load model
    impl = Qwen3Implementation(config)
    impl.load_model()
    
    # Test inference
    response = impl.generate_response("What is the capital of France?")
    print(f"  Sample response: {response[:100]}...")
    
    # Check GPU memory
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i} memory: {allocated:.1f}GB")

if __name__ == "__main__":
    test_model_loading("falcon_40b")
```

### 3. Performance Benchmarks

#### Throughput Testing
```python
# benchmark_large_models.py
import time
from typing import List

def benchmark_inference_speed(model_impl, prompts: List[str], batch_size: int = 1):
    """Benchmark inference speed for large models"""
    start_time = time.time()
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        responses = [model_impl.generate_response(prompt) for prompt in batch]
    
    total_time = time.time() - start_time
    tokens_per_second = len(prompts) * 100 / total_time  # Assuming ~100 tokens per response
    
    return {
        "total_time": total_time,
        "prompts_processed": len(prompts),
        "tokens_per_second": tokens_per_second,
        "time_per_prompt": total_time / len(prompts)
    }
```

---

## Resource Requirements

### 1. Compute Allocation

#### SLURM Resource Requests
```bash
# 40B models
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# 70B models  
#SBATCH --gpus=6
#SBATCH --mem=384G
#SBATCH --time=36:00:00

# 180B models
#SBATCH --gpus=12
#SBATCH --mem=512G
#SBATCH --time=48:00:00
```

### 2. Storage Requirements

#### Model Storage
- **Model weights**: 500GB per 40B model, 1TB per 180B model
- **Cache storage**: 100GB per model for tokenizer and config caches
- **Results storage**: 50GB per evaluation session for large models

#### Estimated Total Storage
- **Models**: 5TB (10 large models)
- **Results**: 1TB (historical evaluations)
- **Logs**: 500GB (detailed multi-GPU logs)
- **Total**: ~7TB additional storage needed

### 3. Network Requirements

#### Data Transfer
- **Model download**: 40B model = ~80GB download
- **Inter-GPU communication**: High-bandwidth required for tensor parallelism
- **Node-to-node**: InfiniBand for pipeline parallelism (180B models)

---

## Implementation Timeline

### Week 1: Infrastructure Setup
- [ ] Update SLURM configuration for multi-GPU jobs
- [ ] Install updated dependencies (vLLM, Ray, etc.)
- [ ] Test basic multi-GPU allocation

### Week 2: Core Implementation
- [ ] Implement ModelConfig updates for tensor parallelism
- [ ] Update Qwen3Implementation for multi-GPU loading
- [ ] Create large model preset configurations

### Week 3: SLURM Integration
- [ ] Create multi-GPU SLURM job templates
- [ ] Update category_evaluation.py for large models
- [ ] Implement enhanced GPU monitoring

### Week 4: Testing and Validation
- [ ] Test Falcon-40B loading and inference
- [ ] Validate memory usage and performance
- [ ] Run sample evaluation categories

### Week 5: Extended Model Support
- [ ] Add support for Llama 2 70B
- [ ] Test pipeline parallelism for 180B models
- [ ] Performance optimization and tuning

### Week 6: Documentation and Deployment
- [ ] Complete testing and validation
- [ ] Create operational documentation
- [ ] Deploy to production evaluation pipeline

---

## Risk Assessment and Mitigation

### 1. Technical Risks

#### Memory Overflow
- **Risk**: Insufficient GPU memory for large models
- **Mitigation**: Dynamic GPU allocation based on model size estimation
- **Fallback**: Automatic quantization (FP16 → INT8) if needed

#### Inter-GPU Communication Bottlenecks
- **Risk**: Slow tensor parallel communication
- **Mitigation**: Verify NVLink/high-bandwidth interconnects
- **Fallback**: Reduce tensor_parallel_size, increase pipeline stages

#### Model Loading Failures
- **Risk**: Large models fail to load due to configuration issues
- **Mitigation**: Progressive validation (test with smaller models first)
- **Fallback**: HuggingFace device_map as alternative to vLLM

### 2. Resource Constraints

#### GPU Availability
- **Risk**: Limited access to multi-GPU nodes
- **Mitigation**: Queue management and job prioritization
- **Fallback**: Time-shared evaluation sessions

#### Storage Limitations
- **Risk**: Insufficient storage for multiple large models
- **Mitigation**: Model caching and cleanup automation
- **Fallback**: On-demand model download/deletion

### 3. Operational Risks

#### Configuration Complexity
- **Risk**: Complex multi-GPU setup prone to errors
- **Mitigation**: Automated configuration validation
- **Fallback**: Manual configuration for critical evaluations

#### Performance Degradation
- **Risk**: Multi-GPU overhead reduces overall throughput
- **Mitigation**: Benchmark and optimize tensor parallel sizes
- **Fallback**: Selective use of large models for key evaluations

---

## Success Metrics

### 1. Technical Success Criteria
- [ ] Successful loading of 40B+ parameter models
- [ ] Inference speed comparable to single-GPU smaller models
- [ ] Memory utilization <90% across all allocated GPUs
- [ ] Error-free evaluation runs for 24+ hours

### 2. Performance Benchmarks
- **Throughput**: Minimum 10 tokens/second for 40B models
- **Memory efficiency**: >80% GPU memory utilization
- **Reliability**: <1% job failure rate
- **Scalability**: Support for 8+ concurrent large model evaluations

### 3. Evaluation Quality
- [ ] Consistent results across single vs multi-GPU configurations
- [ ] Successful completion of all evaluation categories
- [ ] Comparative analysis showing large model advantages
- [ ] Integration with existing result analysis tools

---

## Conclusion

This implementation plan provides a comprehensive roadmap for adding multi-GPU support to handle large open-source models (40B-100B parameters). The existing infrastructure already contains the foundational elements needed, requiring primarily configuration updates and SLURM job modifications rather than architectural changes.

Key advantages of this approach:
- **Leverages existing vLLM infrastructure**
- **Minimal code changes required**
- **Scalable to even larger models (180B+)**
- **Maintains compatibility with current evaluation pipeline**

The plan prioritizes safety and validation, with progressive testing and fallback strategies to ensure stable operation during the transition to multi-GPU model support.

---

## Next Steps

1. **Immediate**: Review and approve this implementation plan
2. **Short-term**: Begin Week 1 infrastructure setup tasks
3. **Medium-term**: Execute implementation timeline over 6 weeks
4. **Long-term**: Evaluate results and plan for next-generation models (200B+)

---

*Document Version: 1.0*  
*Last Updated: January 19, 2025*  
*Author: AI Assistant*  
*Review Status: Pending*