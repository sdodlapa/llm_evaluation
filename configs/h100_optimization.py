"""
H100 GPU Optimization Configurations for Maximum Utilization
Target: 80%+ GPU memory usage and computational efficiency
"""

from dataclasses import dataclass
from typing import Dict, Any, List

# Handle imports for both direct execution and module imports
try:
    from .model_configs import ModelConfig
except ImportError:
    from model_configs import ModelConfig

@dataclass
class H100OptimizedConfig:
    """Configuration optimized specifically for H100 80GB GPU maximum utilization"""
    
    # Memory utilization targets
    memory_target_percent: float = 85.0  # Target 85% of 80GB = 68GB
    
    # Computational optimization
    max_batch_size: int = 256  # Much larger batches for throughput
    max_sequences: int = 512   # Aggressive sequence batching
    max_tokens_per_batch: int = 32768  # Large token batches
    
    # Context optimization  
    max_context_length: int = 65536  # Use more of the available context
    
    # Performance features
    enable_cuda_graphs: bool = True
    enable_chunked_prefill: bool = True
    use_fp8_kv_cache: bool = True  # H100 native FP8 support
    enable_flash_attention: bool = True
    
    # Quantization for memory efficiency
    preferred_quantization: str = "awq_marlin"  # Best performance/memory ratio
    
    def create_high_utilization_config(self, base_config: ModelConfig) -> ModelConfig:
        """Transform base config for maximum H100 utilization"""
        # Create copy
        import copy
        optimized = copy.deepcopy(base_config)
        
        # Memory optimization - target 85% usage
        optimized.gpu_memory_utilization = 0.95  # Very aggressive
        
        # Batch size optimization for throughput
        optimized.max_num_seqs = self.max_sequences
        optimized.evaluation_batch_size = min(64, self.max_batch_size // 4)  # Scale with available memory
        
        # Context optimization
        optimized.max_model_len = min(self.max_context_length, base_config.context_window)
        
        # Use best quantization for memory savings
        if optimized.quantization_method == "none" and optimized.size_gb > 10:
            optimized.quantization_method = self.preferred_quantization
            # Switch to AWQ model variant if available
            if "qwen" in optimized.huggingface_id.lower() and "AWQ" not in optimized.huggingface_id:
                optimized.huggingface_id = optimized.huggingface_id.replace("Instruct", "Instruct-AWQ")
        
        # Performance optimizations
        optimized.enforce_eager = False  # Enable CUDA graphs
        optimized.enable_prefix_caching = True
        
        # Advanced vLLM optimizations for H100
        optimized._vllm_overrides.update({
            # Batch processing
            "max_num_batched_tokens": self.max_tokens_per_batch,
            "max_num_seqs": self.max_sequences,
            
            # Memory optimizations
            "kv_cache_dtype": "fp8" if self.use_fp8_kv_cache else "auto",
            "quantization_param_path": None,
            
            # Performance features
            "disable_log_stats": True,  # Reduce overhead
            "use_v2_block_manager": True,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            
            # H100-specific optimizations
            "enable_flash_attention": self.enable_flash_attention,
            "block_size": 32,  # Larger blocks for H100
            
            # Parallel processing
            "worker_use_ray": False,  # Direct GPU access
            "engine_use_ray": False,
            
            # Memory management
            "swap_space": 8,  # 8GB swap space for overflow
            "gpu_memory_utilization": 0.95,  # Match our aggressive target
        })
        
        return optimized

# Pre-configured high-utilization variants
H100_OPTIMIZED_CONFIGS = {}

def create_h100_optimized_qwen3_14b() -> ModelConfig:
    """Create maximum utilization Qwen-3 14B config for H100"""
    try:
        from .model_configs import MODEL_CONFIGS
    except ImportError:
        from model_configs import MODEL_CONFIGS
    
    optimizer = H100OptimizedConfig()
    base_config = MODEL_CONFIGS["qwen3_14b"]
    
    # Create optimized version
    optimized = optimizer.create_high_utilization_config(base_config)
    optimized.model_name = "Qwen-3 14B Instruct (H100 Optimized)"
    
    # Further H100-specific tweaks
    optimized.gpu_memory_utilization = 0.90  # Very aggressive for 14B quantized
    optimized.max_num_seqs = 256  # High throughput
    optimized.max_model_len = 49152  # Use more context (48K)
    optimized.evaluation_batch_size = 32  # Large batch evaluation
    
    # Ensure we use the AWQ model for memory efficiency
    optimized.huggingface_id = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    optimized.quantization_method = "awq_marlin"
    
    return optimized

def create_h100_optimized_qwen3_8b() -> ModelConfig:
    """Create maximum utilization Qwen-3 8B config for H100"""
    try:
        from .model_configs import MODEL_CONFIGS
    except ImportError:
        from model_configs import MODEL_CONFIGS
    
    optimizer = H100OptimizedConfig()
    base_config = MODEL_CONFIGS["qwen3_8b"]
    
    # Create optimized version  
    optimized = optimizer.create_high_utilization_config(base_config)
    optimized.model_name = "Qwen-3 8B Instruct (H100 Optimized)"
    
    # 8B model can be more aggressive since it's smaller
    optimized.gpu_memory_utilization = 0.92  # Even more aggressive
    optimized.max_num_seqs = 512  # Very high throughput
    optimized.max_model_len = 65536  # Full 64K context
    optimized.evaluation_batch_size = 64  # Large batches
    
    # Use quantization for even more memory savings
    optimized.quantization_method = "awq"  # Enable quantization
    optimized.huggingface_id = "Qwen/Qwen2.5-7B-Instruct"  # Keep original for now
    
    return optimized

def create_multiple_model_config() -> Dict[str, Any]:
    """Configuration to run multiple models simultaneously on H100"""
    return {
        "strategy": "multi_model_pipeline",
        "models": [
            {
                "name": "qwen3_8b_fast",
                "memory_allocation": 0.35,  # 35% of GPU (28GB)
                "max_sequences": 128,
                "use_case": "quick_inference"
            },
            {
                "name": "qwen3_14b_awq",
                "memory_allocation": 0.55,  # 55% of GPU (44GB) 
                "max_sequences": 64,
                "use_case": "high_quality_inference"
            }
        ],
        "shared_memory_pool": 0.10,  # 10% shared (8GB)
        "memory_utilization": 0.95
    }

# Register optimized configs
H100_OPTIMIZED_CONFIGS = {
    "qwen3_14b_h100_max": create_h100_optimized_qwen3_14b(),
    "qwen3_8b_h100_max": create_h100_optimized_qwen3_8b(),
}

def estimate_h100_utilization(config: ModelConfig) -> Dict[str, float]:
    """Estimate actual H100 utilization with optimized config"""
    
    # Memory utilization
    memory_est = estimate_memory_usage_h100(config)
    memory_util = memory_est["total_gb"] / 80.0
    
    # Computational utilization estimate (rough)
    # Based on batch size, sequence length, and model size
    compute_factor = (
        (config.max_num_seqs / 512.0) *  # Sequence batching factor
        (config.max_model_len / 65536.0) *  # Context utilization
        (config.size_gb / 15.0) *  # Model size factor
        (config.evaluation_batch_size / 64.0)  # Evaluation batching
    )
    
    compute_util = min(compute_factor, 0.95)  # Cap at 95%
    
    # Memory bandwidth utilization (H100 has 3.35 TB/s)
    # Higher with larger batches and longer sequences
    bandwidth_factor = (config.max_num_seqs * config.max_model_len) / (512 * 32768)
    bandwidth_util = min(bandwidth_factor * 0.7, 0.85)
    
    return {
        "memory_utilization": memory_util,
        "compute_utilization": compute_util, 
        "memory_bandwidth_utilization": bandwidth_util,
        "overall_efficiency": (memory_util + compute_util + bandwidth_util) / 3,
        "throughput_multiplier": compute_factor
    }

def estimate_memory_usage_h100(config: ModelConfig) -> Dict[str, float]:
    """More accurate memory estimation for H100 optimized configs"""
    
    # Base model size with quantization
    if config.quantization_method in ["awq", "awq_marlin"]:
        base_model_gb = config.size_gb * 0.25  # 4-bit quantization
    elif config.quantization_method == "gptq":
        base_model_gb = config.size_gb * 0.25
    else:
        base_model_gb = config.size_gb * 0.5  # FP16
    
    # KV cache with larger batches and context
    # More accurate estimation: batch_size * seq_len * hidden_dim * layers * 2 (K+V) * dtype_size
    hidden_dim = config.size_gb * 1000 / 32  # Rough estimation
    layers = int(config.size_gb * 2.5)  # Rough layer count
    dtype_bytes = 1 if config._vllm_overrides.get("kv_cache_dtype") == "fp8" else 2
    
    kv_cache_gb = (
        config.max_num_seqs * 
        config.max_model_len * 
        hidden_dim * 
        layers * 
        2 *  # K and V
        dtype_bytes
    ) / (1024**3)
    
    # Activation memory for larger batches
    activation_gb = (config.max_num_seqs * config.max_model_len * hidden_dim * 4) / (1024**3)
    
    # Framework overhead (larger for optimized configs)
    framework_overhead = 3.0
    
    # CUDA graphs memory
    cuda_graph_gb = 2.0 if not config.enforce_eager else 0.0
    
    total_gb = base_model_gb + kv_cache_gb + activation_gb + framework_overhead + cuda_graph_gb
    
    return {
        "base_model_gb": base_model_gb,
        "kv_cache_gb": kv_cache_gb,
        "activation_gb": activation_gb,
        "framework_overhead_gb": framework_overhead,
        "cuda_graph_gb": cuda_graph_gb,
        "total_gb": total_gb,
        "h100_percent": (total_gb / 80.0) * 100
    }

def get_optimization_recommendations(current_config: ModelConfig) -> List[str]:
    """Get specific recommendations to improve H100 utilization"""
    recommendations = []
    
    current_util = estimate_h100_utilization(current_config)
    
    if current_util["memory_utilization"] < 0.60:
        recommendations.append("🚀 Increase gpu_memory_utilization to 0.90+ for better memory usage")
        recommendations.append("📈 Increase max_num_seqs to 256+ for higher throughput")
        recommendations.append("🔧 Increase max_model_len to 49152+ to use more context")
    
    if current_util["compute_utilization"] < 0.70:
        recommendations.append("⚡ Increase evaluation_batch_size to 32+ for better compute usage")
        recommendations.append("🎯 Enable CUDA graphs (enforce_eager=False) for efficiency")
        recommendations.append("🔥 Use larger token batches (max_num_batched_tokens=32768+)")
    
    if current_config.quantization_method == "none" and current_config.size_gb > 8:
        recommendations.append("💾 Enable AWQ quantization to free memory for larger batches")
        recommendations.append("🏃 Use awq_marlin quantization for best speed+memory ratio")
    
    if not current_config.enable_prefix_caching:
        recommendations.append("⚡ Enable prefix_caching for repeated prompt efficiency")
    
    return recommendations

# Test and demo functions
if __name__ == "__main__":
    print("=== H100 Optimization Analysis ===")
    print()
    
    # Test current vs optimized configs
    try:
        from .model_configs import MODEL_CONFIGS
    except ImportError:
        from model_configs import MODEL_CONFIGS
    
    configs_to_test = [
        ("Current Qwen-3 14B", MODEL_CONFIGS["qwen3_14b"]),
        ("H100 Optimized Qwen-3 14B", H100_OPTIMIZED_CONFIGS["qwen3_14b_h100_max"]),
        ("Current Qwen-3 8B", MODEL_CONFIGS["qwen3_8b"]),
        ("H100 Optimized Qwen-3 8B", H100_OPTIMIZED_CONFIGS["qwen3_8b_h100_max"])
    ]
    
    print("=== Utilization Comparison ===")
    for name, config in configs_to_test:
        util = estimate_h100_utilization(config)
        memory = estimate_memory_usage_h100(config)
        
        print(f"\n{name}:")
        print(f"  Memory Usage: {memory['total_gb']:.1f}GB ({memory['h100_percent']:.1f}% of H100)")
        print(f"  Memory Utilization: {util['memory_utilization']:.1%}")
        print(f"  Compute Utilization: {util['compute_utilization']:.1%}")
        print(f"  Bandwidth Utilization: {util['memory_bandwidth_utilization']:.1%}")
        print(f"  Overall Efficiency: {util['overall_efficiency']:.1%}")
        print(f"  Throughput Multiplier: {util['throughput_multiplier']:.2f}x")
        print()
        
        print(f"  Configuration:")
        print(f"    GPU Memory Util: {config.gpu_memory_utilization:.0%}")
        print(f"    Max Sequences: {config.max_num_seqs}")
        print(f"    Max Context: {config.max_model_len}")
        print(f"    Batch Size: {config.evaluation_batch_size}")
        print(f"    Quantization: {config.quantization_method}")
    
    print("\n=== Optimization Recommendations ===")
    current_config = MODEL_CONFIGS["qwen3_14b"]
    recommendations = get_optimization_recommendations(current_config)
    
    print(f"For current Qwen-3 14B config:")
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\n=== Multi-Model Pipeline ===")
    multi_config = create_multiple_model_config()
    print(f"Strategy: {multi_config['strategy']}")
    print(f"Total Memory Utilization: {multi_config['memory_utilization']:.0%}")
    
    for model in multi_config['models']:
        allocated_gb = model['memory_allocation'] * 80
        print(f"  {model['name']}: {allocated_gb:.1f}GB ({model['memory_allocation']:.0%})")