"""
Configuration validation and utility functions
Helper functions for model configuration management, validation, and task recommendations
"""

from typing import Dict, Any, Optional, List
try:
    from .model_registry import ModelConfig, MODEL_CONFIGS
    from .preset_configs import ALL_MODEL_CONFIGS
    from .scientific_configs import SCIENTIFIC_MODEL_CONFIGS
except ImportError:
    from configs.model_registry import ModelConfig, MODEL_CONFIGS
    from configs.preset_configs import ALL_MODEL_CONFIGS
    from configs.scientific_configs import SCIENTIFIC_MODEL_CONFIGS

def get_coding_optimized_models() -> Dict[str, ModelConfig]:
    """Get models optimized for coding tasks"""
    coding_models = {}
    
    # Add Qwen Coder models
    for name, config in ALL_MODEL_CONFIGS.items():
        if "coder" in name or "code" in config.model_name.lower():
            coding_models[name] = config
    
    # Add DeepSeek Coder
    if "deepseek_coder_16b" in MODEL_CONFIGS:
        coding_models["deepseek_coder_16b"] = MODEL_CONFIGS["deepseek_coder_16b"]
    
    return coding_models

def get_advanced_code_generation_models() -> Dict[str, ModelConfig]:
    """Get advanced code generation models"""
    return {
        "qwen3_coder_30b": ALL_MODEL_CONFIGS["qwen3_coder_30b"],
        "deepseek_coder_16b": MODEL_CONFIGS["deepseek_coder_16b"],
        "qwen25_7b": ALL_MODEL_CONFIGS["qwen25_7b"],
    }

def get_data_science_models() -> Dict[str, ModelConfig]:
    """Get models optimized for data science tasks"""
    return {
        "qwen25_math_7b": ALL_MODEL_CONFIGS["qwen25_math_7b"],
        "qwen2_vl_7b": ALL_MODEL_CONFIGS["qwen2_vl_7b"],
        "qwen25_7b": ALL_MODEL_CONFIGS["qwen25_7b"],
    }

def get_mathematical_reasoning_models() -> Dict[str, ModelConfig]:
    """Get models optimized for mathematical reasoning"""
    math_models = {
        "qwen25_math_7b": ALL_MODEL_CONFIGS["qwen25_math_7b"],
        "qwen25_7b": ALL_MODEL_CONFIGS["qwen25_7b"],
    }
    
    # Add WizardMath if available
    if "wizardmath_70b" in ALL_MODEL_CONFIGS:
        math_models["wizardmath_70b"] = ALL_MODEL_CONFIGS["wizardmath_70b"]
    
    return math_models

def get_text_generation_models() -> Dict[str, ModelConfig]:
    """Get general text generation models"""
    text_models = {}
    
    # Add main Qwen models
    text_models.update({
        "qwen25_7b": ALL_MODEL_CONFIGS["qwen25_7b"],
        "qwen3_8b": MODEL_CONFIGS["qwen3_8b"],
        "qwen3_14b": MODEL_CONFIGS["qwen3_14b"],
        "qwen25_3b": ALL_MODEL_CONFIGS["qwen25_3b"],
    })
    
    # Add other general models
    if "mistral_nemo_12b" in ALL_MODEL_CONFIGS:
        text_models["mistral_nemo_12b"] = ALL_MODEL_CONFIGS["mistral_nemo_12b"]
    if "granite_3_1_8b" in ALL_MODEL_CONFIGS:
        text_models["granite_3_1_8b"] = ALL_MODEL_CONFIGS["granite_3_1_8b"]
    
    return text_models

def get_all_specialization_categories() -> Dict[str, Dict[str, ModelConfig]]:
    """Get all models organized by specialization category"""
    return {
        "text_generation": get_text_generation_models(),
        "code_generation": get_advanced_code_generation_models(),
        "data_science": get_data_science_models(), 
        "mathematics": get_mathematical_reasoning_models(),
        "bioinformatics": get_genomic_optimized_models(),
        "multimodal": get_multimodal_models(),
        "efficiency": get_efficiency_models(),
        "research": {"olmo2_13b_research": ALL_MODEL_CONFIGS.get("olmo2_13b_research", MODEL_CONFIGS["olmo2_13b"])}
    }

def get_models_by_specialization(category: str) -> Dict[str, ModelConfig]:
    """Get models by specialization category"""
    all_categories = get_all_specialization_categories()
    return all_categories.get(category, {})

def get_genomic_optimized_models() -> Dict[str, ModelConfig]:
    """Get models optimized for genomic data analysis"""
    genomic_models = {}
    
    # Add genomic specific models
    if "qwen25_1_5b_genomic" in ALL_MODEL_CONFIGS:
        genomic_models["qwen25_1_5b_genomic"] = ALL_MODEL_CONFIGS["qwen25_1_5b_genomic"]
    if "qwen25_72b_genomic" in ALL_MODEL_CONFIGS:
        genomic_models["qwen25_72b_genomic"] = ALL_MODEL_CONFIGS["qwen25_72b_genomic"]
    
    # Add math model for genomic analysis
    if "qwen25_math_7b" in ALL_MODEL_CONFIGS:
        genomic_models["qwen25_math_7b"] = ALL_MODEL_CONFIGS["qwen25_math_7b"]
    
    # Add multimodal for genomic visualization
    if "qwen2_vl_7b" in ALL_MODEL_CONFIGS:
        genomic_models["qwen2_vl_7b"] = ALL_MODEL_CONFIGS["qwen2_vl_7b"]
    
    return genomic_models

def get_multimodal_models() -> Dict[str, ModelConfig]:
    """Get models with multimodal capabilities"""
    multimodal_models = {}
    
    if "qwen2_vl_7b" in ALL_MODEL_CONFIGS:
        multimodal_models["qwen2_vl_7b"] = ALL_MODEL_CONFIGS["qwen2_vl_7b"]
    
    return multimodal_models

def get_efficiency_models() -> Dict[str, ModelConfig]:
    """Get small efficient models for resource-constrained scenarios"""
    efficiency_models = {}
    
    # Add small Qwen models
    for model_name in ["qwen25_0_5b", "qwen25_1_5b_genomic", "qwen25_3b"]:
        if model_name in ALL_MODEL_CONFIGS:
            efficiency_models[model_name] = ALL_MODEL_CONFIGS[model_name]
    
    return efficiency_models

def _get_specialization(model_name: str) -> str:
    """Get specialization description for a model"""
    if "coder" in model_name:
        return "Coding & Software Development"
    elif "math" in model_name:
        return "Mathematics & Reasoning"
    elif "vl" in model_name:
        return "Vision-Language (Multimodal)"
    elif "genomic" in model_name:
        return "Genomic Data Analysis"
    elif "0_5b" in model_name:
        return "Efficiency & Edge Deployment"
    elif "3b" in model_name:
        return "Balanced Efficiency & Performance"
    elif "72b" in model_name:
        return "Maximum Performance & Complex Reasoning"
    else:
        return "General Purpose"

def _get_specialization_from_config(config: ModelConfig) -> str:
    """Get specialization from config object"""
    model_id = config.huggingface_id.lower()
    if "coder" in model_id:
        return "Coding Specialist"
    elif "math" in model_id:
        return "Mathematics Specialist"
    elif "vl" in model_id:
        return "Vision-Language"
    elif config.size_gb >= 50:
        return "Large Scale Reasoning"
    elif config.size_gb <= 2:
        return "Efficiency Optimized"
    else:
        return "General Purpose"

def recommend_config_for_task(task_type: str, available_memory_gb: int = 80) -> Optional[ModelConfig]:
    """Recommend optimal configuration based on task and available memory"""
    if task_type == "agent_development":
        # Prioritize balanced configs with good agent capabilities
        candidates = [ALL_MODEL_CONFIGS["qwen25_7b"], MODEL_CONFIGS["qwen3_8b"], MODEL_CONFIGS["qwen3_14b"]]
    elif task_type == "coding":
        # Use coding-specialized models
        candidates = [ALL_MODEL_CONFIGS["qwen3_coder_30b"], ALL_MODEL_CONFIGS["qwen25_7b"], MODEL_CONFIGS["qwen3_8b"]]
    elif task_type == "genomic_analysis":
        # Use genomic-optimized models
        genomic_models = get_genomic_optimized_models()
        candidates = list(genomic_models.values())
    elif task_type == "multimodal":
        # Use vision-language models
        candidates = list(get_multimodal_models().values())
    elif task_type == "mathematics":
        # Use math-specialized models
        candidates = list(get_mathematical_reasoning_models().values())
    elif task_type == "efficiency":
        # Use small efficient models
        candidates = list(get_efficiency_models().values())
    elif task_type == "performance_testing":
        # Use performance presets
        candidates = [
            ALL_MODEL_CONFIGS["qwen3_coder_30b"].create_preset_variant("performance"),
            ALL_MODEL_CONFIGS["qwen25_7b"].create_preset_variant("performance"),
            MODEL_CONFIGS["qwen3_14b"].create_preset_variant("performance")
        ]
    elif task_type == "memory_constrained":
        # Use memory-optimized presets
        candidates = [
            ALL_MODEL_CONFIGS["qwen25_3b"].create_preset_variant("memory_optimized"),
            ALL_MODEL_CONFIGS["qwen25_7b"].create_preset_variant("memory_optimized"),
            MODEL_CONFIGS["qwen3_8b"].create_preset_variant("memory_optimized")
        ]
    else:
        # Default to balanced approach with new recommended models
        candidates = [ALL_MODEL_CONFIGS["qwen25_7b"], MODEL_CONFIGS["qwen3_8b"], ALL_MODEL_CONFIGS["qwen25_3b"]]
    
    # Filter by memory constraints
    for config in candidates:
        memory_est = estimate_memory_usage(config)
        if memory_est["total_estimated_gb"] <= available_memory_gb * 0.9:  # 10% buffer
            return config
    
    return None

def get_high_priority_models():
    """Get models marked as HIGH priority"""
    all_configs = {**MODEL_CONFIGS, **ALL_MODEL_CONFIGS, **SCIENTIFIC_MODEL_CONFIGS}
    return {k: v for k, v in all_configs.items() if v.priority == "HIGH"}

def get_apache_licensed_models():
    """Get models with Apache 2.0 license (safe for commercial use)"""
    all_configs = {**MODEL_CONFIGS, **ALL_MODEL_CONFIGS, **SCIENTIFIC_MODEL_CONFIGS}
    return {k: v for k, v in all_configs.items() if "Apache" in v.license}

def get_agent_optimized_models():
    """Get models suitable for agentic systems"""
    all_configs = {**MODEL_CONFIGS, **ALL_MODEL_CONFIGS, **SCIENTIFIC_MODEL_CONFIGS}
    return {k: v for k, v in all_configs.items() if v.agent_optimized}

def estimate_memory_usage(config: ModelConfig, context_length: int = None) -> Dict[str, float]:
    """Estimate VRAM usage for a given configuration"""
    if context_length is None:
        context_length = config.max_model_len
    
    # Base model size (quantized)
    if config.quantization_method == "awq":
        base_size = config.size_gb * 0.25  # 4-bit quantization
    elif config.quantization_method == "gptq":
        base_size = config.size_gb * 0.25  # 4-bit quantization  
    else:
        base_size = config.size_gb * 0.5   # FP16
    
    # KV cache estimation (rough)
    # For transformer models: context_len * hidden_size * 2 * layers * 2 (K+V) / 1e9
    # Simplified estimation: ~2-4 MB per 1K context tokens for 7-8B models
    kv_cache_gb = (context_length / 1000) * (config.size_gb / 4) * 0.002
    
    # Framework overhead (vLLM, etc.)
    overhead_gb = 2.0
    
    total_estimated = base_size + kv_cache_gb + overhead_gb
    
    return {
        "base_model_gb": base_size,
        "kv_cache_gb": kv_cache_gb, 
        "overhead_gb": overhead_gb,
        "total_estimated_gb": total_estimated,
        "h100_utilization": total_estimated / 80.0  # H100 has 80GB
    }

def validate_model_config(config: ModelConfig) -> List[str]:
    """Validate a model configuration and return any issues"""
    issues = []
    
    # Check required fields
    if not config.model_name:
        issues.append("model_name is required")
    if not config.huggingface_id:
        issues.append("huggingface_id is required")
    if not config.license:
        issues.append("license is required")
    
    # Check numeric constraints
    if config.size_gb <= 0:
        issues.append("size_gb must be positive")
    if config.context_window <= 0:
        issues.append("context_window must be positive")
    if config.max_model_len <= 0:
        issues.append("max_model_len must be positive")
    if not (0.1 <= config.gpu_memory_utilization <= 1.0):
        issues.append("gpu_memory_utilization must be between 0.1 and 1.0")
    
    # Check preset validity
    if config.preset not in ["balanced", "performance", "memory_optimized"]:
        issues.append("preset must be 'balanced', 'performance', or 'memory_optimized'")
    
    # Check priority validity
    if config.priority not in ["HIGH", "MEDIUM", "LOW"]:
        issues.append("priority must be 'HIGH', 'MEDIUM', or 'LOW'")
    
    # Memory usage validation
    memory_est = estimate_memory_usage(config)
    if memory_est["total_estimated_gb"] > 80:
        issues.append(f"Estimated memory usage ({memory_est['total_estimated_gb']:.1f}GB) exceeds H100 capacity (80GB)")
    
    return issues

def get_all_model_configs() -> Dict[str, ModelConfig]:
    """Get all model configurations from all modules"""
    all_configs = {}
    all_configs.update(MODEL_CONFIGS)
    all_configs.update(ALL_MODEL_CONFIGS)
    all_configs.update(SCIENTIFIC_MODEL_CONFIGS)
    
    # Remove duplicates (prioritize scientific configs, then all_model_configs, then base configs)
    return all_configs

def get_model_by_huggingface_id(hf_id: str) -> Optional[ModelConfig]:
    """Find model configuration by HuggingFace ID"""
    all_configs = get_all_model_configs()
    for config in all_configs.values():
        if config.huggingface_id == hf_id:
            return config
    return None

def get_models_by_size_range(min_gb: float, max_gb: float) -> Dict[str, ModelConfig]:
    """Get models within a specific size range"""
    all_configs = get_all_model_configs()
    return {k: v for k, v in all_configs.items() if min_gb <= v.size_gb <= max_gb}