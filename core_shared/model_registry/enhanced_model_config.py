"""
Enhanced ModelConfig with multi-GPU and engine selection support

Extends the existing ModelConfig with capabilities for hybrid architecture
while maintaining backward compatibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
import copy


class ModelSizeCategory(Enum):
    """Model size categories for engine selection"""
    SMALL = "small"        # ≤7B parameters
    MEDIUM = "medium"      # 7B-30B parameters  
    LARGE = "large"        # 30B-70B parameters
    EXTRA_LARGE = "xl"     # 70B+ parameters


class EnginePreference(Enum):
    """Engine selection preferences"""
    LIGHTWEIGHT = "lightweight"
    DISTRIBUTED = "distributed"
    AUTO = "auto"


@dataclass
class MultiGPUConfig:
    """Multi-GPU configuration parameters"""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    auto_parallel: bool = True
    min_gpu_memory_gb: float = 80.0
    preferred_gpu_topology: str = "any"  # "nvlink", "pcie", "any"
    enable_cuda_graphs: bool = True
    max_parallel_loading_workers: int = 4


@dataclass
class LightweightOptimizations:
    """Optimizations for lightweight engine"""
    fast_loading: bool = True
    memory_mapping: bool = True
    cache_embeddings: bool = False
    use_flash_attention: bool = True
    enable_kv_cache: bool = True
    batch_size_optimization: bool = True


@dataclass
class DistributedOptimizations:
    """Optimizations for distributed engine"""
    persistent_service: bool = True
    async_loading: bool = True
    distributed_cache: bool = True
    load_balancing: str = "round_robin"  # "round_robin", "least_loaded", "weighted"
    auto_scaling: bool = False
    checkpoint_frequency: int = 100  # Save state every N evaluations


@dataclass
class EngineSelectionCriteria:
    """Criteria for automatic engine selection"""
    model_size_threshold_gb: float = 50.0  # Updated: Simplified single threshold for clean engine selection
    memory_utilization_threshold: float = 0.9
    prefer_speed: bool = True
    prefer_memory_efficiency: bool = False
    max_latency_seconds: Optional[float] = None
    min_throughput_tokens_per_sec: Optional[float] = None


@dataclass
class EnhancedModelConfig:
    """Enhanced ModelConfig with multi-GPU and engine selection support"""
    
    # === EXISTING FIELDS (Backward Compatibility) ===
    model_name: str
    huggingface_id: str
    license: str
    size_gb: float
    context_window: int
    preset: str = "balanced"
    
    # Specialization metadata (existing)
    specialization_category: str = "general"
    specialization_subcategory: str = "general_purpose"
    primary_use_cases: List[str] = field(default_factory=lambda: ["general"])
    
    # Basic model settings (existing)
    quantization_method: str = "none"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    priority: str = "HIGH"
    agent_optimized: bool = True
    
    # Advanced vLLM settings (existing)
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    use_v2_block_manager: bool = True
    enforce_eager: bool = False
    
    # Agent-specific optimizations (existing)
    function_calling_format: str = "json"
    max_function_calls_per_turn: int = 5
    agent_temperature: float = 0.1
    
    # Evaluation settings (existing)
    evaluation_batch_size: int = 8
    benchmark_iterations: int = 3
    
    # Legacy multi-GPU fields (for compatibility)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # === NEW HYBRID ARCHITECTURE FIELDS ===
    
    # Engine selection
    preferred_engine: EnginePreference = EnginePreference.AUTO
    engine_selection_criteria: EngineSelectionCriteria = field(default_factory=EngineSelectionCriteria)
    
    # Enhanced multi-GPU configuration
    multi_gpu_config: MultiGPUConfig = field(default_factory=MultiGPUConfig)
    
    # Engine-specific optimizations
    lightweight_optimizations: LightweightOptimizations = field(default_factory=LightweightOptimizations)
    distributed_optimizations: DistributedOptimizations = field(default_factory=DistributedOptimizations)
    
    # Model characteristics for engine selection
    model_size_category: Optional[ModelSizeCategory] = None
    estimated_parameters_b: Optional[float] = None  # Billion parameters
    memory_footprint_multiplier: float = 2.0  # Model size to memory ratio
    
    # Performance targets
    target_tokens_per_second: Optional[float] = None
    target_latency_seconds: Optional[float] = None
    max_memory_gb: Optional[float] = None
    
    # Compatibility and testing
    tested_engines: List[str] = field(default_factory=list)
    compatibility_notes: str = ""
    last_validated: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields and validate configuration"""
        # Sync legacy fields with new multi_gpu_config
        if hasattr(self, 'multi_gpu_config'):
            self.multi_gpu_config.tensor_parallel_size = self.tensor_parallel_size
            self.multi_gpu_config.pipeline_parallel_size = self.pipeline_parallel_size
        
        # Auto-determine model size category
        if self.model_size_category is None:
            self.model_size_category = self._determine_size_category()
        
        # Estimate parameters if not provided
        if self.estimated_parameters_b is None:
            self.estimated_parameters_b = self._estimate_parameters()
        
        # Validate configuration
        self._validate_config()
    
    def _determine_size_category(self) -> ModelSizeCategory:
        """Determine model size category based on size_gb"""
        if self.size_gb <= 15:  # ~7B parameters
            return ModelSizeCategory.SMALL
        elif self.size_gb <= 60:  # ~30B parameters
            return ModelSizeCategory.MEDIUM
        elif self.size_gb <= 140:  # ~70B parameters
            return ModelSizeCategory.LARGE
        else:
            return ModelSizeCategory.EXTRA_LARGE
    
    def _estimate_parameters(self) -> float:
        """Estimate parameter count from model size"""
        # Rough estimation: 1B parameters ≈ 2GB in fp16
        return self.size_gb / 2.0
    
    def _validate_config(self):
        """Validate configuration consistency"""
        # Validate tensor parallel configuration
        if self.tensor_parallel_size > 8:
            print(f"Warning: Large tensor_parallel_size ({self.tensor_parallel_size}) may cause issues")
        
        # Validate memory utilization
        if self.gpu_memory_utilization > 0.95:
            print(f"Warning: High GPU memory utilization ({self.gpu_memory_utilization}) may cause OOM")
    
    def get_optimal_engine(self) -> EnginePreference:
        """Determine optimal engine for this model configuration"""
        if self.preferred_engine != EnginePreference.AUTO:
            return self.preferred_engine
        
        # Automatic engine selection logic
        criteria = self.engine_selection_criteria
        
        # Size-based selection
        if self.size_gb <= criteria.model_size_threshold_gb and self.tensor_parallel_size == 1:
            return EnginePreference.LIGHTWEIGHT
        else:
            return EnginePreference.DISTRIBUTED
    
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Calculate resource requirements for this model"""
        total_memory = self.size_gb * self.memory_footprint_multiplier
        gpu_count = max(self.tensor_parallel_size, self.pipeline_parallel_size)
        
        return {
            "gpu_memory_gb": total_memory / gpu_count,
            "gpu_count": gpu_count,
            "total_memory_gb": total_memory,
            "cpu_cores": 4 * gpu_count,
            "estimated_load_time_seconds": min(self.size_gb * 5, 300),  # Cap at 5 minutes
            "disk_space_gb": self.size_gb * 1.5  # Include cache space
        }
    
    def to_legacy_config(self) -> Dict[str, Any]:
        """Convert to legacy ModelConfig format for backward compatibility"""
        # Return only the original fields
        legacy_fields = [
            'model_name', 'huggingface_id', 'license', 'size_gb', 'context_window',
            'preset', 'specialization_category', 'specialization_subcategory', 
            'primary_use_cases', 'quantization_method', 'max_model_len',
            'gpu_memory_utilization', 'trust_remote_code', 'torch_dtype',
            'priority', 'agent_optimized', 'max_num_seqs', 'enable_prefix_caching',
            'use_v2_block_manager', 'enforce_eager', 'function_calling_format',
            'max_function_calls_per_turn', 'agent_temperature', 'evaluation_batch_size',
            'benchmark_iterations', 'tensor_parallel_size', 'pipeline_parallel_size'
        ]
        
        config_dict = asdict(self)
        return {field: config_dict[field] for field in legacy_fields if field in config_dict}
    
    def get_vllm_config(self) -> Dict[str, Any]:
        """Get vLLM-specific configuration parameters"""
        return {
            "model": self.huggingface_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.torch_dtype,
            "quantization": self.quantization_method if self.quantization_method != "none" else None,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "use_v2_block_manager": self.use_v2_block_manager,
            "enforce_eager": self.enforce_eager,
            "enable_cuda_graphs": self.multi_gpu_config.enable_cuda_graphs
        }
    
    def clone_with_overrides(self, **overrides) -> 'EnhancedModelConfig':
        """Create a copy with specified field overrides"""
        config_dict = asdict(self)
        config_dict.update(overrides)
        return EnhancedModelConfig(**config_dict)
    
    def update_multi_gpu_config(self, tensor_parallel: int = None, pipeline_parallel: int = None):
        """Update multi-GPU configuration and sync legacy fields"""
        if tensor_parallel is not None:
            self.tensor_parallel_size = tensor_parallel
            self.multi_gpu_config.tensor_parallel_size = tensor_parallel
        
        if pipeline_parallel is not None:
            self.pipeline_parallel_size = pipeline_parallel
            self.multi_gpu_config.pipeline_parallel_size = pipeline_parallel
    
    def is_large_model(self) -> bool:
        """Check if this is considered a large model requiring distributed processing"""
        return (self.model_size_category in [ModelSizeCategory.LARGE, ModelSizeCategory.EXTRA_LARGE] or
                self.size_gb > self.engine_selection_criteria.model_size_threshold_gb or
                self.tensor_parallel_size > 1 or
                self.pipeline_parallel_size > 1)
    
    def get_engine_compatibility(self) -> Dict[str, bool]:
        """Get compatibility with different engines"""
        return {
            "lightweight": not self.is_large_model(),
            "distributed": True,  # Distributed engine should handle all models
            "recommended": self.get_optimal_engine().value
        }