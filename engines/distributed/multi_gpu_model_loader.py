"""
Multi-GPU Model Loader for Distributed Evaluation Engine

This module handles loading and distributing large models (30B-180B parameters) 
across multiple GPUs using tensor parallelism and pipeline parallelism strategies.
Designed for optimal performance with large language models.
"""

import os
import time
import logging
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import accelerate
    from accelerate import infer_auto_device_map, dispatch_model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import project modules
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig

logger = logging.getLogger(__name__)

class DistributionStrategy(Enum):
    """Model distribution strategies for multi-GPU setups"""
    TENSOR_PARALLEL = "tensor_parallel"      # Split tensors across GPUs
    PIPELINE_PARALLEL = "pipeline_parallel"  # Split layers across GPUs
    DATA_PARALLEL = "data_parallel"         # Replicate model, split data
    HYBRID = "hybrid"                       # Combination of strategies
    AUTO = "auto"                          # Automatic strategy selection

@dataclass
class GPUAllocation:
    """GPU allocation information for distributed loading"""
    gpu_id: int
    memory_allocated_gb: float
    memory_available_gb: float
    model_layers: Optional[List[str]] = None
    tensor_shards: Optional[List[str]] = None
    is_primary: bool = False

@dataclass
class DistributedModelInfo:
    """Information about a distributed model loading"""
    model_name: str
    total_parameters: int
    distribution_strategy: DistributionStrategy
    gpu_allocations: List[GPUAllocation]
    loading_time_seconds: float
    memory_usage_gb: float
    model_instance: Any = None
    tokenizer: Any = None
    device_map: Optional[Dict[str, Any]] = None
    communication_backend: str = "nccl"
    
class MultiGPUModelLoader:
    """
    Advanced multi-GPU model loader for distributed evaluation engine.
    
    Supports:
    - Tensor parallelism for large model distribution
    - Pipeline parallelism for sequential processing
    - Automatic device mapping and memory optimization
    - Dynamic load balancing across available GPUs
    - Fault tolerance and recovery mechanisms
    """
    
    def __init__(self, 
                 gpu_ids: Optional[List[int]] = None,
                 max_memory_per_gpu: Optional[Dict[int, str]] = None,
                 default_strategy: DistributionStrategy = DistributionStrategy.AUTO):
        """
        Initialize multi-GPU model loader
        
        Args:
            gpu_ids: List of GPU IDs to use (auto-detect if None)
            max_memory_per_gpu: Memory limits per GPU (e.g., {0: "20GB", 1: "20GB"})
            default_strategy: Default distribution strategy
        """
        
        self.gpu_ids = gpu_ids or self._detect_available_gpus()
        self.gpu_count = len(self.gpu_ids)
        self.max_memory_per_gpu = max_memory_per_gpu or {}
        self.default_strategy = default_strategy
        
        # State management
        self._loaded_models: Dict[str, DistributedModelInfo] = {}
        self._device_allocations: Dict[int, float] = {gpu_id: 0.0 for gpu_id in self.gpu_ids}
        self._initialization_lock = threading.Lock()
        
        # Configuration
        self.tensor_parallel_size = min(self.gpu_count, 8)  # Max 8-way tensor parallel
        self.pipeline_parallel_size = max(1, self.gpu_count // self.tensor_parallel_size)
        
        logger.info(f"Initialized multi-GPU loader with {self.gpu_count} GPUs: {self.gpu_ids}")
        logger.info(f"Tensor parallel size: {self.tensor_parallel_size}, Pipeline parallel size: {self.pipeline_parallel_size}")
    
    def _detect_available_gpus(self) -> List[int]:
        """Detect available GPUs"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU-only mode")
            return []
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU-only mode")
            return []
        
        gpu_count = torch.cuda.device_count()
        available_gpus = []
        
        for gpu_id in range(gpu_count):
            try:
                # Test GPU availability
                torch.cuda.set_device(gpu_id)
                memory_info = torch.cuda.get_device_properties(gpu_id)
                
                # Only use GPUs with sufficient memory (>= 8GB)
                if memory_info.total_memory >= 8 * 1024**3:
                    available_gpus.append(gpu_id)
                    logger.info(f"GPU {gpu_id}: {memory_info.name}, {memory_info.total_memory / 1024**3:.1f}GB")
                
            except Exception as e:
                logger.warning(f"GPU {gpu_id} not available: {e}")
        
        return available_gpus
    
    def can_load_model(self, model_config: EnhancedModelConfig) -> bool:
        """
        Check if the model can be loaded with available resources
        
        Args:
            model_config: Enhanced model configuration
            
        Returns:
            True if model can be loaded, False otherwise
        """
        
        if not self.gpu_ids:
            logger.warning("No GPUs available for distributed loading")
            return False
        
        # Check model size requirements
        model_size_gb = getattr(model_config, 'size_gb', 0)
        
        if model_size_gb < 30:  # Small models should use lightweight engine
            logger.debug(f"Model {model_config.model_name} too small for distributed engine ({model_size_gb}GB)")
            return False
        
        # Calculate total available memory
        total_available_memory = 0
        for gpu_id in self.gpu_ids:
            if gpu_id in self.max_memory_per_gpu:
                # Parse memory string (e.g., "20GB" -> 20.0)
                memory_str = self.max_memory_per_gpu[gpu_id]
                memory_gb = float(memory_str.replace('GB', '').replace('gb', ''))
                total_available_memory += memory_gb
            else:
                # Use default GPU memory detection
                try:
                    torch.cuda.set_device(gpu_id)
                    props = torch.cuda.get_device_properties(gpu_id)
                    available_memory = (props.total_memory / 1024**3) * 0.9  # 90% utilization
                    total_available_memory += available_memory
                except Exception:
                    total_available_memory += 20.0  # Conservative fallback
        
        # Model needs approximately 2x its size for loading + inference
        required_memory = model_size_gb * 2.0
        
        can_load = total_available_memory >= required_memory
        
        logger.info(f"Model {model_config.model_name}: Size {model_size_gb}GB, "
                   f"Required {required_memory}GB, Available {total_available_memory:.1f}GB, "
                   f"Can load: {can_load}")
        
        return can_load
    
    def determine_optimal_strategy(self, model_config: EnhancedModelConfig) -> DistributionStrategy:
        """
        Determine optimal distribution strategy for the model
        
        Args:
            model_config: Enhanced model configuration
            
        Returns:
            Optimal distribution strategy
        """
        
        model_size_gb = getattr(model_config, 'size_gb', 0)
        gpu_count = len(self.gpu_ids)
        
        # Strategy decision logic
        if model_size_gb >= 100 and gpu_count >= 8:
            # Very large models with many GPUs - use hybrid approach
            return DistributionStrategy.HYBRID
        elif model_size_gb >= 60 and gpu_count >= 4:
            # Large models with adequate GPUs - use tensor parallelism
            return DistributionStrategy.TENSOR_PARALLEL
        elif model_size_gb >= 30 and gpu_count >= 2:
            # Medium-large models - use pipeline parallelism
            return DistributionStrategy.PIPELINE_PARALLEL
        else:
            # Fallback to data parallelism
            return DistributionStrategy.DATA_PARALLEL
    
    def load_model_distributed(self, model_config: EnhancedModelConfig,
                             strategy: Optional[DistributionStrategy] = None) -> DistributedModelInfo:
        """
        Load model using distributed multi-GPU strategy
        
        Args:
            model_config: Enhanced model configuration
            strategy: Distribution strategy (auto-detect if None)
            
        Returns:
            Distributed model information
            
        Raises:
            RuntimeError: If model loading fails
        """
        
        with self._initialization_lock:
            model_name = model_config.model_name
            
            if model_name in self._loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return self._loaded_models[model_name]
            
            if not self.can_load_model(model_config):
                raise RuntimeError(f"Cannot load model {model_name} - insufficient resources")
            
            # Determine strategy
            if strategy is None or strategy == DistributionStrategy.AUTO:
                strategy = self.determine_optimal_strategy(model_config)
            
            logger.info(f"Loading model {model_name} with strategy: {strategy.value}")
            
            start_time = time.time()
            
            try:
                # Load model based on strategy
                if strategy == DistributionStrategy.TENSOR_PARALLEL:
                    model_info = self._load_tensor_parallel(model_config)
                elif strategy == DistributionStrategy.PIPELINE_PARALLEL:
                    model_info = self._load_pipeline_parallel(model_config)
                elif strategy == DistributionStrategy.HYBRID:
                    model_info = self._load_hybrid_parallel(model_config)
                else:
                    model_info = self._load_data_parallel(model_config)
                
                # Finalize model info
                model_info.loading_time_seconds = time.time() - start_time
                model_info.distribution_strategy = strategy
                
                # Store loaded model
                self._loaded_models[model_name] = model_info
                
                logger.info(f"Successfully loaded {model_name} in {model_info.loading_time_seconds:.2f}s "
                           f"using {strategy.value} strategy")
                
                return model_info
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Cleanup on failure
                self._cleanup_failed_loading(model_name)
                raise RuntimeError(f"Model loading failed: {e}")
    
    def _load_tensor_parallel(self, model_config: EnhancedModelConfig) -> DistributedModelInfo:
        """Load model using tensor parallelism"""
        model_name = model_config.model_name
        
        logger.info(f"Loading {model_name} with tensor parallelism across {self.tensor_parallel_size} GPUs")
        
        # Create device map for tensor parallelism
        device_map = self._create_tensor_parallel_device_map(model_config)
        
        # Load tokenizer
        tokenizer = self._load_tokenizer(model_config)
        
        # Load model with device map
        if TRANSFORMERS_AVAILABLE:
            model = AutoModelForCausalLM.from_pretrained(
                getattr(model_config, 'huggingface_id', model_name),
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=getattr(model_config, 'trust_remote_code', True),
                max_memory=self.max_memory_per_gpu if self.max_memory_per_gpu else None
            )
        else:
            # Mock model for testing without transformers
            model = self._create_mock_distributed_model(model_config)
        
        # Create GPU allocations
        gpu_allocations = []
        for i, gpu_id in enumerate(self.gpu_ids[:self.tensor_parallel_size]):
            allocation = GPUAllocation(
                gpu_id=gpu_id,
                memory_allocated_gb=getattr(model_config, 'size_gb', 30) / self.tensor_parallel_size,
                memory_available_gb=self._get_gpu_available_memory(gpu_id),
                tensor_shards=[f"layer_{j}" for j in range(i, 100, self.tensor_parallel_size)],
                is_primary=(i == 0)
            )
            gpu_allocations.append(allocation)
        
        return DistributedModelInfo(
            model_name=model_name,
            total_parameters=self._estimate_parameters(model_config),
            distribution_strategy=DistributionStrategy.TENSOR_PARALLEL,
            gpu_allocations=gpu_allocations,
            loading_time_seconds=0.0,  # Will be set by caller
            memory_usage_gb=getattr(model_config, 'size_gb', 30),
            model_instance=model,
            tokenizer=tokenizer,
            device_map=device_map,
            communication_backend="nccl"
        )
    
    def _load_pipeline_parallel(self, model_config: EnhancedModelConfig) -> DistributedModelInfo:
        """Load model using pipeline parallelism"""
        model_name = model_config.model_name
        
        logger.info(f"Loading {model_name} with pipeline parallelism across {self.pipeline_parallel_size} GPUs")
        
        # Create device map for pipeline parallelism
        device_map = self._create_pipeline_parallel_device_map(model_config)
        
        # Load tokenizer
        tokenizer = self._load_tokenizer(model_config)
        
        # Load model with device map
        if TRANSFORMERS_AVAILABLE:
            model = AutoModelForCausalLM.from_pretrained(
                getattr(model_config, 'huggingface_id', model_name),
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=getattr(model_config, 'trust_remote_code', True)
            )
        else:
            model = self._create_mock_distributed_model(model_config)
        
        # Create GPU allocations with layer distribution
        gpu_allocations = []
        layers_per_gpu = 50 // self.pipeline_parallel_size  # Assume ~50 layers total
        
        for i, gpu_id in enumerate(self.gpu_ids[:self.pipeline_parallel_size]):
            start_layer = i * layers_per_gpu
            end_layer = min((i + 1) * layers_per_gpu, 50)
            
            allocation = GPUAllocation(
                gpu_id=gpu_id,
                memory_allocated_gb=getattr(model_config, 'size_gb', 30) / self.pipeline_parallel_size,
                memory_available_gb=self._get_gpu_available_memory(gpu_id),
                model_layers=[f"layer_{j}" for j in range(start_layer, end_layer)],
                is_primary=(i == 0)
            )
            gpu_allocations.append(allocation)
        
        return DistributedModelInfo(
            model_name=model_name,
            total_parameters=self._estimate_parameters(model_config),
            distribution_strategy=DistributionStrategy.PIPELINE_PARALLEL,
            gpu_allocations=gpu_allocations,
            loading_time_seconds=0.0,
            memory_usage_gb=getattr(model_config, 'size_gb', 30),
            model_instance=model,
            tokenizer=tokenizer,
            device_map=device_map,
            communication_backend="nccl"
        )
    
    def _load_hybrid_parallel(self, model_config: EnhancedModelConfig) -> DistributedModelInfo:
        """Load model using hybrid parallelism (tensor + pipeline)"""
        model_name = model_config.model_name
        
        logger.info(f"Loading {model_name} with hybrid parallelism")
        
        # For hybrid, combine tensor and pipeline parallelism
        # This is a simplified implementation - production would use frameworks like DeepSpeed
        
        device_map = self._create_hybrid_device_map(model_config)
        tokenizer = self._load_tokenizer(model_config)
        
        if TRANSFORMERS_AVAILABLE:
            model = AutoModelForCausalLM.from_pretrained(
                getattr(model_config, 'huggingface_id', model_name),
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=getattr(model_config, 'trust_remote_code', True)
            )
        else:
            model = self._create_mock_distributed_model(model_config)
        
        # Create complex GPU allocations for hybrid approach
        gpu_allocations = []
        for i, gpu_id in enumerate(self.gpu_ids):
            allocation = GPUAllocation(
                gpu_id=gpu_id,
                memory_allocated_gb=getattr(model_config, 'size_gb', 30) / len(self.gpu_ids),
                memory_available_gb=self._get_gpu_available_memory(gpu_id),
                model_layers=[f"stage_{i//2}_layer_{j}" for j in range(i%2, 25, 2)],
                tensor_shards=[f"tensor_shard_{i%4}"],
                is_primary=(i == 0)
            )
            gpu_allocations.append(allocation)
        
        return DistributedModelInfo(
            model_name=model_name,
            total_parameters=self._estimate_parameters(model_config),
            distribution_strategy=DistributionStrategy.HYBRID,
            gpu_allocations=gpu_allocations,
            loading_time_seconds=0.0,
            memory_usage_gb=getattr(model_config, 'size_gb', 30),
            model_instance=model,
            tokenizer=tokenizer,
            device_map=device_map,
            communication_backend="nccl"
        )
    
    def _load_data_parallel(self, model_config: EnhancedModelConfig) -> DistributedModelInfo:
        """Load model using data parallelism (model replication)"""
        model_name = model_config.model_name
        
        logger.info(f"Loading {model_name} with data parallelism")
        
        # For data parallelism, replicate model on each GPU
        tokenizer = self._load_tokenizer(model_config)
        
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            # Load model on first GPU then replicate
            model = AutoModelForCausalLM.from_pretrained(
                getattr(model_config, 'huggingface_id', model_name),
                torch_dtype=torch.float16,
                trust_remote_code=getattr(model_config, 'trust_remote_code', True)
            )
            
            # Move to first GPU
            if self.gpu_ids:
                model = model.to(f'cuda:{self.gpu_ids[0]}')
            
            # Wrap with DistributedDataParallel if multiple GPUs
            if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                model = DDP(model, device_ids=[self.gpu_ids[0]])
        else:
            model = self._create_mock_distributed_model(model_config)
        
        # Create GPU allocations (model replicated on each GPU)
        model_size = getattr(model_config, 'size_gb', 30)
        gpu_allocations = []
        
        for gpu_id in self.gpu_ids:
            allocation = GPUAllocation(
                gpu_id=gpu_id,
                memory_allocated_gb=model_size,  # Full model on each GPU
                memory_available_gb=self._get_gpu_available_memory(gpu_id),
                model_layers=["full_model"],
                is_primary=(gpu_id == self.gpu_ids[0])
            )
            gpu_allocations.append(allocation)
        
        return DistributedModelInfo(
            model_name=model_name,
            total_parameters=self._estimate_parameters(model_config),
            distribution_strategy=DistributionStrategy.DATA_PARALLEL,
            gpu_allocations=gpu_allocations,
            loading_time_seconds=0.0,
            memory_usage_gb=model_size * len(self.gpu_ids),  # Replicated across GPUs
            model_instance=model,
            tokenizer=tokenizer,
            device_map={"": f"cuda:{self.gpu_ids[0]}"},
            communication_backend="nccl"
        )
    
    def _create_tensor_parallel_device_map(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Create device map for tensor parallelism"""
        device_map = {}
        
        # Distribute layers across GPUs for tensor parallelism
        gpu_devices = [f"cuda:{gpu_id}" for gpu_id in self.gpu_ids[:self.tensor_parallel_size]]
        
        # Simple round-robin assignment of layers
        for i in range(50):  # Assume up to 50 layers
            device_map[f"model.layers.{i}"] = gpu_devices[i % len(gpu_devices)]
        
        # Assign special layers
        device_map["model.embed_tokens"] = gpu_devices[0]
        device_map["model.norm"] = gpu_devices[-1]
        device_map["lm_head"] = gpu_devices[-1]
        
        return device_map
    
    def _create_pipeline_parallel_device_map(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Create device map for pipeline parallelism"""
        device_map = {}
        
        # Assign consecutive layers to each GPU
        layers_per_gpu = 50 // self.pipeline_parallel_size
        
        for gpu_idx, gpu_id in enumerate(self.gpu_ids[:self.pipeline_parallel_size]):
            device = f"cuda:{gpu_id}"
            start_layer = gpu_idx * layers_per_gpu
            end_layer = min((gpu_idx + 1) * layers_per_gpu, 50)
            
            for layer_idx in range(start_layer, end_layer):
                device_map[f"model.layers.{layer_idx}"] = device
        
        # First GPU gets embeddings, last GPU gets final layers
        device_map["model.embed_tokens"] = f"cuda:{self.gpu_ids[0]}"
        device_map["model.norm"] = f"cuda:{self.gpu_ids[self.pipeline_parallel_size-1]}"
        device_map["lm_head"] = f"cuda:{self.gpu_ids[self.pipeline_parallel_size-1]}"
        
        return device_map
    
    def _create_hybrid_device_map(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Create device map for hybrid parallelism"""
        # Simplified hybrid mapping - in production, use DeepSpeed or similar
        device_map = {}
        
        # Combine pipeline and tensor parallelism strategies
        total_layers = 50
        stages = 2  # Two pipeline stages
        layers_per_stage = total_layers // stages
        
        for stage in range(stages):
            stage_gpus = self.gpu_ids[stage*4:(stage+1)*4]  # 4 GPUs per stage
            
            for layer_offset in range(layers_per_stage):
                layer_idx = stage * layers_per_stage + layer_offset
                gpu_idx = layer_offset % len(stage_gpus)
                device_map[f"model.layers.{layer_idx}"] = f"cuda:{stage_gpus[gpu_idx]}"
        
        # Assign special layers
        device_map["model.embed_tokens"] = f"cuda:{self.gpu_ids[0]}"
        device_map["model.norm"] = f"cuda:{self.gpu_ids[-1]}"
        device_map["lm_head"] = f"cuda:{self.gpu_ids[-1]}"
        
        return device_map
    
    def _load_tokenizer(self, model_config: EnhancedModelConfig) -> Any:
        """Load tokenizer for the model"""
        try:
            if TRANSFORMERS_AVAILABLE:
                tokenizer = AutoTokenizer.from_pretrained(
                    getattr(model_config, 'huggingface_id', model_config.model_name),
                    trust_remote_code=getattr(model_config, 'trust_remote_code', True)
                )
                
                # Set pad token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                return tokenizer
            else:
                # Mock tokenizer for testing
                return MockTokenizer()
                
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using mock tokenizer")
            return MockTokenizer()
    
    def _create_mock_distributed_model(self, model_config: EnhancedModelConfig) -> Any:
        """Create mock model for testing without actual model loading"""
        class MockDistributedModel:
            def __init__(self, config):
                self.config = config
                self.device = f"cuda:{self.gpu_ids[0]}" if self.gpu_ids else "cpu"
            
            def generate(self, *args, **kwargs):
                # Mock generation
                return torch.tensor([[1, 2, 3, 4, 5]]) if TORCH_AVAILABLE else [1, 2, 3, 4, 5]
            
            def to(self, device):
                self.device = device
                return self
        
        return MockDistributedModel(model_config)
    
    def _get_gpu_available_memory(self, gpu_id: int) -> float:
        """Get available memory for a GPU in GB"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory = props.total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
                return total_memory - allocated_memory
            else:
                return 20.0  # Mock value for testing
        except Exception:
            return 20.0  # Conservative fallback
    
    def _estimate_parameters(self, model_config: EnhancedModelConfig) -> int:
        """Estimate model parameters from size"""
        size_gb = getattr(model_config, 'size_gb', 30)
        # Rough estimate: 1B parameters ≈ 2GB in fp16
        return int(size_gb * 0.5 * 1e9)
    
    def _cleanup_failed_loading(self, model_name: str) -> None:
        """Cleanup resources after failed model loading"""
        try:
            if model_name in self._loaded_models:
                del self._loaded_models[model_name]
            
            # Clear GPU memory
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for gpu_id in self.gpu_ids:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
            
            logger.info(f"Cleaned up after failed loading of {model_name}")
            
        except Exception as e:
            logger.error(f"Cleanup failed for {model_name}: {e}")
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a distributed model and free resources
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successfully unloaded, False otherwise
        """
        
        with self._initialization_lock:
            if model_name not in self._loaded_models:
                logger.warning(f"Model {model_name} not loaded")
                return False
            
            try:
                model_info = self._loaded_models[model_name]
                
                # Clean up model instance
                if hasattr(model_info.model_instance, 'cpu'):
                    model_info.model_instance.cpu()
                
                del model_info.model_instance
                del model_info.tokenizer
                
                # Update device allocations
                for allocation in model_info.gpu_allocations:
                    self._device_allocations[allocation.gpu_id] -= allocation.memory_allocated_gb
                
                # Remove from loaded models
                del self._loaded_models[model_name]
                
                # Clear GPU caches
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    for gpu_id in self.gpu_ids:
                        torch.cuda.set_device(gpu_id)
                        torch.cuda.empty_cache()
                
                logger.info(f"Successfully unloaded model {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model {model_name}: {e}")
                return False
    
    def get_model_info(self, model_name: str) -> Optional[DistributedModelInfo]:
        """Get information about a loaded model"""
        return self._loaded_models.get(model_name)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self._loaded_models.keys())
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status and allocations"""
        status = {
            'gpu_count': self.gpu_count,
            'gpu_ids': self.gpu_ids,
            'loaded_models': len(self._loaded_models),
            'gpu_allocations': {}
        }
        
        for gpu_id in self.gpu_ids:
            try:
                status['gpu_allocations'][gpu_id] = {
                    'allocated_gb': self._device_allocations[gpu_id],
                    'available_gb': self._get_gpu_available_memory(gpu_id),
                    'utilization': self._get_gpu_utilization(gpu_id)
                }
            except Exception as e:
                status['gpu_allocations'][gpu_id] = {'error': str(e)}
        
        return status
    
    def _get_gpu_utilization(self, gpu_id: int) -> float:
        """Get GPU utilization percentage"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # This is a simplified implementation
                # In production, use nvidia-ml-py or similar for accurate utilization
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                return (allocated / total) * 100.0
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def cleanup(self) -> None:
        """Cleanup all loaded models and resources"""
        logger.info("Cleaning up multi-GPU model loader")
        
        # Unload all models
        model_names = list(self._loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        
        # Clear all state
        self._loaded_models.clear()
        self._device_allocations = {gpu_id: 0.0 for gpu_id in self.gpu_ids}
        
        logger.info("Multi-GPU model loader cleanup completed")

# Mock tokenizer for testing
class MockTokenizer:
    """Mock tokenizer for testing environments"""
    
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab_size = 32000
    
    def encode(self, text: str, **kwargs) -> List[int]:
        # Simple mock encoding
        return [hash(word) % 1000 for word in text.split()]
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        # Simple mock decoding
        return " ".join([f"token_{token_id}" for token_id in token_ids])
    
    def __call__(self, text: Union[str, List[str]], **kwargs):
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        else:
            return {"input_ids": [self.encode(t) for t in text]}