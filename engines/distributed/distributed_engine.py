"""
Distributed Evaluation Engine Core

This module provides the main interface for distributed evaluation engine,
integrating the multi-GPU model loader and distributed orchestrator for
efficient large model evaluation with advanced memory management and
multi-GPU communication.
"""

import time
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import uuid

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import project modules
from core_shared.interfaces.evaluation_interfaces import (
    EvaluationRequest, EvaluationResult, EvaluationEngine, EngineType
)
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
from .multi_gpu_model_loader import MultiGPUModelLoader, DistributedModelInfo, DistributionStrategy
from .distributed_orchestrator import (
    DistributedEvaluationOrchestrator, WorkloadPriority, GPUClusterState, ClusterMetrics
)

logger = logging.getLogger(__name__)

class DistributedEngineState(Enum):
    """States of the distributed engine"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    SCALING = "scaling"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class DistributedEngineConfig:
    """Configuration for distributed evaluation engine"""
    max_concurrent_evaluations: int = 4
    enable_dynamic_scaling: bool = True
    enable_fault_tolerance: bool = True
    memory_optimization_level: str = "balanced"  # "aggressive", "balanced", "conservative"
    communication_backend: str = "nccl"  # "nccl", "gloo", "mpi"
    scheduling_strategy: str = "priority_first"  # "priority_first", "round_robin", "load_balanced"
    performance_monitoring: bool = True
    automatic_model_offloading: bool = True
    cross_gpu_memory_sharing: bool = True
    pipeline_optimization: bool = True

@dataclass
class DistributedEvaluationCapabilities:
    """Capabilities and limits of the distributed engine"""
    max_model_size_gb: float
    max_concurrent_models: int
    supported_model_types: List[str]
    tensor_parallel_sizes: List[int]
    pipeline_parallel_sizes: List[int]
    total_gpu_memory_gb: float
    max_batch_size: int
    supports_dynamic_batching: bool
    supports_model_parallelism: bool
    supports_gradient_checkpointing: bool

class DistributedEvaluationEngine(EvaluationEngine):
    """
    Advanced distributed evaluation engine for large language models.
    
    Features:
    - Multi-GPU model loading with tensor/pipeline/hybrid parallelism
    - Intelligent workload distribution and resource allocation
    - Dynamic scaling and fault tolerance
    - Performance monitoring and optimization
    - Memory management and cross-GPU communication
    - Integration with lightweight engine for hybrid architecture
    """
    
    def __init__(self, config: Optional[DistributedEngineConfig] = None):
        """
        Initialize distributed evaluation engine
        
        Args:
            config: Engine configuration parameters
        """
        
        super().__init__()
        
        self.config = config or DistributedEngineConfig()
        self.engine_type = EngineType.DISTRIBUTED
        
        # Core components
        self.model_loader: Optional[MultiGPUModelLoader] = None
        self.orchestrator: Optional[DistributedEvaluationOrchestrator] = None
        
        # State management
        self._state = DistributedEngineState.INITIALIZING
        self._initialization_lock = threading.Lock()
        self._engine_metrics: Dict[str, Any] = {}
        self._loaded_models: Dict[str, DistributedModelInfo] = {}
        
        # Performance tracking
        self._evaluation_count = 0
        self._total_evaluation_time = 0.0
        self._last_performance_update = time.time()
        
        # Initialize components
        self._initialize_engine()
        
        logger.info("Distributed evaluation engine initialized")
        
    def _initialize_engine(self) -> None:
        """Initialize engine components"""
        try:
            with self._initialization_lock:
                logger.info("Initializing distributed evaluation engine...")
                
                # Initialize model loader
                self.model_loader = MultiGPUModelLoader(
                    max_models=self.config.max_concurrent_models if hasattr(self.config, 'max_concurrent_models') else 3,
                    memory_optimization=self.config.memory_optimization_level,
                    enable_pipeline_parallelism=self.config.pipeline_optimization,
                    communication_backend=self.config.communication_backend
                )
                
                # Initialize orchestrator
                self.orchestrator = DistributedEvaluationOrchestrator(
                    model_loader=self.model_loader,
                    max_concurrent_tasks=self.config.max_concurrent_evaluations,
                    enable_fault_tolerance=self.config.enable_fault_tolerance,
                    scheduling_strategy=self.config.scheduling_strategy
                )
                
                # Start orchestrator
                self.orchestrator.start()
                
                # Check capabilities
                self._capabilities = self._determine_capabilities()
                
                self._state = DistributedEngineState.READY
                
                logger.info("Distributed engine initialization completed")
                logger.info(f"Capabilities: {self._capabilities.max_model_size_gb:.1f}GB max model, "
                          f"{self._capabilities.total_gpu_memory_gb:.1f}GB total memory")
                
        except Exception as e:
            self._state = DistributedEngineState.ERROR
            logger.error(f"Failed to initialize distributed engine: {e}")
            raise
    
    def can_handle_request(self, request: EvaluationRequest) -> bool:
        """
        Check if the distributed engine can handle the evaluation request
        
        Args:
            request: Evaluation request to check
            
        Returns:
            True if the engine can handle the request
        """
        
        if self._state != DistributedEngineState.READY:
            logger.warning(f"Engine not ready (state: {self._state.value})")
            return False
        
        if not hasattr(request, 'model_config') or request.model_config is None:
            logger.warning("Request missing model_config")
            return False
        
        model_config = request.model_config
        
        # Check if model is large enough to require distributed processing
        estimated_size = self._estimate_model_size(model_config)
        if estimated_size < 15.0:  # Less than 15GB - use lightweight engine
            logger.debug(f"Model {model_config.model_name} ({estimated_size:.1f}GB) better suited for lightweight engine")
            return False
        
        # Check if we have sufficient resources
        if not self.model_loader.can_load_model(model_config):
            logger.warning(f"Insufficient resources for model {model_config.model_name}")
            return False
        
        # Check capabilities
        if estimated_size > self._capabilities.max_model_size_gb:
            logger.warning(f"Model size {estimated_size:.1f}GB exceeds maximum {self._capabilities.max_model_size_gb:.1f}GB")
            return False
        
        return True
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Perform distributed evaluation
        
        Args:
            request: Evaluation request
            
        Returns:
            Evaluation result
        """
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting distributed evaluation for model {request.model_config.model_name}")
            
            # Validate request
            if not self.can_handle_request(request):
                raise ValueError("Request cannot be handled by distributed engine")
            
            # Update state
            self._state = DistributedEngineState.RUNNING
            
            # Determine priority
            priority = self._determine_evaluation_priority(request)
            
            # Submit to orchestrator
            task_id = self.orchestrator.submit_evaluation_request(
                request=request,
                priority=priority,
                callback=None
            )
            
            logger.info(f"Submitted evaluation task {task_id} with priority {priority.name}")
            
            # Wait for completion with progress monitoring
            result = await self._wait_for_completion(task_id)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, result)
            
            # Update state
            self._state = DistributedEngineState.READY
            
            logger.info(f"Distributed evaluation completed in {result.execution_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            self._state = DistributedEngineState.ERROR
            logger.error(f"Distributed evaluation failed: {e}")
            
            # Create error result
            error_result = EvaluationResult(
                request_id=request.request_id,
                model_name=request.model_config.model_name,
                dataset_name=",".join(request.datasets),
                metrics={},
                engine_used=EngineType.DISTRIBUTED,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=time.time(),
                execution_time_seconds=time.time() - start_time
            )
            
            return error_result
    
    def get_capabilities(self) -> DistributedEvaluationCapabilities:
        """Get engine capabilities"""
        return self._capabilities
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        cluster_status = self.orchestrator.get_cluster_status() if self.orchestrator else {}
        
        return {
            "engine_type": self.engine_type.value,
            "state": self._state.value,
            "capabilities": {
                "max_model_size_gb": self._capabilities.max_model_size_gb,
                "total_gpu_memory_gb": self._capabilities.total_gpu_memory_gb,
                "max_concurrent_models": self._capabilities.max_concurrent_models,
                "supported_parallelism": {
                    "tensor_parallel": self._capabilities.tensor_parallel_sizes,
                    "pipeline_parallel": self._capabilities.pipeline_parallel_sizes
                }
            },
            "performance": {
                "evaluations_completed": self._evaluation_count,
                "average_evaluation_time": (self._total_evaluation_time / max(self._evaluation_count, 1)),
                "current_throughput": self._calculate_current_throughput()
            },
            "cluster_status": cluster_status,
            "loaded_models": {name: {
                "size_gb": info.memory_usage_gb,
                "strategy": info.distribution_strategy.value,
                "gpu_count": len(info.gpu_allocations)
            } for name, info in self._loaded_models.items()},
            "configuration": {
                "max_concurrent_evaluations": self.config.max_concurrent_evaluations,
                "memory_optimization": self.config.memory_optimization_level,
                "scheduling_strategy": self.config.scheduling_strategy,
                "fault_tolerance": self.config.enable_fault_tolerance
            }
        }
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model types"""
        return self._capabilities.supported_model_types
    
    def preload_model(self, model_config: EnhancedModelConfig) -> bool:
        """
        Preload a model for faster evaluation
        
        Args:
            model_config: Model configuration
            
        Returns:
            True if preloading succeeded
        """
        
        try:
            logger.info(f"Preloading model {model_config.model_name}")
            
            # Check if already loaded
            if model_config.model_name in self._loaded_models:
                logger.info(f"Model {model_config.model_name} already loaded")
                return True
            
            # Load model
            model_info = self.model_loader.load_model_distributed(model_config)
            self._loaded_models[model_config.model_name] = model_info
            
            logger.info(f"Successfully preloaded model {model_config.model_name}")
            logger.info(f"Distribution: {model_info.distribution_strategy.value}, "
                       f"GPUs: {len(model_info.gpu_allocations)}, "
                       f"Memory: {model_info.memory_usage_gb:.1f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to preload model {model_config.model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free resources
        
        Args:
            model_name: Name of model to unload
            
        Returns:
            True if unloading succeeded
        """
        
        try:
            if model_name not in self._loaded_models:
                logger.warning(f"Model {model_name} not loaded")
                return False
            
            # Unload from model loader
            success = self.model_loader.unload_model(model_name)
            
            if success:
                del self._loaded_models[model_name]
                logger.info(f"Successfully unloaded model {model_name}")
            else:
                logger.error(f"Failed to unload model {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage across GPUs
        
        Returns:
            Optimization results
        """
        
        try:
            logger.info("Starting memory optimization")
            
            optimization_results = {
                "before_optimization": {},
                "after_optimization": {},
                "actions_taken": [],
                "memory_freed_gb": 0.0
            }
            
            # Get current memory usage
            if self.model_loader:
                gpu_status = self.model_loader.get_gpu_status()
                optimization_results["before_optimization"] = gpu_status
                
                # Perform optimization
                freed_memory = 0.0
                
                # Optimize loaded models
                for model_name in list(self._loaded_models.keys()):
                    if self.config.automatic_model_offloading:
                        # Check if model is being used
                        if not self._is_model_active(model_name):
                            logger.info(f"Unloading inactive model {model_name}")
                            memory_before = self._loaded_models[model_name].memory_usage_gb
                            if self.unload_model(model_name):
                                freed_memory += memory_before
                                optimization_results["actions_taken"].append(f"Unloaded {model_name}")
                
                # Clear caches
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    for gpu_id in self.model_loader.gpu_ids:
                        torch.cuda.set_device(gpu_id)
                        torch.cuda.empty_cache()
                    optimization_results["actions_taken"].append("Cleared GPU caches")
                
                # Get final memory usage
                optimization_results["after_optimization"] = self.model_loader.get_gpu_status()
                optimization_results["memory_freed_gb"] = freed_memory
                
                logger.info(f"Memory optimization completed, freed {freed_memory:.1f}GB")
                
            return optimization_results
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {"error": str(e)}
    
    def shutdown(self) -> None:
        """Shutdown the distributed engine"""
        logger.info("Shutting down distributed evaluation engine")
        
        self._state = DistributedEngineState.SHUTDOWN
        
        # Stop orchestrator
        if self.orchestrator:
            self.orchestrator.stop()
        
        # Unload all models
        for model_name in list(self._loaded_models.keys()):
            self.unload_model(model_name)
        
        # Cleanup model loader
        if self.model_loader:
            self.model_loader.cleanup()
        
        logger.info("Distributed engine shutdown completed")
    
    def _determine_capabilities(self) -> DistributedEvaluationCapabilities:
        """Determine engine capabilities based on available resources"""
        
        if not self.model_loader:
            raise RuntimeError("Model loader not initialized")
        
        gpu_status = self.model_loader.get_gpu_status()
        total_memory = sum(gpu["total_memory_gb"] for gpu in gpu_status.values())
        
        # Conservative estimates for max model size (use 80% of available memory)
        max_model_size = total_memory * 0.8
        
        # Determine supported parallelism
        num_gpus = len(self.model_loader.gpu_ids)
        tensor_parallel_sizes = [1, 2, 4, 8]
        tensor_parallel_sizes = [size for size in tensor_parallel_sizes if size <= num_gpus]
        
        pipeline_parallel_sizes = [1, 2, 4]
        pipeline_parallel_sizes = [size for size in pipeline_parallel_sizes if size <= num_gpus]
        
        return DistributedEvaluationCapabilities(
            max_model_size_gb=max_model_size,
            max_concurrent_models=self.config.max_concurrent_evaluations,
            supported_model_types=[
                "CausalLM", "Seq2SeqLM", "ChatModel", 
                "InstructModel", "CodeModel", "MathModel"
            ],
            tensor_parallel_sizes=tensor_parallel_sizes,
            pipeline_parallel_sizes=pipeline_parallel_sizes,
            total_gpu_memory_gb=total_memory,
            max_batch_size=16,  # Conservative batch size for large models
            supports_dynamic_batching=True,
            supports_model_parallelism=True,
            supports_gradient_checkpointing=True
        )
    
    def _estimate_model_size(self, model_config: EnhancedModelConfig) -> float:
        """Estimate model size in GB"""
        # This is a simplified estimation
        # In production, this would use more sophisticated methods
        
        model_name = model_config.model_name.lower()
        
        # Size estimation based on model name patterns
        if "7b" in model_name or "6.7b" in model_name:
            return 14.0  # ~14GB for 7B models
        elif "13b" in model_name or "12b" in model_name:
            return 26.0  # ~26GB for 13B models
        elif "30b" in model_name or "33b" in model_name:
            return 60.0  # ~60GB for 30B models
        elif "65b" in model_name or "70b" in model_name:
            return 130.0  # ~130GB for 70B models
        elif "175b" in model_name or "180b" in model_name:
            return 350.0  # ~350GB for 175B models
        else:
            # Default estimation based on parameters if available
            if hasattr(model_config, 'parameters'):
                # Rough estimate: 2 bytes per parameter for FP16
                return (model_config.parameters * 2) / (1024**3)
            else:
                # Conservative default
                return 20.0
    
    def _determine_evaluation_priority(self, request: EvaluationRequest) -> WorkloadPriority:
        """Determine priority for an evaluation request"""
        
        # Priority logic could be more sophisticated
        # For now, use simple heuristics
        
        if hasattr(request, 'priority'):
            priority_map = {
                "critical": WorkloadPriority.CRITICAL,
                "high": WorkloadPriority.HIGH,
                "normal": WorkloadPriority.NORMAL,
                "low": WorkloadPriority.LOW
            }
            return priority_map.get(request.priority.lower(), WorkloadPriority.NORMAL)
        
        # Default priority
        return WorkloadPriority.NORMAL
    
    async def _wait_for_completion(self, task_id: str) -> EvaluationResult:
        """Wait for task completion with progress monitoring"""
        
        timeout = 1800  # 30 minute timeout
        start_time = time.time()
        check_interval = 2.0  # Check every 2 seconds
        
        while time.time() - start_time < timeout:
            status = self.orchestrator.get_task_status(task_id)
            
            if status["status"] == "completed":
                return status["result"]
            
            elif status["status"] == "failed":
                raise RuntimeError(f"Task failed: {status['error']}")
            
            elif status["status"] == "running":
                progress = status.get("progress", 0.0)
                logger.debug(f"Task {task_id} progress: {progress:.1f}%")
            
            # Wait before next check
            await asyncio.sleep(check_interval)
        
        # Timeout
        self.orchestrator.cancel_task(task_id)
        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
    
    def _update_performance_metrics(self, start_time: float, result: EvaluationResult) -> None:
        """Update performance tracking metrics"""
        
        execution_time = time.time() - start_time
        
        self._evaluation_count += 1
        self._total_evaluation_time += execution_time
        self._last_performance_update = time.time()
        
        # Log performance summary
        avg_time = self._total_evaluation_time / self._evaluation_count
        
        logger.info(f"Performance update: {self._evaluation_count} evaluations, "
                   f"avg time: {avg_time:.2f}s, throughput: {self._calculate_current_throughput():.2f} eval/hour")
    
    def _calculate_current_throughput(self) -> float:
        """Calculate current evaluation throughput"""
        if self._evaluation_count == 0:
            return 0.0
        
        avg_time = self._total_evaluation_time / self._evaluation_count
        return 3600.0 / avg_time if avg_time > 0 else 0.0  # evaluations per hour
    
    def _is_model_active(self, model_name: str) -> bool:
        """Check if a model is currently being used"""
        
        if not self.orchestrator:
            return False
        
        # Check running tasks
        cluster_status = self.orchestrator.get_cluster_status()
        active_tasks = cluster_status.get("active_tasks", 0)
        
        # Simple heuristic: if no active tasks, models are not active
        return active_tasks > 0


# Factory function for easy instantiation
def create_distributed_engine(
    max_concurrent_evaluations: int = 4,
    memory_optimization: str = "balanced",
    enable_fault_tolerance: bool = True,
    **kwargs
) -> DistributedEvaluationEngine:
    """
    Factory function to create a distributed evaluation engine
    
    Args:
        max_concurrent_evaluations: Maximum concurrent evaluations
        memory_optimization: Memory optimization level
        enable_fault_tolerance: Enable fault tolerance
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured distributed evaluation engine
    """
    
    config = DistributedEngineConfig(
        max_concurrent_evaluations=max_concurrent_evaluations,
        memory_optimization_level=memory_optimization,
        enable_fault_tolerance=enable_fault_tolerance,
        **kwargs
    )
    
    return DistributedEvaluationEngine(config)


# Mock implementation for testing without GPU hardware
class MockDistributedEngine(DistributedEvaluationEngine):
    """Mock distributed engine for testing"""
    
    def __init__(self, config: Optional[DistributedEngineConfig] = None):
        """Initialize mock distributed engine"""
        
        # Skip parent initialization to avoid GPU requirements
        self.config = config or DistributedEngineConfig()
        self.engine_type = EngineType.DISTRIBUTED
        self._state = DistributedEngineState.READY
        self._evaluation_count = 0
        self._total_evaluation_time = 0.0
        self._loaded_models = {}
        
        # Mock capabilities
        self._capabilities = DistributedEvaluationCapabilities(
            max_model_size_gb=200.0,
            max_concurrent_models=4,
            supported_model_types=["CausalLM", "Seq2SeqLM"],
            tensor_parallel_sizes=[1, 2, 4, 8],
            pipeline_parallel_sizes=[1, 2, 4],
            total_gpu_memory_gb=320.0,
            max_batch_size=16,
            supports_dynamic_batching=True,
            supports_model_parallelism=True,
            supports_gradient_checkpointing=True
        )
        
        logger.info("Mock distributed engine initialized")
    
    def can_handle_request(self, request: EvaluationRequest) -> bool:
        """Mock request handling check"""
        return hasattr(request, 'model_config') and request.model_config is not None
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Mock evaluation"""
        start_time = time.time()
        
        # Simulate evaluation time
        await asyncio.sleep(2.0)
        
        execution_time = time.time() - start_time
        self._evaluation_count += 1
        self._total_evaluation_time += execution_time
        
        return EvaluationResult(
            request_id=request.request_id,
            model_name=request.model_config.model_name,
            dataset_name=",".join(request.datasets),
            metrics={
                "accuracy": 0.89,
                "f1_score": 0.87,
                "tokens_per_second": 85.0,
                "gpu_utilization": 0.85
            },
            engine_used=EngineType.DISTRIBUTED,
            success=True,
            started_at=start_time,
            completed_at=time.time(),
            execution_time_seconds=execution_time,
            tokens_processed=500,
            performance_data={
                "distributed_gpus": 4,
                "memory_usage_gb": 45.2,
                "parallelism_strategy": "tensor_parallel"
            }
        )
    
    def shutdown(self) -> None:
        """Mock shutdown"""
        self._state = DistributedEngineState.SHUTDOWN
        logger.info("Mock distributed engine shutdown")
    
    # Required abstract methods from EvaluationEngine
    def can_handle(self, request: EvaluationRequest) -> bool:
        """Check if engine can handle request"""
        return self.can_handle_request(request)
    
    def initialize(self) -> None:
        """Initialize engine"""
        pass  # Already initialized in __init__
    
    def cleanup(self) -> None:
        """Cleanup engine resources"""
        self.shutdown()
    
    def get_resource_requirements(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Get resource requirements for request"""
        return {
            "gpu_memory_gb": 40.0,
            "gpu_count": 4,
            "estimated_duration_seconds": 120.0
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get mock engine status"""
        return {
            "engine_type": self.engine_type.value,
            "state": self._state.value,
            "capabilities": {
                "max_model_size_gb": self._capabilities.max_model_size_gb,
                "total_gpu_memory_gb": self._capabilities.total_gpu_memory_gb,
                "max_concurrent_models": self._capabilities.max_concurrent_models,
                "supported_parallelism": {
                    "tensor_parallel": self._capabilities.tensor_parallel_sizes,
                    "pipeline_parallel": self._capabilities.pipeline_parallel_sizes
                }
            },
            "performance": {
                "evaluations_completed": self._evaluation_count,
                "average_evaluation_time": (self._total_evaluation_time / max(self._evaluation_count, 1)),
                "current_throughput": 3600.0 / max(self._total_evaluation_time / max(self._evaluation_count, 1), 1.0)
            },
            "cluster_status": {
                "active_tasks": 0,
                "queued_tasks": 0,
                "cluster_state": "idle"
            },
            "loaded_models": {},
            "configuration": {
                "max_concurrent_evaluations": self.config.max_concurrent_evaluations,
                "memory_optimization": getattr(self.config, 'memory_optimization_level', 'balanced'),
                "scheduling_strategy": getattr(self.config, 'scheduling_strategy', 'priority_first'),
                "fault_tolerance": getattr(self.config, 'enable_fault_tolerance', True)
            }
        }