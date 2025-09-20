"""
Distributed Evaluation Engine Module

This module provides a comprehensive distributed evaluation engine for large
language models, featuring multi-GPU coordination, advanced resource management,
and seamless integration with lightweight evaluation engines.

Key Components:
- MultiGPUModelLoader: Advanced model loading with multiple distribution strategies
- DistributedEvaluationOrchestrator: Workload scheduling and resource management
- DistributedEvaluationEngine: Main engine interface for distributed evaluation
- MultiGPUPerformanceMonitor: Real-time performance monitoring and optimization

Supported Features:
- Models up to 180B parameters across 8+ GPUs
- Tensor, Pipeline, and Hybrid parallelism strategies
- Fault tolerance and automatic recovery
- Real-time performance monitoring and alerting
- Hybrid architecture with lightweight engine integration
"""

from .distributed_engine import (
    DistributedEvaluationEngine, 
    DistributedEngineConfig,
    MockDistributedEngine,
    create_distributed_engine
)

from .multi_gpu_model_loader import (
    MultiGPUModelLoader,
    DistributionStrategy,
    DistributedModelInfo,
    GPUAllocation
)

from .distributed_orchestrator import (
    DistributedEvaluationOrchestrator,
    WorkloadPriority,
    GPUClusterState,
    WorkloadTask,
    GPUAllocationPlan,
    ClusterMetrics
)

from .performance_monitor import (
    MultiGPUPerformanceMonitor,
    MetricType,
    AlertSeverity,
    PerformanceMetric,
    GPUMetrics,
    CommunicationMetrics,
    SystemMetrics,
    PerformanceAlert,
    PerformanceSummary
)

__all__ = [
    # Main engine classes
    'DistributedEvaluationEngine',
    'DistributedEngineConfig', 
    'MockDistributedEngine',
    'create_distributed_engine',
    
    # Model loading
    'MultiGPUModelLoader',
    'DistributionStrategy',
    'DistributedModelInfo',
    'GPUAllocation',
    
    # Orchestration
    'DistributedEvaluationOrchestrator',
    'WorkloadPriority',
    'GPUClusterState',
    'WorkloadTask',
    'GPUAllocationPlan',
    'ClusterMetrics',
    
    # Performance monitoring
    'MultiGPUPerformanceMonitor',
    'MetricType',
    'AlertSeverity',
    'PerformanceMetric',
    'GPUMetrics',
    'CommunicationMetrics',
    'SystemMetrics',
    'PerformanceAlert',
    'PerformanceSummary'
]

# Version info
__version__ = "1.0.0"
__author__ = "LLM Evaluation Framework Team"
__description__ = "Distributed evaluation engine for large language models"