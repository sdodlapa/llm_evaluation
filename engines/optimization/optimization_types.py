"""
Optimization Engine Data Types and Structures
Core data structures for Phase 4 optimization engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import time
from datetime import datetime


class StrategyType(Enum):
    """Available evaluation strategies"""
    LIGHTWEIGHT = "lightweight"
    DISTRIBUTED_TENSOR = "distributed_tensor_parallel"
    DISTRIBUTED_PIPELINE = "distributed_pipeline_parallel"
    DISTRIBUTED_HYBRID = "distributed_hybrid_parallel"
    AUTO_SELECT = "auto_select"


class OptimizationGoal(Enum):
    """Optimization objectives"""
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_MEMORY = "minimize_memory"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCED = "balanced"


@dataclass
class ModelProfile:
    """Profile of a model's characteristics"""
    name: str
    size_gb: float
    parameter_count: Optional[int] = None
    architecture: Optional[str] = None
    context_length: Optional[int] = None
    attention_type: Optional[str] = None
    
    # Performance characteristics
    memory_efficiency: Optional[float] = None  # Memory usage per parameter
    compute_intensity: Optional[float] = None  # FLOPS per parameter
    parallelization_friendly: Optional[bool] = None
    
    # Historical performance data
    average_eval_time: Optional[float] = None
    success_rate: float = 1.0
    
    def __post_init__(self):
        """Validate and compute derived fields"""
        if self.size_gb <= 0:
            raise ValueError("Model size must be positive")
        
        # Estimate parameter count if not provided
        if self.parameter_count is None and self.size_gb > 0:
            # Rough estimate: 2 bytes per parameter (FP16)
            self.parameter_count = int(self.size_gb * 1024**3 / 2)


@dataclass
class HardwareProfile:
    """Profile of available hardware resources"""
    num_gpus: int
    gpu_memory_gb: List[float]
    gpu_compute_capability: List[str]
    cpu_cores: int
    system_memory_gb: float
    
    # Network characteristics for multi-GPU
    inter_gpu_bandwidth_gbps: Optional[float] = None
    network_latency_ms: Optional[float] = None
    
    # Current utilization
    gpu_utilization: List[float] = field(default_factory=list)
    memory_utilization: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate hardware profile"""
        if self.num_gpus != len(self.gpu_memory_gb):
            raise ValueError("Number of GPUs must match memory list length")
        
        if len(self.gpu_utilization) == 0:
            self.gpu_utilization = [0.0] * self.num_gpus
        
        if len(self.memory_utilization) == 0:
            self.memory_utilization = [0.0] * self.num_gpus
    
    @property
    def total_gpu_memory_gb(self) -> float:
        """Total GPU memory across all devices"""
        return sum(self.gpu_memory_gb)
    
    @property
    def available_gpu_memory_gb(self) -> List[float]:
        """Available memory per GPU considering utilization"""
        return [
            memory * (1 - util) 
            for memory, util in zip(self.gpu_memory_gb, self.memory_utilization)
        ]
    
    @property
    def is_multi_gpu(self) -> bool:
        """Whether multi-GPU strategies are available"""
        return self.num_gpus > 1


@dataclass
class EvaluationProfile:
    """Profile of evaluation requirements and constraints"""
    dataset_size: int
    batch_size: Optional[int] = None
    max_eval_time_minutes: Optional[float] = None
    memory_limit_gb: Optional[float] = None
    cost_limit_usd: Optional[float] = None
    
    # Quality requirements
    min_success_rate: float = 0.95
    confidence_level: float = 0.95
    
    # Scheduling constraints
    priority: int = 5  # 1-10 scale, 10 = highest
    deadline: Optional[datetime] = None
    
    # Optimization preferences
    optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED
    allow_distributed: bool = True
    allow_caching: bool = True


@dataclass
class PerformancePrediction:
    """Predicted performance for a strategy"""
    strategy: StrategyType
    estimated_time_minutes: float
    estimated_memory_gb: float
    estimated_cost_usd: float
    confidence: float  # 0-1 scale
    
    # Resource requirements
    required_gpus: int
    gpu_memory_per_device: List[float]
    peak_memory_gb: float
    
    # Quality predictions
    expected_success_rate: float
    potential_bottlenecks: List[str]
    risk_factors: List[str]
    
    def __post_init__(self):
        """Validate prediction"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if not 0 <= self.expected_success_rate <= 1:
            raise ValueError("Success rate must be between 0 and 1")


@dataclass
class StrategyRecommendation:
    """Recommendation for evaluation strategy"""
    recommended_strategy: StrategyType
    prediction: PerformancePrediction
    alternatives: List[PerformancePrediction]
    
    # Reasoning
    recommendation_reason: str
    trade_offs: List[str]
    warnings: List[str]
    
    # Configuration
    optimal_batch_size: Optional[int] = None
    suggested_settings: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_score(self) -> float:
        """Overall confidence in the recommendation"""
        return self.prediction.confidence
    
    def get_alternative_by_strategy(self, strategy: StrategyType) -> Optional[PerformancePrediction]:
        """Get alternative prediction for specific strategy"""
        for alt in self.alternatives:
            if alt.strategy == strategy:
                return alt
        return None


@dataclass
class OptimizationSettings:
    """Configuration for optimization controller"""
    optimization_goal: OptimizationGoal
    max_evaluation_time_minutes: float
    memory_safety_margin: float = 0.1  # 10% safety margin
    
    # Dynamic optimization
    enable_dynamic_batching: bool = True
    enable_memory_optimization: bool = True
    enable_caching: bool = True
    
    # Quality control
    min_confidence_threshold: float = 0.7
    max_retries: int = 3
    enable_fallback_strategies: bool = True
    
    # Performance tuning
    batch_size_multiplier: float = 1.0
    memory_optimization_level: int = 1  # 0=off, 1=basic, 2=aggressive
    
    def validate(self):
        """Validate optimization settings"""
        if self.max_evaluation_time_minutes <= 0:
            raise ValueError("Max evaluation time must be positive")
        if not 0 <= self.memory_safety_margin <= 0.5:
            raise ValueError("Memory safety margin must be between 0 and 0.5")
        if not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")


@dataclass
class OptimizationMetrics:
    """Metrics for tracking optimization performance"""
    # Strategy selection metrics (required fields)
    strategy_selected: StrategyType
    selection_time_ms: float
    
    # Optional fields with defaults
    timestamp: datetime = field(default_factory=datetime.now)
    prediction_accuracy: Optional[float] = None
    
    # Performance metrics
    actual_eval_time_minutes: Optional[float] = None
    actual_memory_gb: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    
    # Quality metrics
    success_rate: float = 1.0
    quality_score: Optional[float] = None
    
    # Resource utilization
    gpu_utilization: List[float] = field(default_factory=list)
    memory_efficiency: Optional[float] = None
    
    # Optimization effectiveness
    time_improvement_pct: Optional[float] = None
    memory_savings_pct: Optional[float] = None
    cost_savings_pct: Optional[float] = None
    
    def compute_prediction_error(self, prediction: PerformancePrediction) -> Dict[str, float]:
        """Compute prediction error metrics"""
        errors = {}
        
        if self.actual_eval_time_minutes is not None:
            time_error = abs(self.actual_eval_time_minutes - prediction.estimated_time_minutes)
            errors['time_error_pct'] = (time_error / prediction.estimated_time_minutes) * 100
        
        if self.actual_memory_gb is not None:
            memory_error = abs(self.actual_memory_gb - prediction.estimated_memory_gb)
            errors['memory_error_pct'] = (memory_error / prediction.estimated_memory_gb) * 100
        
        if self.actual_cost_usd is not None:
            cost_error = abs(self.actual_cost_usd - prediction.estimated_cost_usd)
            errors['cost_error_pct'] = (cost_error / prediction.estimated_cost_usd) * 100
        
        return errors


# Utility functions
def create_default_hardware_profile(num_gpus: int = 1, gpu_memory_gb: float = 16.0) -> HardwareProfile:
    """Create a default hardware profile for testing"""
    return HardwareProfile(
        num_gpus=num_gpus,
        gpu_memory_gb=[gpu_memory_gb] * num_gpus,
        gpu_compute_capability=["8.0"] * num_gpus,
        cpu_cores=16,
        system_memory_gb=64.0,
        inter_gpu_bandwidth_gbps=300.0 if num_gpus > 1 else None
    )


def create_default_evaluation_profile(dataset_size: int = 1000) -> EvaluationProfile:
    """Create a default evaluation profile for testing"""
    return EvaluationProfile(
        dataset_size=dataset_size,
        batch_size=None,  # Will be optimized
        max_eval_time_minutes=30.0,
        optimization_goal=OptimizationGoal.BALANCED
    )