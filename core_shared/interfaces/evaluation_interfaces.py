"""
Core evaluation interfaces for hybrid architecture

Defines abstract base classes and data structures that are shared
between lightweight and distributed evaluation engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time


class EngineType(Enum):
    """Types of evaluation engines available"""
    LIGHTWEIGHT = "lightweight"
    DISTRIBUTED = "distributed"
    AUTO = "auto"


class ResourceType(Enum):
    """Types of computational resources"""
    GPU_MEMORY = "gpu_memory"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    DISK_SPACE = "disk_space"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class ResourceRequirements:
    """Resource requirements for evaluation tasks"""
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 1
    system_memory_gb: float = 8.0
    disk_space_gb: float = 1.0
    network_bandwidth_mbps: float = 100.0
    
    # Multi-GPU specific
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    min_gpu_count: int = 1
    preferred_gpu_topology: str = "any"  # "nvlink", "pcie", "any"
    
    # Timing constraints
    max_evaluation_time_minutes: Optional[int] = None
    priority: str = "medium"  # "high", "medium", "low"


@dataclass
class EngineCapabilities:
    """Capabilities and constraints of an evaluation engine"""
    engine_type: EngineType
    max_model_size_gb: float
    supports_tensor_parallel: bool = False
    supports_pipeline_parallel: bool = False
    max_gpu_count: int = 1
    supported_dtypes: List[str] = field(default_factory=lambda: ["float16", "bfloat16"])
    supported_quantization: List[str] = field(default_factory=lambda: ["none", "awq", "gptq"])
    
    # Performance characteristics
    avg_tokens_per_second: Optional[float] = None
    memory_efficiency_score: float = 1.0  # 0.0 to 1.0
    startup_time_seconds: float = 60.0
    
    # Operational constraints
    requires_persistent_service: bool = False
    supports_batch_processing: bool = True
    max_concurrent_evaluations: int = 1


@dataclass
class EvaluationRequest:
    """Request for model evaluation"""
    # Core request information
    request_id: str
    model_config: Any  # Will be enhanced ModelConfig
    datasets: List[str]
    evaluation_params: Dict[str, Any]
    
    # Resource and performance constraints
    resource_constraints: Optional[ResourceRequirements] = None
    preferred_engine: Optional[EngineType] = None
    priority: str = "medium"
    
    # Metadata
    submitted_at: float = field(default_factory=time.time)
    submitter: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default resource constraints if not provided"""
        if self.resource_constraints is None:
            self.resource_constraints = ResourceRequirements()


@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    # Request identification
    request_id: str
    model_name: str
    dataset_name: str
    
    # Core evaluation metrics
    metrics: Dict[str, float]
    raw_outputs: List[str] = field(default_factory=list)
    
    # Performance and resource data
    performance_data: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    engine_used: EngineType = EngineType.AUTO
    execution_time_seconds: float = 0.0
    tokens_processed: int = 0
    tokens_per_second: float = 0.0
    
    # Quality and reliability metrics
    success: bool = True
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Timestamps
    started_at: float = field(default_factory=time.time)
    completed_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.execution_time_seconds > 0 and self.tokens_processed > 0:
            self.tokens_per_second = self.tokens_processed / self.execution_time_seconds


class EvaluationEngine(ABC):
    """Abstract base class for all evaluation engines"""
    
    def __init__(self, engine_id: str, capabilities: EngineCapabilities):
        self.engine_id = engine_id
        self.capabilities = capabilities
        self._is_initialized = False
        self._active_evaluations = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the evaluation engine
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def can_handle(self, request: EvaluationRequest) -> bool:
        """Determine if this engine can handle the evaluation request
        
        Args:
            request: The evaluation request to assess
            
        Returns:
            bool: True if engine can handle request, False otherwise
        """
        pass
    
    @abstractmethod
    def get_resource_requirements(self, request: EvaluationRequest) -> ResourceRequirements:
        """Calculate resource requirements for the evaluation request
        
        Args:
            request: The evaluation request to assess
            
        Returns:
            ResourceRequirements: Detailed resource requirements
        """
        pass
    
    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Execute the evaluation request
        
        Args:
            request: The evaluation request to execute
            
        Returns:
            EvaluationResult: Complete evaluation result
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up engine resources"""
        pass
    
    # Common utility methods
    def is_initialized(self) -> bool:
        """Check if engine is initialized"""
        return self._is_initialized
    
    def get_active_evaluations(self) -> Dict[str, EvaluationRequest]:
        """Get currently active evaluations"""
        return self._active_evaluations.copy()
    
    def get_capabilities(self) -> EngineCapabilities:
        """Get engine capabilities"""
        return self.capabilities
    
    def validate_request(self, request: EvaluationRequest) -> List[str]:
        """Validate evaluation request and return list of issues
        
        Args:
            request: Request to validate
            
        Returns:
            List[str]: List of validation issues (empty if valid)
        """
        issues = []
        
        # Basic validation
        if not request.request_id:
            issues.append("Request ID is required")
        
        if not request.model_config:
            issues.append("Model configuration is required")
        
        if not request.datasets:
            issues.append("At least one dataset is required")
        
        # Resource validation
        if request.resource_constraints:
            if request.resource_constraints.tensor_parallel_size > self.capabilities.max_gpu_count:
                issues.append(f"Requested tensor parallel size ({request.resource_constraints.tensor_parallel_size}) exceeds engine limit ({self.capabilities.max_gpu_count})")
        
        return issues


class EngineRegistry:
    """Registry for managing evaluation engines"""
    
    def __init__(self):
        self._engines: Dict[str, EvaluationEngine] = {}
        self._engine_priorities: Dict[EngineType, int] = {
            EngineType.LIGHTWEIGHT: 1,
            EngineType.DISTRIBUTED: 2
        }
    
    def register_engine(self, engine: EvaluationEngine):
        """Register an evaluation engine"""
        self._engines[engine.engine_id] = engine
    
    def get_engine(self, engine_id: str) -> Optional[EvaluationEngine]:
        """Get engine by ID"""
        return self._engines.get(engine_id)
    
    def get_compatible_engines(self, request: EvaluationRequest) -> List[EvaluationEngine]:
        """Get list of engines that can handle the request"""
        compatible = []
        for engine in self._engines.values():
            if engine.can_handle(request):
                compatible.append(engine)
        
        # Sort by engine type priority
        compatible.sort(key=lambda e: self._engine_priorities.get(e.capabilities.engine_type, 999))
        return compatible
    
    def select_optimal_engine(self, request: EvaluationRequest) -> Optional[EvaluationEngine]:
        """Select the optimal engine for the request"""
        compatible_engines = self.get_compatible_engines(request)
        
        if not compatible_engines:
            return None
        
        # If specific engine requested, try to honor it
        if request.preferred_engine and request.preferred_engine != EngineType.AUTO:
            for engine in compatible_engines:
                if engine.capabilities.engine_type == request.preferred_engine:
                    return engine
        
        # Otherwise return highest priority compatible engine
        return compatible_engines[0]