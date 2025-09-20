# Engines Package
# Comprehensive evaluation engine implementations

# Phase 2: Lightweight Engine
from .lightweight.lightweight_engine import LightweightEvaluationEngine

# Phase 3: Distributed Engine  
from .distributed.distributed_engine import DistributedEvaluationEngine, MockDistributedEngine
from .distributed.multi_gpu_model_loader import MultiGPUModelLoader
from .distributed.distributed_orchestrator import DistributedEvaluationOrchestrator
from .distributed.performance_monitor import MultiGPUPerformanceMonitor

# Phase 4: Optimization Engine
from .optimization.strategy_selector import StrategySelector
from .optimization.performance_predictor import PerformancePredictor
from .optimization.optimization_controller import OptimizationController
from .optimization.optimization_types import (
    StrategyType, OptimizationGoal, ModelProfile, HardwareProfile, 
    EvaluationProfile, OptimizationSettings, OptimizationMetrics
)

__all__ = [
    
    # Phase 2: Lightweight
    "LightweightEvaluationEngine",
    
    # Phase 3: Distributed
    "DistributedEvaluationEngine",
    "MockDistributedEngine", 
    "MultiGPUModelLoader",
    "DistributedEvaluationOrchestrator",
    "MultiGPUPerformanceMonitor",
    
    # Phase 4: Optimization
    "StrategySelector",
    "PerformancePredictor", 
    "OptimizationController",
    "StrategyType",
    "OptimizationGoal",
    "ModelProfile",
    "HardwareProfile",
    "EvaluationProfile", 
    "OptimizationSettings",
    "OptimizationMetrics"
]

# Version tracking
__version__ = "4.0.0"
__phases__ = {
    "Phase 2": "Lightweight Engine - Complete ✅",
    "Phase 3": "Distributed Engine - Complete ✅", 
    "Phase 4": "Optimization Engine - Complete ✅",
    "Phase 5": "Production Scaling - Planned 📋"
}