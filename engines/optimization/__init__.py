"""
Phase 4: Advanced Optimization Engine
Intelligent performance optimization and adaptive evaluation strategies
"""

# Core optimization components
from .strategy_selector import StrategySelector, StrategyRecommendation
from .performance_predictor import PerformancePredictor, PerformancePrediction
from .optimization_controller import OptimizationController, OptimizationSettings

# Optimization data structures
from .optimization_types import (
    ModelProfile,
    HardwareProfile,
    EvaluationProfile,
    OptimizationMetrics,
    StrategyType,
    OptimizationGoal
)

__all__ = [
    # Core components
    "StrategySelector",
    "PerformancePredictor", 
    "OptimizationController",
    
    # Data structures
    "StrategyRecommendation",
    "PerformancePrediction",
    "OptimizationSettings",
    "ModelProfile",
    "HardwareProfile", 
    "EvaluationProfile",
    "OptimizationMetrics",
    "StrategyType",
    "OptimizationGoal"
]

# Version info
__version__ = "4.0.0"
__phase__ = "Phase 4: Advanced Optimization Engine"