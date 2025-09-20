"""
Core shared interfaces for hybrid evaluation architecture

This module provides abstract base classes and data structures
that are shared between lightweight and distributed evaluation engines.
"""

from .evaluation_interfaces import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationEngine,
    ResourceRequirements,
    EngineCapabilities
)

__all__ = [
    'EvaluationRequest',
    'EvaluationResult', 
    'EvaluationEngine',
    'ResourceRequirements',
    'EngineCapabilities'
]