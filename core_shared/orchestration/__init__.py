"""
Orchestration layer for hybrid evaluation architecture

Provides intelligent engine selection, request routing, and resource management
for coordinating between lightweight and distributed evaluation engines.
"""

from .evaluation_orchestrator import EvaluationOrchestrator, OrchestratorConfig

__all__ = [
    'EvaluationOrchestrator',
    'OrchestratorConfig'
]