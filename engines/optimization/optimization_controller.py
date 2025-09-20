"""
Optimization Controller: Real-Time Parameter Tuning
Advanced component for dynamic optimization during evaluation execution
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque

from .optimization_types import (
    OptimizationGoal,
    OptimizationSettings,
    OptimizationMetrics,
    StrategyType
)

logger = logging.getLogger(__name__)


class OptimizationAction(Enum):
    """Available optimization actions"""
    ADJUST_BATCH_SIZE = "adjust_batch_size"
    ADJUST_MEMORY_ALLOCATION = "adjust_memory_allocation"
    ENABLE_MEMORY_OPTIMIZATION = "enable_memory_optimization"
    ADJUST_PARALLELISM = "adjust_parallelism"
    TRIGGER_GARBAGE_COLLECTION = "trigger_garbage_collection"
    SWITCH_STRATEGY = "switch_strategy"
    ABORT_EVALUATION = "abort_evaluation"


@dataclass
class OptimizationEvent:
    """Event triggered during optimization"""
    timestamp: float
    action: OptimizationAction
    parameters: Dict[str, Any]
    reason: str
    success: bool = True
    impact: Optional[str] = None


@dataclass
class ResourceMonitorData:
    """Real-time resource monitoring data"""
    timestamp: float
    gpu_utilization: List[float]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    gpu_temperature: List[float]
    system_memory_used: float
    system_memory_total: float
    
    @property
    def gpu_memory_utilization(self) -> List[float]:
        """GPU memory utilization percentages"""
        return [used / total if total > 0 else 0 
                for used, total in zip(self.gpu_memory_used, self.gpu_memory_total)]
    
    @property
    def max_gpu_memory_utilization(self) -> float:
        """Maximum GPU memory utilization across all GPUs"""
        utils = self.gpu_memory_utilization
        return max(utils) if utils else 0.0
    
    @property
    def avg_gpu_utilization(self) -> float:
        """Average GPU compute utilization"""
        return sum(self.gpu_utilization) / len(self.gpu_utilization) if self.gpu_utilization else 0.0


class OptimizationController:
    """
    Real-time optimization controller for evaluation performance
    
    Monitors evaluation progress and resource usage, making dynamic adjustments
    to optimize performance based on configured goals and constraints.
    """
    
    def __init__(self, settings: OptimizationSettings):
        self.settings = settings
        self.settings.validate()
        
        # State tracking
        self.is_active = False
        self.current_evaluation_id: Optional[str] = None
        self.optimization_events: List[OptimizationEvent] = []
        self.resource_history: deque = deque(maxlen=100)  # Last 100 monitoring points
        
        # Dynamic parameters
        self.current_batch_size: Optional[int] = None
        self.current_memory_optimization_level: int = self.settings.memory_optimization_level
        self.last_optimization_time: float = 0
        
        # Callbacks for optimization actions
        self.action_callbacks: Dict[OptimizationAction, Callable] = {}
        
        # Performance tracking
        self.performance_baseline: Optional[Dict[str, float]] = None
        self.recent_performance: deque = deque(maxlen=10)
        
        # Threading for monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_stop_event = threading.Event()
        
        logger.info(f"OptimizationController initialized with goal: {self.settings.optimization_goal.value}")
    
    def start_optimization(self, evaluation_id: str, initial_batch_size: int = None):
        """Start optimization for a new evaluation"""
        if self.is_active:
            logger.warning("Optimization already active, stopping previous session")
            self.stop_optimization()
        
        self.is_active = True
        self.current_evaluation_id = evaluation_id
        self.current_batch_size = initial_batch_size
        self.optimization_events.clear()
        self.resource_history.clear()
        self.recent_performance.clear()
        self.performance_baseline = None
        self.last_optimization_time = time.time()
        
        # Start monitoring thread
        self.monitor_stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started optimization for evaluation {evaluation_id}")
        
        # Record start event
        self._record_event(
            OptimizationAction.ADJUST_BATCH_SIZE,
            {"batch_size": initial_batch_size},
            "Optimization session started"
        )
    
    def stop_optimization(self) -> OptimizationMetrics:
        """Stop optimization and return final metrics"""
        if not self.is_active:
            logger.warning("Optimization not active")
            return self._create_empty_metrics()
        
        self.is_active = False
        
        # Stop monitoring thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_stop_event.set()
            self.monitor_thread.join(timeout=5)
        
        # Calculate final metrics
        metrics = self._calculate_final_metrics()
        
        logger.info(f"Stopped optimization for evaluation {self.current_evaluation_id}")
        return metrics
    
    def register_action_callback(self, action: OptimizationAction, callback: Callable):
        """Register callback function for optimization actions"""
        self.action_callbacks[action] = callback
        logger.debug(f"Registered callback for action: {action.value}")
    
    def report_resource_usage(self, resource_data: ResourceMonitorData):
        """Report current resource usage for optimization decisions"""
        if not self.is_active:
            return
        
        self.resource_history.append(resource_data)
        
        # Check if optimization is needed
        if self._should_optimize(resource_data):
            self._perform_optimization(resource_data)
    
    def report_performance_metrics(self, throughput: float, latency: float, quality_score: float = None):
        """Report current performance metrics"""
        if not self.is_active:
            return
        
        performance = {
            'timestamp': time.time(),
            'throughput': throughput,
            'latency': latency,
            'quality_score': quality_score
        }
        
        self.recent_performance.append(performance)
        
        # Set baseline if this is the first measurement
        if self.performance_baseline is None:
            self.performance_baseline = {
                'throughput': throughput,
                'latency': latency,
                'quality_score': quality_score
            }
            logger.debug(f"Set performance baseline: {self.performance_baseline}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self.monitor_stop_event.wait(timeout=5):  # Check every 5 seconds
            if not self.is_active:
                break
            
            # Perform periodic optimization checks
            self._periodic_optimization_check()
    
    def _should_optimize(self, resource_data: ResourceMonitorData) -> bool:
        """Determine if optimization action is needed"""
        # Don't optimize too frequently
        time_since_last = time.time() - self.last_optimization_time
        if time_since_last < 30:  # Wait at least 30 seconds between optimizations
            return False
        
        # Check for optimization triggers
        
        # Memory pressure trigger
        max_memory_util = resource_data.max_gpu_memory_utilization
        if max_memory_util > 0.9:  # 90% memory usage
            logger.info(f"High memory utilization detected: {max_memory_util:.1%}")
            return True
        
        # Low GPU utilization trigger
        avg_gpu_util = resource_data.avg_gpu_utilization
        if avg_gpu_util < 0.5 and len(self.resource_history) > 5:  # 50% GPU utilization
            # Check if this is consistent
            recent_utils = [data.avg_gpu_utilization for data in list(self.resource_history)[-5:]]
            if all(util < 0.6 for util in recent_utils):
                logger.info(f"Low GPU utilization detected: {avg_gpu_util:.1%}")
                return True
        
        # Temperature trigger
        max_temp = max(resource_data.gpu_temperature) if resource_data.gpu_temperature else 0
        if max_temp > 85:  # 85°C
            logger.info(f"High GPU temperature detected: {max_temp}°C")
            return True
        
        # Performance degradation trigger
        if self._detect_performance_degradation():
            logger.info("Performance degradation detected")
            return True
        
        return False
    
    def _perform_optimization(self, resource_data: ResourceMonitorData):
        """Perform optimization based on current state"""
        self.last_optimization_time = time.time()
        
        # Determine optimization strategy based on settings and current state
        optimization_strategy = self._determine_optimization_strategy(resource_data)
        
        for action, params in optimization_strategy:
            try:
                success = self._execute_optimization_action(action, params)
                self._record_event(action, params, self._get_action_reason(action, resource_data), success)
                
                if success:
                    logger.info(f"Successfully executed optimization action: {action.value}")
                else:
                    logger.warning(f"Failed to execute optimization action: {action.value}")
                    
            except Exception as e:
                logger.error(f"Error executing optimization action {action.value}: {e}")
                self._record_event(action, params, f"Error: {str(e)}", False)
    
    def _determine_optimization_strategy(self, resource_data: ResourceMonitorData) -> List[tuple]:
        """Determine what optimization actions to take"""
        actions = []
        
        max_memory_util = resource_data.max_gpu_memory_utilization
        avg_gpu_util = resource_data.avg_gpu_utilization
        
        # Memory optimization strategy
        if max_memory_util > 0.9:
            if self.settings.enable_memory_optimization:
                # Aggressive memory optimization
                if self.current_memory_optimization_level < 2:
                    actions.append((OptimizationAction.ENABLE_MEMORY_OPTIMIZATION, {"level": 2}))
                
                # Reduce batch size if possible
                if self.current_batch_size and self.current_batch_size > 1:
                    new_batch_size = max(1, self.current_batch_size // 2)
                    actions.append((OptimizationAction.ADJUST_BATCH_SIZE, {"batch_size": new_batch_size}))
                
                # Trigger garbage collection
                actions.append((OptimizationAction.TRIGGER_GARBAGE_COLLECTION, {}))
            
            elif max_memory_util > 0.95:
                # Critical memory situation - consider aborting
                actions.append((OptimizationAction.ABORT_EVALUATION, {"reason": "Critical memory usage"}))
        
        # Performance optimization strategy
        elif avg_gpu_util < 0.5:
            if self.settings.enable_dynamic_batching and self.current_batch_size:
                # Increase batch size to improve utilization
                new_batch_size = min(64, self.current_batch_size * 2)
                if new_batch_size != self.current_batch_size:
                    actions.append((OptimizationAction.ADJUST_BATCH_SIZE, {"batch_size": new_batch_size}))
        
        # Temperature management
        max_temp = max(resource_data.gpu_temperature) if resource_data.gpu_temperature else 0
        if max_temp > 85:
            # Reduce workload to manage temperature
            if self.current_batch_size and self.current_batch_size > 1:
                new_batch_size = max(1, int(self.current_batch_size * 0.8))
                actions.append((OptimizationAction.ADJUST_BATCH_SIZE, {"batch_size": new_batch_size}))
        
        return actions
    
    def _execute_optimization_action(self, action: OptimizationAction, params: Dict[str, Any]) -> bool:
        """Execute optimization action using registered callbacks"""
        if action not in self.action_callbacks:
            logger.warning(f"No callback registered for action: {action.value}")
            return False
        
        try:
            callback = self.action_callbacks[action]
            result = callback(params)
            
            # Update internal state based on action
            if action == OptimizationAction.ADJUST_BATCH_SIZE:
                self.current_batch_size = params.get("batch_size")
            elif action == OptimizationAction.ENABLE_MEMORY_OPTIMIZATION:
                self.current_memory_optimization_level = params.get("level", self.current_memory_optimization_level)
            
            return result if isinstance(result, bool) else True
            
        except Exception as e:
            logger.error(f"Callback error for action {action.value}: {e}")
            return False
    
    def _get_action_reason(self, action: OptimizationAction, resource_data: ResourceMonitorData) -> str:
        """Get human-readable reason for optimization action"""
        if action == OptimizationAction.ADJUST_BATCH_SIZE:
            if resource_data.max_gpu_memory_utilization > 0.9:
                return "Reducing batch size due to high memory usage"
            elif resource_data.avg_gpu_utilization < 0.5:
                return "Increasing batch size to improve GPU utilization"
            else:
                return "Adjusting batch size for optimization"
        
        elif action == OptimizationAction.ENABLE_MEMORY_OPTIMIZATION:
            return f"Enabling memory optimization level {self.current_memory_optimization_level}"
        
        elif action == OptimizationAction.TRIGGER_GARBAGE_COLLECTION:
            return "Triggering garbage collection to free memory"
        
        elif action == OptimizationAction.ABORT_EVALUATION:
            return "Aborting evaluation due to resource constraints"
        
        else:
            return f"Executing {action.value} for optimization"
    
    def _detect_performance_degradation(self) -> bool:
        """Detect if performance has degraded significantly"""
        if len(self.recent_performance) < 3 or not self.performance_baseline:
            return False
        
        # Check recent throughput vs baseline
        recent_throughputs = [p['throughput'] for p in list(self.recent_performance)[-3:]]
        avg_recent_throughput = sum(recent_throughputs) / len(recent_throughputs)
        
        baseline_throughput = self.performance_baseline['throughput']
        if baseline_throughput > 0:
            degradation = (baseline_throughput - avg_recent_throughput) / baseline_throughput
            return degradation > 0.2  # 20% degradation threshold
        
        return False
    
    def _periodic_optimization_check(self):
        """Perform periodic optimization checks"""
        if not self.resource_history:
            return
        
        latest_data = self.resource_history[-1]
        
        # Check for long-term trends
        if len(self.resource_history) >= 10:
            # Analyze memory usage trend
            memory_utils = [data.max_gpu_memory_utilization for data in self.resource_history]
            recent_trend = sum(memory_utils[-5:]) / 5 - sum(memory_utils[-10:-5]) / 5
            
            if recent_trend > 0.1:  # Memory usage increasing by 10%
                logger.info("Detected increasing memory usage trend")
                if self._should_optimize(latest_data):
                    self._perform_optimization(latest_data)
    
    def _record_event(self, action: OptimizationAction, parameters: Dict[str, Any], reason: str, success: bool = True):
        """Record optimization event"""
        event = OptimizationEvent(
            timestamp=time.time(),
            action=action,
            parameters=parameters.copy(),
            reason=reason,
            success=success
        )
        self.optimization_events.append(event)
        logger.debug(f"Recorded optimization event: {action.value} - {reason}")
    
    def _calculate_final_metrics(self) -> OptimizationMetrics:
        """Calculate final optimization metrics"""
        if not self.optimization_events:
            return self._create_empty_metrics()
        
        # Calculate optimization effectiveness
        successful_optimizations = sum(1 for event in self.optimization_events if event.success)
        total_optimizations = len(self.optimization_events)
        
        # Calculate performance improvements if available
        time_improvement = None
        memory_savings = None
        
        if self.performance_baseline and self.recent_performance:
            recent_perf = list(self.recent_performance)[-1]
            if self.performance_baseline['throughput'] > 0:
                improvement = (recent_perf['throughput'] - self.performance_baseline['throughput']) / self.performance_baseline['throughput']
                time_improvement = improvement * 100  # Convert to percentage
        
        # Calculate average resource utilization
        avg_gpu_util = []
        memory_efficiency = None
        
        if self.resource_history:
            all_gpu_utils = []
            all_memory_utils = []
            
            for data in self.resource_history:
                all_gpu_utils.extend(data.gpu_utilization)
                all_memory_utils.extend(data.gpu_memory_utilization)
            
            avg_gpu_util = [sum(all_gpu_utils) / len(all_gpu_utils)] if all_gpu_utils else []
            memory_efficiency = sum(all_memory_utils) / len(all_memory_utils) if all_memory_utils else None
        
        return OptimizationMetrics(
            strategy_selected=StrategyType.AUTO_SELECT,  # Placeholder
            selection_time_ms=0,  # Placeholder
            success_rate=successful_optimizations / total_optimizations if total_optimizations > 0 else 1.0,
            gpu_utilization=avg_gpu_util,
            memory_efficiency=memory_efficiency,
            time_improvement_pct=time_improvement,
            memory_savings_pct=memory_savings
        )
    
    def _create_empty_metrics(self) -> OptimizationMetrics:
        """Create empty metrics for cases with no optimization data"""
        return OptimizationMetrics(
            strategy_selected=StrategyType.AUTO_SELECT,
            selection_time_ms=0,
            success_rate=1.0
        )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization session"""
        if not self.is_active and not self.optimization_events:
            return {"status": "no_optimization_data"}
        
        summary = {
            "status": "active" if self.is_active else "completed",
            "evaluation_id": self.current_evaluation_id,
            "total_events": len(self.optimization_events),
            "successful_events": sum(1 for e in self.optimization_events if e.success),
            "current_batch_size": self.current_batch_size,
            "memory_optimization_level": self.current_memory_optimization_level,
            "optimization_goal": self.settings.optimization_goal.value
        }
        
        # Event breakdown
        event_counts = {}
        for event in self.optimization_events:
            action = event.action.value
            event_counts[action] = event_counts.get(action, 0) + 1
        summary["event_breakdown"] = event_counts
        
        # Recent resource status
        if self.resource_history:
            latest = self.resource_history[-1]
            summary["latest_resource_status"] = {
                "gpu_utilization": latest.avg_gpu_utilization,
                "memory_utilization": latest.max_gpu_memory_utilization,
                "max_temperature": max(latest.gpu_temperature) if latest.gpu_temperature else None
            }
        
        # Performance trend
        if len(self.recent_performance) >= 2:
            recent = list(self.recent_performance)
            first_perf = recent[0]
            latest_perf = recent[-1]
            
            if first_perf['throughput'] > 0:
                throughput_change = (latest_perf['throughput'] - first_perf['throughput']) / first_perf['throughput'] * 100
                summary["throughput_change_pct"] = throughput_change
        
        return summary
    
    def export_optimization_log(self) -> List[Dict[str, Any]]:
        """Export optimization events log"""
        return [
            {
                "timestamp": event.timestamp,
                "action": event.action.value,
                "parameters": event.parameters,
                "reason": event.reason,
                "success": event.success,
                "impact": event.impact
            }
            for event in self.optimization_events
        ]