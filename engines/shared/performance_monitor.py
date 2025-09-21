"""
Simple Performance Monitor
Non-intrusive performance tracking for model operations

This module provides decorator-based performance monitoring that
integrates cleanly without modifying existing code.
"""

import time
import functools
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a monitored operation"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_time(self) -> float:
        """Average execution time"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def recent_avg(self) -> float:
        """Average of recent executions"""
        return sum(self.recent_times) / len(self.recent_times) if self.recent_times else 0.0
    
    def update(self, execution_time: float):
        """Update metrics with new execution time"""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.recent_times.append(execution_time)

class SimplePerformanceMonitor:
    """
    Simple, thread-safe performance monitor
    
    Provides decorator-based monitoring with minimal overhead
    and no impact on existing code structure.
    """
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.Lock()
        self.enabled = True
        
        logger.info("Initialized SimplePerformanceMonitor")
    
    def monitor(self, name: str = None) -> Callable:
        """
        Decorator for monitoring function performance
        
        Args:
            name: Custom name for the operation (defaults to function name)
            
        Example:
            @monitor.monitor("model_inference")
            def run_model(inputs):
                return model(inputs)
        """
        def decorator(func: Callable) -> Callable:
            operation_name = name or f"{func.__module__}.{func.__qualname__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    self._record_execution(operation_name, execution_time)
            
            return wrapper
        return decorator
    
    def time_operation(self, name: str):
        """
        Context manager for timing operations
        
        Example:
            with monitor.time_operation("data_processing"):
                process_data()
        """
        return _OperationTimer(self, name)
    
    def _record_execution(self, name: str, execution_time: float):
        """Record execution time for an operation"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetrics(name)
            
            self.metrics[name].update(execution_time)
    
    def get_metrics(self, name: str = None) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Args:
            name: Specific operation name (None for all metrics)
            
        Returns:
            Dictionary of metrics
        """
        with self.lock:
            if name:
                metric = self.metrics.get(name)
                if metric:
                    return self._format_metric(metric)
                return {}
            
            return {
                op_name: self._format_metric(metric)
                for op_name, metric in self.metrics.items()
            }
    
    def _format_metric(self, metric: PerformanceMetrics) -> Dict[str, Any]:
        """Format metric for output"""
        return {
            "call_count": metric.call_count,
            "total_time": round(metric.total_time, 4),
            "avg_time": round(metric.avg_time, 4),
            "min_time": round(metric.min_time, 4),
            "max_time": round(metric.max_time, 4),
            "recent_avg": round(metric.recent_avg, 4),
            "calls_per_second": (
                metric.call_count / metric.total_time 
                if metric.total_time > 0 else 0.0
            )
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored operations"""
        with self.lock:
            if not self.metrics:
                return {"status": "No operations monitored"}
            
            # Calculate totals
            total_calls = sum(m.call_count for m in self.metrics.values())
            total_time = sum(m.total_time for m in self.metrics.values())
            
            # Find top operations
            sorted_metrics = sorted(
                self.metrics.values(),
                key=lambda m: m.total_time,
                reverse=True
            )
            
            return {
                "total_operations": len(self.metrics),
                "total_calls": total_calls,
                "total_time": round(total_time, 4),
                "top_operations": [
                    {
                        "name": m.name,
                        "time_share": round(m.total_time / total_time * 100, 1),
                        "avg_time": round(m.avg_time, 4),
                        "call_count": m.call_count
                    }
                    for m in sorted_metrics[:5]
                ],
                "enabled": self.enabled
            }
    
    def reset_metrics(self, name: str = None):
        """Reset metrics for specific operation or all operations"""
        with self.lock:
            if name:
                if name in self.metrics:
                    del self.metrics[name]
            else:
                self.metrics.clear()
    
    def enable(self):
        """Enable performance monitoring"""
        self.enabled = True
        logger.info("Performance monitoring enabled")
    
    def disable(self):
        """Disable performance monitoring"""
        self.enabled = False
        logger.info("Performance monitoring disabled")

class _OperationTimer:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: SimplePerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        if self.monitor.enabled:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor.enabled and self.start_time:
            end_time = time.perf_counter()
            execution_time = end_time - self.start_time
            self.monitor._record_execution(self.name, execution_time)

# Global monitor instance for convenience
performance_monitor = SimplePerformanceMonitor()

# Convenience decorators
def monitor_inference(func: Callable) -> Callable:
    """Convenience decorator for monitoring inference operations"""
    return performance_monitor.monitor("inference")(func)

def monitor_compilation(func: Callable) -> Callable:
    """Convenience decorator for monitoring compilation operations"""
    return performance_monitor.monitor("compilation")(func)

def monitor_evaluation(func: Callable) -> Callable:
    """Convenience decorator for monitoring evaluation operations"""
    return performance_monitor.monitor("evaluation")(func)