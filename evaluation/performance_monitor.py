"""
Enhanced Performance Monitoring for LLM Evaluation
Captures real-time GPU/memory metrics during model evaluation
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available. GPU monitoring will be limited.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available. Using alternative GPU monitoring.")

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement"""
    timestamp: float
    gpu_utilization_percent: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_temperature_c: float
    cpu_percent: float
    ram_used_gb: float
    tokens_processed: int
    elapsed_time_seconds: float

@dataclass
class EvaluationMetrics:
    """Aggregated metrics for a complete evaluation run"""
    model_name: str
    preset: str
    dataset: str
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    
    # Performance metrics
    avg_gpu_utilization: float
    peak_gpu_utilization: float
    avg_gpu_memory_gb: float
    peak_gpu_memory_gb: float
    avg_gpu_temperature: float
    peak_gpu_temperature: float
    
    # Throughput metrics  
    total_tokens_processed: int
    avg_throughput_tokens_per_second: float
    peak_throughput_tokens_per_second: float
    avg_latency_ms: float
    
    # Efficiency metrics
    memory_efficiency: float
    gpu_utilization_efficiency: float
    tokens_per_gb_memory: float
    
    # Dataset-specific metrics
    dataset_samples_processed: int
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    task_specific_metrics: Dict[str, Any] = field(default_factory=dict)

class GPUMonitor:
    """Real-time GPU monitoring using multiple backends"""
    
    def __init__(self):
        self.nvml_initialized = False
        self.gpu_id = 0
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                logger.info("NVML GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                
    def get_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            if self.nvml_initialized:
                info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                return float(info.gpu)
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].load * 100
        except Exception as e:
            logger.warning(f"Failed to get GPU utilization: {e}")
        return 0.0
    
    def get_memory_used(self) -> float:
        """Get current GPU memory usage in GB"""
        try:
            if self.nvml_initialized:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return info.used / (1024**3)  # Convert to GB
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].memoryUsed / 1024  # Convert to GB
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
        return 0.0
    
    def get_memory_total(self) -> float:
        """Get total GPU memory in GB"""
        try:
            if self.nvml_initialized:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return info.total / (1024**3)  # Convert to GB
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].memoryTotal / 1024  # Convert to GB
        except Exception as e:
            logger.warning(f"Failed to get GPU total memory: {e}")
        return 80.0  # Default H100 memory
    
    def get_temperature(self) -> float:
        """Get current GPU temperature in Celsius"""
        try:
            if self.nvml_initialized:
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                return float(temp)
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].temperature
        except Exception as e:
            logger.warning(f"Failed to get GPU temperature: {e}")
        return 0.0

class ThroughputCalculator:
    """Calculate real-time throughput and latency metrics"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.token_times: List[tuple] = []  # (timestamp, tokens)
        self.request_times: List[tuple] = []  # (start_time, end_time)
        
    def record_tokens(self, num_tokens: int):
        """Record tokens processed at current time"""
        current_time = time.time()
        self.token_times.append((current_time, num_tokens))
        
        # Keep only recent records
        cutoff_time = current_time - 60  # Last 60 seconds
        self.token_times = [(t, tokens) for t, tokens in self.token_times if t > cutoff_time]
    
    def record_request(self, start_time: float, end_time: float):
        """Record request timing"""
        self.request_times.append((start_time, end_time))
        
        # Keep only recent records
        if len(self.request_times) > self.window_size:
            self.request_times = self.request_times[-self.window_size:]
    
    def get_current_throughput(self) -> float:
        """Calculate current tokens per second"""
        if len(self.token_times) < 2:
            return 0.0
        
        current_time = time.time()
        recent_time = current_time - 10  # Last 10 seconds
        
        recent_tokens = [(t, tokens) for t, tokens in self.token_times if t > recent_time]
        if len(recent_tokens) < 2:
            return 0.0
        
        total_tokens = sum(tokens for _, tokens in recent_tokens)
        time_span = recent_tokens[-1][0] - recent_tokens[0][0]
        
        return total_tokens / time_span if time_span > 0 else 0.0
    
    def get_avg_latency(self) -> float:
        """Calculate average latency in milliseconds"""
        if not self.request_times:
            return 0.0
        
        latencies = [(end - start) * 1000 for start, end in self.request_times]
        return sum(latencies) / len(latencies)

class LivePerformanceMonitor:
    """Main performance monitoring class"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.gpu_monitor = GPUMonitor()
        self.throughput_calculator = ThroughputCalculator()
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.snapshots: List[PerformanceSnapshot] = []
        
        self.start_time = None
        self.total_tokens_processed = 0
        
    def start_monitoring(self, model_name: str, preset: str, dataset: str):
        """Start real-time monitoring session"""
        logger.info(f"Starting performance monitoring for {model_name}_{preset} on {dataset}")
        
        self.model_name = model_name
        self.preset = preset
        self.dataset = dataset
        self.start_time = datetime.now()
        self.total_tokens_processed = 0
        self.snapshots = []
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitoring_loop(self):
        """Internal monitoring loop"""
        loop_start = time.time()
        
        while self.is_monitoring:
            try:
                snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    gpu_utilization_percent=self.gpu_monitor.get_utilization(),
                    gpu_memory_used_gb=self.gpu_monitor.get_memory_used(),
                    gpu_memory_total_gb=self.gpu_monitor.get_memory_total(),
                    gpu_temperature_c=self.gpu_monitor.get_temperature(),
                    cpu_percent=psutil.cpu_percent(interval=None),
                    ram_used_gb=psutil.virtual_memory().used / (1024**3),
                    tokens_processed=self.total_tokens_processed,
                    elapsed_time_seconds=time.time() - loop_start
                )
                
                self.snapshots.append(snapshot)
                
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def record_tokens_processed(self, num_tokens: int):
        """Record tokens processed for throughput calculation"""
        self.total_tokens_processed += num_tokens
        self.throughput_calculator.record_tokens(num_tokens)
    
    def record_request_timing(self, start_time: float, end_time: float):
        """Record request timing for latency calculation"""
        self.throughput_calculator.record_request(start_time, end_time)
    
    def stop_monitoring(self, dataset_samples_processed: int, 
                       accuracy_metrics: Dict[str, float] = None,
                       task_specific_metrics: Dict[str, Any] = None) -> EvaluationMetrics:
        """Stop monitoring and return aggregated metrics"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate aggregated metrics
        if self.snapshots:
            avg_gpu_util = sum(s.gpu_utilization_percent for s in self.snapshots) / len(self.snapshots)
            peak_gpu_util = max(s.gpu_utilization_percent for s in self.snapshots)
            avg_gpu_memory = sum(s.gpu_memory_used_gb for s in self.snapshots) / len(self.snapshots)
            peak_gpu_memory = max(s.gpu_memory_used_gb for s in self.snapshots)
            avg_gpu_temp = sum(s.gpu_temperature_c for s in self.snapshots) / len(self.snapshots)
            peak_gpu_temp = max(s.gpu_temperature_c for s in self.snapshots)
        else:
            avg_gpu_util = peak_gpu_util = 0.0
            avg_gpu_memory = peak_gpu_memory = 0.0
            avg_gpu_temp = peak_gpu_temp = 0.0
        
        # Calculate throughput metrics
        avg_throughput = self.total_tokens_processed / total_duration if total_duration > 0 else 0.0
        avg_latency = self.throughput_calculator.get_avg_latency()
        
        # Calculate efficiency metrics
        memory_efficiency = (avg_gpu_memory / self.gpu_monitor.get_memory_total()) * 100 if avg_gpu_memory > 0 else 0.0
        gpu_utilization_efficiency = avg_gpu_util
        tokens_per_gb = self.total_tokens_processed / avg_gpu_memory if avg_gpu_memory > 0 else 0.0
        
        metrics = EvaluationMetrics(
            model_name=self.model_name,
            preset=self.preset,
            dataset=self.dataset,
            start_time=self.start_time,
            end_time=end_time,
            total_duration_seconds=total_duration,
            
            # Performance metrics
            avg_gpu_utilization=avg_gpu_util,
            peak_gpu_utilization=peak_gpu_util,
            avg_gpu_memory_gb=avg_gpu_memory,
            peak_gpu_memory_gb=peak_gpu_memory,
            avg_gpu_temperature=avg_gpu_temp,
            peak_gpu_temperature=peak_gpu_temp,
            
            # Throughput metrics
            total_tokens_processed=self.total_tokens_processed,
            avg_throughput_tokens_per_second=avg_throughput,
            peak_throughput_tokens_per_second=self.throughput_calculator.get_current_throughput(),
            avg_latency_ms=avg_latency,
            
            # Efficiency metrics
            memory_efficiency=memory_efficiency,
            gpu_utilization_efficiency=gpu_utilization_efficiency,
            tokens_per_gb_memory=tokens_per_gb,
            
            # Dataset metrics
            dataset_samples_processed=dataset_samples_processed,
            accuracy_metrics=accuracy_metrics or {},
            task_specific_metrics=task_specific_metrics or {}
        )
        
        logger.info(f"Performance monitoring completed for {self.model_name}_{self.preset} on {self.dataset}")
        logger.info(f"Average throughput: {avg_throughput:.1f} tokens/sec, Peak memory: {peak_gpu_memory:.1f}GB")
        
        return metrics
    
    def save_detailed_snapshots(self, filepath: str):
        """Save detailed monitoring snapshots to file"""
        snapshot_data = [
            {
                'timestamp': s.timestamp,
                'gpu_utilization_percent': s.gpu_utilization_percent,
                'gpu_memory_used_gb': s.gpu_memory_used_gb,
                'gpu_temperature_c': s.gpu_temperature_c,
                'cpu_percent': s.cpu_percent,
                'tokens_processed': s.tokens_processed,
                'elapsed_time_seconds': s.elapsed_time_seconds
            }
            for s in self.snapshots
        ]
        
        with open(filepath, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"Detailed performance snapshots saved to {filepath}")