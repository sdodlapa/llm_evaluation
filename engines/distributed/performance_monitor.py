"""
Multi-GPU Performance Monitor

This module provides comprehensive performance monitoring for distributed
evaluation engine, tracking cross-GPU communication metrics, resource
utilization, bottleneck detection, and optimization recommendations.
"""

import time
import threading
import logging
import statistics
import json
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid

try:
    import torch
    import psutil
    import numpy as np
    TORCH_AVAILABLE = True
    PSUTIL_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_USAGE = "memory_usage"
    TEMPERATURE = "temperature"
    POWER_CONSUMPTION = "power_consumption"
    COMMUNICATION_LATENCY = "communication_latency"
    BANDWIDTH_UTILIZATION = "bandwidth_utilization"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: float
    metric_type: MetricType
    gpu_id: Optional[int]
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GPUMetrics:
    """Comprehensive GPU metrics"""
    gpu_id: int
    timestamp: float
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_utilization_percent: float
    temperature_celsius: float
    power_draw_watts: float
    clock_speed_mhz: int
    fan_speed_percent: float
    error_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "gpu_id": self.gpu_id,
            "timestamp": self.timestamp,
            "utilization_percent": self.utilization_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "memory_utilization_percent": self.memory_utilization_percent,
            "temperature_celsius": self.temperature_celsius,
            "power_draw_watts": self.power_draw_watts,
            "clock_speed_mhz": self.clock_speed_mhz,
            "fan_speed_percent": self.fan_speed_percent,
            "error_count": self.error_count
        }

@dataclass
class CommunicationMetrics:
    """Cross-GPU communication performance metrics"""
    timestamp: float
    source_gpu: int
    target_gpu: int
    operation_type: str  # "allreduce", "broadcast", "p2p", etc.
    data_size_bytes: int
    latency_ms: float
    bandwidth_gbps: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: float
    cpu_utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_io_read_mbps: float
    disk_io_write_mbps: float
    network_io_recv_mbps: float
    network_io_sent_mbps: float
    process_count: int
    load_average: Tuple[float, float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "disk_io_read_mbps": self.disk_io_read_mbps,
            "disk_io_write_mbps": self.disk_io_write_mbps,
            "network_io_recv_mbps": self.network_io_recv_mbps,
            "network_io_sent_mbps": self.network_io_sent_mbps,
            "process_count": self.process_count,
            "load_average": self.load_average
        }

@dataclass
class PerformanceAlert:
    """Performance alert/warning"""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    metric_type: MetricType
    gpu_id: Optional[int]
    title: str
    description: str
    current_value: float
    threshold_value: float
    recommendation: str
    auto_resolvable: bool = False

@dataclass
class PerformanceSummary:
    """Performance summary over a time period"""
    start_time: float
    end_time: float
    duration_seconds: float
    gpu_metrics: Dict[int, Dict[str, Any]]  # GPU ID -> metric stats
    communication_stats: Dict[str, Any]
    system_stats: Dict[str, Any]
    alerts_generated: int
    bottlenecks_detected: List[str]
    optimization_suggestions: List[str]

class MultiGPUPerformanceMonitor:
    """
    Advanced performance monitoring system for multi-GPU distributed evaluation.
    
    Features:
    - Real-time GPU metrics collection and analysis
    - Cross-GPU communication monitoring
    - Bottleneck detection and performance optimization
    - Automated alerting and recommendations
    - Historical performance tracking and reporting
    - Resource utilization optimization
    """
    
    def __init__(self, 
                 gpu_ids: List[int],
                 monitoring_interval: float = 1.0,
                 history_size: int = 3600,  # 1 hour at 1-second intervals
                 enable_alerts: bool = True):
        """
        Initialize multi-GPU performance monitor
        
        Args:
            gpu_ids: List of GPU IDs to monitor
            monitoring_interval: Metrics collection interval in seconds
            history_size: Number of data points to keep in history
            enable_alerts: Enable automated alerting
        """
        
        self.gpu_ids = gpu_ids
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        
        # Metrics storage
        self._gpu_metrics_history: Dict[int, deque] = {
            gpu_id: deque(maxlen=history_size) for gpu_id in gpu_ids
        }
        self._communication_metrics: deque = deque(maxlen=history_size)
        self._system_metrics_history: deque = deque(maxlen=history_size)
        
        # Alert system
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._alert_thresholds = self._initialize_alert_thresholds()
        
        # Monitoring control
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Performance tracking
        self._performance_stats: Dict[str, Any] = {}
        self._last_summary_time = time.time()
        
        # Communication tracking
        self._communication_operations = defaultdict(list)
        self._bandwidth_measurements = defaultdict(list)
        
        logger.info(f"Initialized performance monitor for {len(gpu_ids)} GPUs")
        logger.info(f"Monitoring interval: {monitoring_interval}s, History size: {history_size}")
    
    def start(self) -> None:
        """Start performance monitoring"""
        if self._running:
            logger.warning("Performance monitor already running")
            return
        
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop(self) -> None:
        """Stop performance monitoring"""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10.0)
        
        logger.info("Performance monitoring stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics snapshot"""
        with self._lock:
            current_time = time.time()
            
            # Collect current GPU metrics
            gpu_metrics = {}
            for gpu_id in self.gpu_ids:
                metrics = self._collect_gpu_metrics(gpu_id)
                if metrics:
                    gpu_metrics[gpu_id] = metrics.to_dict()
            
            # Collect system metrics
            system_metrics = self._collect_system_metrics()
            
            # Get recent communication stats
            recent_comm = [m for m in self._communication_metrics 
                          if current_time - m.timestamp < 60.0]  # Last minute
            
            communication_stats = self._analyze_communication_metrics(recent_comm)
            
            return {
                "timestamp": current_time,
                "gpu_metrics": gpu_metrics,
                "system_metrics": system_metrics.to_dict() if system_metrics else {},
                "communication_stats": communication_stats,
                "active_alerts": len(self._active_alerts),
                "monitoring_status": "running" if self._running else "stopped"
            }
    
    def get_performance_summary(self, duration_minutes: int = 60) -> PerformanceSummary:
        """Get performance summary for specified duration"""
        end_time = time.time()
        start_time = end_time - (duration_minutes * 60)
        
        with self._lock:
            # Analyze GPU metrics
            gpu_stats = {}
            for gpu_id in self.gpu_ids:
                gpu_history = [m for m in self._gpu_metrics_history[gpu_id] 
                              if start_time <= m.timestamp <= end_time]
                
                if gpu_history:
                    gpu_stats[gpu_id] = self._calculate_gpu_statistics(gpu_history)
            
            # Analyze communication metrics
            comm_history = [m for m in self._communication_metrics 
                           if start_time <= m.timestamp <= end_time]
            
            communication_stats = self._analyze_communication_metrics(comm_history)
            
            # Analyze system metrics
            system_history = [m for m in self._system_metrics_history 
                             if start_time <= m.timestamp <= end_time]
            
            system_stats = self._calculate_system_statistics(system_history)
            
            # Count alerts in period
            alerts_in_period = [a for a in self._alert_history 
                               if start_time <= a.timestamp <= end_time]
            
            # Detect bottlenecks and generate suggestions
            bottlenecks = self._detect_bottlenecks(gpu_stats, communication_stats, system_stats)
            suggestions = self._generate_optimization_suggestions(gpu_stats, communication_stats, bottlenecks)
            
            return PerformanceSummary(
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_minutes * 60,
                gpu_metrics=gpu_stats,
                communication_stats=communication_stats,
                system_stats=system_stats,
                alerts_generated=len(alerts_in_period),
                bottlenecks_detected=bottlenecks,
                optimization_suggestions=suggestions
            )
    
    def record_communication_event(self, 
                                   source_gpu: int,
                                   target_gpu: int,
                                   operation_type: str,
                                   data_size_bytes: int,
                                   latency_ms: float,
                                   success: bool = True,
                                   error_message: Optional[str] = None) -> None:
        """Record a cross-GPU communication event"""
        
        bandwidth_gbps = (data_size_bytes * 8) / (latency_ms * 1e6) if latency_ms > 0 else 0.0
        
        comm_metric = CommunicationMetrics(
            timestamp=time.time(),
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            operation_type=operation_type,
            data_size_bytes=data_size_bytes,
            latency_ms=latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self._communication_metrics.append(comm_metric)
            
            # Track by operation type
            self._communication_operations[operation_type].append(comm_metric)
            
            # Track bandwidth
            gpu_pair = (min(source_gpu, target_gpu), max(source_gpu, target_gpu))
            self._bandwidth_measurements[gpu_pair].append(bandwidth_gbps)
        
        # Check for communication performance alerts
        if self.enable_alerts:
            self._check_communication_alerts(comm_metric)
    
    def get_gpu_utilization_trend(self, gpu_id: int, duration_minutes: int = 30) -> Dict[str, Any]:
        """Get GPU utilization trend analysis"""
        end_time = time.time()
        start_time = end_time - (duration_minutes * 60)
        
        with self._lock:
            gpu_history = [m for m in self._gpu_metrics_history[gpu_id] 
                          if start_time <= m.timestamp <= end_time]
        
        if not gpu_history:
            return {"error": "No data available for specified period"}
        
        utilizations = [m.utilization_percent for m in gpu_history]
        memory_utils = [m.memory_utilization_percent for m in gpu_history]
        temperatures = [m.temperature_celsius for m in gpu_history]
        
        return {
            "gpu_id": gpu_id,
            "period_minutes": duration_minutes,
            "data_points": len(gpu_history),
            "utilization": {
                "current": utilizations[-1] if utilizations else 0,
                "average": statistics.mean(utilizations),
                "min": min(utilizations),
                "max": max(utilizations),
                "std_dev": statistics.stdev(utilizations) if len(utilizations) > 1 else 0
            },
            "memory_utilization": {
                "current": memory_utils[-1] if memory_utils else 0,
                "average": statistics.mean(memory_utils),
                "min": min(memory_utils),
                "max": max(memory_utils),
                "std_dev": statistics.stdev(memory_utils) if len(memory_utils) > 1 else 0
            },
            "temperature": {
                "current": temperatures[-1] if temperatures else 0,
                "average": statistics.mean(temperatures),
                "min": min(temperatures),
                "max": max(temperatures)
            }
        }
    
    def get_communication_matrix(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get communication pattern matrix between GPUs"""
        end_time = time.time()
        start_time = end_time - (duration_minutes * 60)
        
        with self._lock:
            comm_history = [m for m in self._communication_metrics 
                           if start_time <= m.timestamp <= end_time]
        
        # Initialize matrix
        matrix = {
            "communication_count": defaultdict(lambda: defaultdict(int)),
            "total_data_gb": defaultdict(lambda: defaultdict(float)),
            "average_latency_ms": defaultdict(lambda: defaultdict(list)),
            "average_bandwidth_gbps": defaultdict(lambda: defaultdict(list))
        }
        
        # Populate matrix
        for comm in comm_history:
            src, dst = comm.source_gpu, comm.target_gpu
            
            matrix["communication_count"][src][dst] += 1
            matrix["total_data_gb"][src][dst] += comm.data_size_bytes / (1024**3)
            matrix["average_latency_ms"][src][dst].append(comm.latency_ms)
            matrix["average_bandwidth_gbps"][src][dst].append(comm.bandwidth_gbps)
        
        # Calculate averages
        result = {
            "period_minutes": duration_minutes,
            "gpu_pairs": {},
            "summary": {
                "total_communications": len(comm_history),
                "total_data_transferred_gb": sum(comm.data_size_bytes / (1024**3) for comm in comm_history),
                "average_latency_ms": statistics.mean([comm.latency_ms for comm in comm_history]) if comm_history else 0,
                "average_bandwidth_gbps": statistics.mean([comm.bandwidth_gbps for comm in comm_history]) if comm_history else 0
            }
        }
        
        for src in matrix["communication_count"]:
            for dst in matrix["communication_count"][src]:
                pair_key = f"{src}->{dst}"
                latencies = matrix["average_latency_ms"][src][dst]
                bandwidths = matrix["average_bandwidth_gbps"][src][dst]
                
                result["gpu_pairs"][pair_key] = {
                    "communication_count": matrix["communication_count"][src][dst],
                    "total_data_gb": matrix["total_data_gb"][src][dst],
                    "average_latency_ms": statistics.mean(latencies) if latencies else 0,
                    "average_bandwidth_gbps": statistics.mean(bandwidths) if bandwidths else 0,
                    "max_latency_ms": max(latencies) if latencies else 0,
                    "max_bandwidth_gbps": max(bandwidths) if bandwidths else 0
                }
        
        return result
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of active performance alerts"""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get current optimization recommendations"""
        summary = self.get_performance_summary(duration_minutes=15)  # Last 15 minutes
        return summary.optimization_suggestions
    
    def export_performance_data(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Export performance data for external analysis"""
        end_time = time.time()
        start_time = end_time - (duration_minutes * 60)
        
        with self._lock:
            # Export GPU metrics
            gpu_data = {}
            for gpu_id in self.gpu_ids:
                gpu_history = [m.to_dict() for m in self._gpu_metrics_history[gpu_id] 
                              if start_time <= m.timestamp <= end_time]
                gpu_data[str(gpu_id)] = gpu_history
            
            # Export communication metrics
            comm_data = []
            for comm in self._communication_metrics:
                if start_time <= comm.timestamp <= end_time:
                    comm_data.append({
                        "timestamp": comm.timestamp,
                        "source_gpu": comm.source_gpu,
                        "target_gpu": comm.target_gpu,
                        "operation_type": comm.operation_type,
                        "data_size_bytes": comm.data_size_bytes,
                        "latency_ms": comm.latency_ms,
                        "bandwidth_gbps": comm.bandwidth_gbps,
                        "success": comm.success,
                        "error_message": comm.error_message
                    })
            
            # Export system metrics
            system_data = []
            for sys_metric in self._system_metrics_history:
                if start_time <= sys_metric.timestamp <= end_time:
                    system_data.append({
                        "timestamp": sys_metric.timestamp,
                        "cpu_utilization_percent": sys_metric.cpu_utilization_percent,
                        "memory_used_gb": sys_metric.memory_used_gb,
                        "memory_total_gb": sys_metric.memory_total_gb,
                        "disk_io_read_mbps": sys_metric.disk_io_read_mbps,
                        "disk_io_write_mbps": sys_metric.disk_io_write_mbps,
                        "network_io_recv_mbps": sys_metric.network_io_recv_mbps,
                        "network_io_sent_mbps": sys_metric.network_io_sent_mbps,
                        "process_count": sys_metric.process_count,
                        "load_average": sys_metric.load_average
                    })
            
            # Export alerts
            alert_data = []
            for alert in self._alert_history:
                if start_time <= alert.timestamp <= end_time:
                    alert_data.append({
                        "alert_id": alert.alert_id,
                        "timestamp": alert.timestamp,
                        "severity": alert.severity.value,
                        "metric_type": alert.metric_type.value,
                        "gpu_id": alert.gpu_id,
                        "title": alert.title,
                        "description": alert.description,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "recommendation": alert.recommendation
                    })
            
            return {
                "export_info": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": duration_minutes,
                    "gpu_count": len(self.gpu_ids),
                    "export_timestamp": time.time()
                },
                "gpu_metrics": gpu_data,
                "communication_metrics": comm_data,
                "system_metrics": system_data,
                "alerts": alert_data
            }
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        logger.info("Performance monitoring loop started")
        
        while self._running:
            try:
                start_time = time.time()
                
                # Collect GPU metrics
                self._collect_all_gpu_metrics()
                
                # Collect system metrics
                self._collect_system_metrics_periodic()
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_performance_alerts()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        logger.info("Performance monitoring loop ended")
    
    def _collect_all_gpu_metrics(self) -> None:
        """Collect metrics for all GPUs"""
        for gpu_id in self.gpu_ids:
            try:
                metrics = self._collect_gpu_metrics(gpu_id)
                if metrics:
                    with self._lock:
                        self._gpu_metrics_history[gpu_id].append(metrics)
            except Exception as e:
                logger.warning(f"Failed to collect metrics for GPU {gpu_id}: {e}")
    
    def _collect_gpu_metrics(self, gpu_id: int) -> Optional[GPUMetrics]:
        """Collect comprehensive metrics for a single GPU"""
        try:
            current_time = time.time()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                
                # Memory info
                memory_info = torch.cuda.memory_stats(gpu_id)
                memory_used = memory_info.get('allocated_bytes.all.current', 0) / (1024**3)
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                
                # Utilization (simplified - would use nvidia-ml-py in production)
                utilization = min(95.0, max(0.0, (memory_used / memory_total) * 100 + 
                                           (hash(str(current_time)) % 20 - 10)))  # Mock variation
                
                # Mock other metrics (would use nvidia-ml-py for real values)
                temperature = 65.0 + (hash(str(gpu_id + current_time)) % 20)
                power_draw = 200.0 + (hash(str(gpu_id * 2 + current_time)) % 50)
                clock_speed = 1500 + (hash(str(gpu_id * 3 + current_time)) % 200)
                fan_speed = 50.0 + (hash(str(gpu_id * 4 + current_time)) % 30)
                
                return GPUMetrics(
                    gpu_id=gpu_id,
                    timestamp=current_time,
                    utilization_percent=utilization,
                    memory_used_gb=memory_used,
                    memory_total_gb=memory_total,
                    memory_utilization_percent=(memory_used / memory_total) * 100,
                    temperature_celsius=temperature,
                    power_draw_watts=power_draw,
                    clock_speed_mhz=clock_speed,
                    fan_speed_percent=fan_speed,
                    error_count=0
                )
            else:
                # Mock metrics for testing
                return GPUMetrics(
                    gpu_id=gpu_id,
                    timestamp=current_time,
                    utilization_percent=75.0 + (hash(str(gpu_id + current_time)) % 20),
                    memory_used_gb=12.0 + (hash(str(gpu_id)) % 8),
                    memory_total_gb=24.0,
                    memory_utilization_percent=50.0 + (hash(str(gpu_id * 2)) % 40),
                    temperature_celsius=70.0 + (hash(str(gpu_id * 3)) % 15),
                    power_draw_watts=220.0 + (hash(str(gpu_id * 4)) % 60),
                    clock_speed_mhz=1600 + (hash(str(gpu_id * 5)) % 300),
                    fan_speed_percent=60.0 + (hash(str(gpu_id * 6)) % 25),
                    error_count=0
                )
                
        except Exception as e:
            logger.warning(f"Failed to collect GPU {gpu_id} metrics: {e}")
            return None
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect system-wide performance metrics"""
        try:
            current_time = time.time()
            
            if PSUTIL_AVAILABLE:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read_mbps = (disk_io.read_bytes / (1024**2)) if disk_io else 0
                disk_write_mbps = (disk_io.write_bytes / (1024**2)) if disk_io else 0
                
                # Network I/O
                net_io = psutil.net_io_counters()
                net_recv_mbps = (net_io.bytes_recv / (1024**2)) if net_io else 0
                net_sent_mbps = (net_io.bytes_sent / (1024**2)) if net_io else 0
                
                # Process count
                process_count = len(psutil.pids())
                
                # Load average
                load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0.0, 0.0, 0.0)
                
                return SystemMetrics(
                    timestamp=current_time,
                    cpu_utilization_percent=cpu_percent,
                    memory_used_gb=memory.used / (1024**3),
                    memory_total_gb=memory.total / (1024**3),
                    disk_io_read_mbps=disk_read_mbps,
                    disk_io_write_mbps=disk_write_mbps,
                    network_io_recv_mbps=net_recv_mbps,
                    network_io_sent_mbps=net_sent_mbps,
                    process_count=process_count,
                    load_average=load_avg
                )
            else:
                # Mock system metrics
                return SystemMetrics(
                    timestamp=current_time,
                    cpu_utilization_percent=45.0 + (hash(str(current_time)) % 30),
                    memory_used_gb=32.0 + (hash(str(current_time * 2)) % 16),
                    memory_total_gb=64.0,
                    disk_io_read_mbps=50.0 + (hash(str(current_time * 3)) % 100),
                    disk_io_write_mbps=30.0 + (hash(str(current_time * 4)) % 70),
                    network_io_recv_mbps=100.0 + (hash(str(current_time * 5)) % 200),
                    network_io_sent_mbps=80.0 + (hash(str(current_time * 6)) % 150),
                    process_count=150 + (hash(str(current_time * 7)) % 50),
                    load_average=(1.5, 1.8, 2.1)
                )
                
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return None
    
    def _collect_system_metrics_periodic(self) -> None:
        """Collect system metrics periodically"""
        metrics = self._collect_system_metrics()
        if metrics:
            with self._lock:
                self._system_metrics_history.append(metrics)
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds for different metrics"""
        return {
            "gpu_utilization": {"warning": 90.0, "critical": 95.0},
            "memory_utilization": {"warning": 85.0, "critical": 95.0},
            "temperature": {"warning": 80.0, "critical": 90.0},
            "power_consumption": {"warning": 300.0, "critical": 350.0},
            "communication_latency": {"warning": 50.0, "critical": 100.0},
            "error_rate": {"warning": 0.01, "critical": 0.05}
        }
    
    def _check_performance_alerts(self) -> None:
        """Check for performance alerts based on current metrics"""
        current_time = time.time()
        
        # Check GPU metrics
        for gpu_id in self.gpu_ids:
            if self._gpu_metrics_history[gpu_id]:
                latest_metric = self._gpu_metrics_history[gpu_id][-1]
                self._check_gpu_alerts(latest_metric)
        
        # Check system metrics
        if self._system_metrics_history:
            latest_system = self._system_metrics_history[-1]
            self._check_system_alerts(latest_system)
    
    def _check_gpu_alerts(self, metrics: GPUMetrics) -> None:
        """Check for GPU-specific alerts"""
        gpu_id = metrics.gpu_id
        
        # GPU utilization alert
        if metrics.utilization_percent >= self._alert_thresholds["gpu_utilization"]["critical"]:
            self._create_alert(
                metric_type=MetricType.GPU_UTILIZATION,
                severity=AlertSeverity.CRITICAL,
                gpu_id=gpu_id,
                title=f"Critical GPU {gpu_id} Utilization",
                description=f"GPU {gpu_id} utilization at {metrics.utilization_percent:.1f}%",
                current_value=metrics.utilization_percent,
                threshold_value=self._alert_thresholds["gpu_utilization"]["critical"],
                recommendation="Consider load balancing or reducing concurrent tasks"
            )
        
        # Memory utilization alert
        if metrics.memory_utilization_percent >= self._alert_thresholds["memory_utilization"]["critical"]:
            self._create_alert(
                metric_type=MetricType.MEMORY_USAGE,
                severity=AlertSeverity.CRITICAL,
                gpu_id=gpu_id,
                title=f"Critical GPU {gpu_id} Memory Usage",
                description=f"GPU {gpu_id} memory usage at {metrics.memory_utilization_percent:.1f}%",
                current_value=metrics.memory_utilization_percent,
                threshold_value=self._alert_thresholds["memory_utilization"]["critical"],
                recommendation="Free GPU memory or use model parallelism"
            )
        
        # Temperature alert
        if metrics.temperature_celsius >= self._alert_thresholds["temperature"]["critical"]:
            self._create_alert(
                metric_type=MetricType.TEMPERATURE,
                severity=AlertSeverity.CRITICAL,
                gpu_id=gpu_id,
                title=f"Critical GPU {gpu_id} Temperature",
                description=f"GPU {gpu_id} temperature at {metrics.temperature_celsius:.1f}°C",
                current_value=metrics.temperature_celsius,
                threshold_value=self._alert_thresholds["temperature"]["critical"],
                recommendation="Check cooling system and reduce workload if necessary"
            )
    
    def _check_system_alerts(self, metrics: SystemMetrics) -> None:
        """Check for system-wide alerts"""
        # CPU utilization
        if metrics.cpu_utilization_percent >= 90.0:
            self._create_alert(
                metric_type=MetricType.GPU_UTILIZATION,  # Using GPU_UTILIZATION as placeholder
                severity=AlertSeverity.WARNING,
                gpu_id=None,
                title="High CPU Utilization",
                description=f"System CPU utilization at {metrics.cpu_utilization_percent:.1f}%",
                current_value=metrics.cpu_utilization_percent,
                threshold_value=90.0,
                recommendation="Monitor CPU-intensive processes and consider optimization"
            )
        
        # Memory utilization
        memory_percent = (metrics.memory_used_gb / metrics.memory_total_gb) * 100
        if memory_percent >= 90.0:
            self._create_alert(
                metric_type=MetricType.MEMORY_USAGE,
                severity=AlertSeverity.WARNING,
                gpu_id=None,
                title="High System Memory Usage",
                description=f"System memory usage at {memory_percent:.1f}%",
                current_value=memory_percent,
                threshold_value=90.0,
                recommendation="Free system memory or increase available RAM"
            )
    
    def _check_communication_alerts(self, comm_metric: CommunicationMetrics) -> None:
        """Check for communication performance alerts"""
        if comm_metric.latency_ms >= self._alert_thresholds["communication_latency"]["critical"]:
            self._create_alert(
                metric_type=MetricType.COMMUNICATION_LATENCY,
                severity=AlertSeverity.CRITICAL,
                gpu_id=comm_metric.source_gpu,
                title=f"High Communication Latency",
                description=f"Communication latency {comm_metric.latency_ms:.1f}ms between GPU {comm_metric.source_gpu} and {comm_metric.target_gpu}",
                current_value=comm_metric.latency_ms,
                threshold_value=self._alert_thresholds["communication_latency"]["critical"],
                recommendation="Check inter-GPU bandwidth and optimize communication patterns"
            )
    
    def _create_alert(self, 
                     metric_type: MetricType,
                     severity: AlertSeverity,
                     title: str,
                     description: str,
                     current_value: float,
                     threshold_value: float,
                     recommendation: str,
                     gpu_id: Optional[int] = None) -> None:
        """Create a new performance alert"""
        
        alert_id = str(uuid.uuid4())
        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            metric_type=metric_type,
            gpu_id=gpu_id,
            title=title,
            description=description,
            current_value=current_value,
            threshold_value=threshold_value,
            recommendation=recommendation
        )
        
        with self._lock:
            # Add to active alerts (replace if similar alert exists)
            alert_key = f"{metric_type.value}_{gpu_id}_{severity.value}"
            self._active_alerts[alert_key] = alert
            self._alert_history.append(alert)
        
        logger.warning(f"Performance alert: {title} - {description}")
    
    def _analyze_communication_metrics(self, comm_metrics: List[CommunicationMetrics]) -> Dict[str, Any]:
        """Analyze communication metrics"""
        if not comm_metrics:
            return {
                "total_operations": 0,
                "average_latency_ms": 0.0,
                "average_bandwidth_gbps": 0.0,
                "success_rate": 1.0
            }
        
        successful = [m for m in comm_metrics if m.success]
        
        return {
            "total_operations": len(comm_metrics),
            "successful_operations": len(successful),
            "success_rate": len(successful) / len(comm_metrics),
            "average_latency_ms": statistics.mean([m.latency_ms for m in successful]) if successful else 0.0,
            "max_latency_ms": max([m.latency_ms for m in successful]) if successful else 0.0,
            "average_bandwidth_gbps": statistics.mean([m.bandwidth_gbps for m in successful]) if successful else 0.0,
            "max_bandwidth_gbps": max([m.bandwidth_gbps for m in successful]) if successful else 0.0,
            "total_data_transferred_gb": sum([m.data_size_bytes for m in comm_metrics]) / (1024**3),
            "operation_types": list(set([m.operation_type for m in comm_metrics]))
        }
    
    def _calculate_gpu_statistics(self, gpu_metrics: List[GPUMetrics]) -> Dict[str, Any]:
        """Calculate statistics for GPU metrics"""
        if not gpu_metrics:
            return {}
        
        utilizations = [m.utilization_percent for m in gpu_metrics]
        memory_utils = [m.memory_utilization_percent for m in gpu_metrics]
        temperatures = [m.temperature_celsius for m in gpu_metrics]
        power_draws = [m.power_draw_watts for m in gpu_metrics]
        
        return {
            "data_points": len(gpu_metrics),
            "utilization": {
                "average": statistics.mean(utilizations),
                "min": min(utilizations),
                "max": max(utilizations),
                "std_dev": statistics.stdev(utilizations) if len(utilizations) > 1 else 0
            },
            "memory_utilization": {
                "average": statistics.mean(memory_utils),
                "min": min(memory_utils),
                "max": max(memory_utils),
                "std_dev": statistics.stdev(memory_utils) if len(memory_utils) > 1 else 0
            },
            "temperature": {
                "average": statistics.mean(temperatures),
                "min": min(temperatures),
                "max": max(temperatures)
            },
            "power_consumption": {
                "average": statistics.mean(power_draws),
                "min": min(power_draws),
                "max": max(power_draws)
            }
        }
    
    def _calculate_system_statistics(self, system_metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Calculate statistics for system metrics"""
        if not system_metrics:
            return {}
        
        cpu_utils = [m.cpu_utilization_percent for m in system_metrics]
        memory_utils = [(m.memory_used_gb / m.memory_total_gb) * 100 for m in system_metrics]
        
        return {
            "data_points": len(system_metrics),
            "cpu_utilization": {
                "average": statistics.mean(cpu_utils),
                "min": min(cpu_utils),
                "max": max(cpu_utils)
            },
            "memory_utilization": {
                "average": statistics.mean(memory_utils),
                "min": min(memory_utils),
                "max": max(memory_utils)
            }
        }
    
    def _detect_bottlenecks(self, gpu_stats: Dict[int, Dict], comm_stats: Dict, system_stats: Dict) -> List[str]:
        """Detect performance bottlenecks"""
        bottlenecks = []
        
        # GPU bottlenecks
        for gpu_id, stats in gpu_stats.items():
            if stats.get("utilization", {}).get("average", 0) > 90:
                bottlenecks.append(f"GPU {gpu_id} high utilization")
            
            if stats.get("memory_utilization", {}).get("average", 0) > 85:
                bottlenecks.append(f"GPU {gpu_id} high memory usage")
        
        # Communication bottlenecks
        if comm_stats.get("average_latency_ms", 0) > 20:
            bottlenecks.append("High inter-GPU communication latency")
        
        if comm_stats.get("success_rate", 1.0) < 0.95:
            bottlenecks.append("Poor communication reliability")
        
        # System bottlenecks
        if system_stats.get("cpu_utilization", {}).get("average", 0) > 80:
            bottlenecks.append("High CPU utilization")
        
        if system_stats.get("memory_utilization", {}).get("average", 0) > 85:
            bottlenecks.append("High system memory usage")
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, gpu_stats: Dict, comm_stats: Dict, bottlenecks: List[str]) -> List[str]:
        """Generate optimization suggestions based on performance data"""
        suggestions = []
        
        if "high utilization" in " ".join(bottlenecks):
            suggestions.append("Consider reducing batch size or implementing dynamic batching")
            suggestions.append("Enable gradient checkpointing to reduce memory usage")
        
        if "high memory usage" in " ".join(bottlenecks):
            suggestions.append("Use model parallelism to distribute memory load")
            suggestions.append("Enable automatic model offloading")
        
        if "communication latency" in " ".join(bottlenecks):
            suggestions.append("Optimize communication patterns and reduce synchronization points")
            suggestions.append("Use more efficient collective communication operations")
        
        if "CPU utilization" in " ".join(bottlenecks):
            suggestions.append("Optimize data preprocessing and loading")
            suggestions.append("Use more efficient data formats")
        
        if not suggestions:
            suggestions.append("Performance is optimal - no immediate optimizations needed")
        
        return suggestions
    
    def _cleanup_old_data(self) -> None:
        """Clean up old performance data to manage memory"""
        # This is automatically handled by deque maxlen, but we could add more cleanup here
        pass
    
    def cleanup(self) -> None:
        """Cleanup monitor resources"""
        logger.info("Cleaning up performance monitor")
        
        self.stop()
        
        with self._lock:
            self._gpu_metrics_history.clear()
            self._communication_metrics.clear()
            self._system_metrics_history.clear()
            self._active_alerts.clear()
            self._alert_history.clear()
        
        logger.info("Performance monitor cleanup completed")