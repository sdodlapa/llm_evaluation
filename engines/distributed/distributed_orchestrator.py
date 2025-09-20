"""
Distributed Evaluation Orchestrator

This module manages workload distribution, GPU allocation, and coordinated 
evaluation across multiple GPUs for large models in the distributed engine.
Handles load balancing, fault tolerance, and efficient resource utilization.
"""

import time
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, PriorityQueue
import uuid

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import project modules
from core_shared.interfaces.evaluation_interfaces import EvaluationRequest, EvaluationResult, EngineType
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
from .multi_gpu_model_loader import MultiGPUModelLoader, DistributedModelInfo, DistributionStrategy

logger = logging.getLogger(__name__)

class WorkloadPriority(Enum):
    """Priority levels for evaluation workloads"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class GPUClusterState(Enum):
    """States of the GPU cluster"""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAULT = "fault"

@dataclass
class WorkloadTask:
    """Individual workload task for evaluation"""
    task_id: str
    request: EvaluationRequest
    model_info: DistributedModelInfo
    priority: WorkloadPriority
    created_at: float
    estimated_duration: float
    gpu_requirements: List[int]
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        # Priority queue comparison (lower number = higher priority)
        return self.priority.value < other.priority.value

@dataclass
class GPUAllocationPlan:
    """Plan for allocating GPUs to evaluation tasks"""
    task_id: str
    gpu_ids: List[int]
    memory_allocation: Dict[int, float]  # GPU ID -> memory in GB
    estimated_start_time: float
    estimated_completion_time: float
    communication_group: Optional[str] = None

@dataclass
class ClusterMetrics:
    """Performance metrics for the GPU cluster"""
    timestamp: float
    total_gpus: int
    active_gpus: int
    idle_gpus: int
    fault_gpus: int
    total_memory_gb: float
    used_memory_gb: float
    average_utilization: float
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    
class DistributedEvaluationOrchestrator:
    """
    Advanced orchestrator for managing distributed evaluation workloads.
    
    Features:
    - Multi-GPU workload scheduling and load balancing
    - Dynamic resource allocation and optimization
    - Fault tolerance and recovery mechanisms
    - Performance monitoring and optimization
    - Priority-based task scheduling
    - Cross-GPU communication coordination
    """
    
    def __init__(self, 
                 model_loader: Optional[MultiGPUModelLoader] = None,
                 max_concurrent_tasks: int = 4,
                 enable_fault_tolerance: bool = True,
                 scheduling_strategy: str = "priority_first"):
        """
        Initialize distributed evaluation orchestrator
        
        Args:
            model_loader: Multi-GPU model loader instance
            max_concurrent_tasks: Maximum number of concurrent evaluation tasks
            enable_fault_tolerance: Enable fault tolerance and recovery
            scheduling_strategy: Task scheduling strategy ("priority_first", "round_robin", "load_balanced")
        """
        
        # Core components
        self.model_loader = model_loader or MultiGPUModelLoader()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_fault_tolerance = enable_fault_tolerance
        self.scheduling_strategy = scheduling_strategy
        
        # State management
        self._task_queue = PriorityQueue()
        self._running_tasks: Dict[str, WorkloadTask] = {}
        self._completed_tasks: Dict[str, EvaluationResult] = {}
        self._failed_tasks: Dict[str, str] = {}  # task_id -> error message
        
        # Resource management
        self._gpu_allocations: Dict[int, Optional[str]] = {}  # GPU ID -> task_id
        self._allocation_plans: Dict[str, GPUAllocationPlan] = {}
        self._cluster_state = GPUClusterState.IDLE
        
        # Threading and async support
        self._orchestrator_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self._running = False
        self._lock = threading.Lock()
        
        # Metrics and monitoring
        self._metrics_history: List[ClusterMetrics] = []
        self._last_metrics_update = 0.0
        
        # Initialize GPU allocations
        self._initialize_gpu_tracking()
        
        logger.info(f"Initialized distributed orchestrator with {len(self.model_loader.gpu_ids)} GPUs")
        logger.info(f"Max concurrent tasks: {max_concurrent_tasks}, Strategy: {scheduling_strategy}")
    
    def _initialize_gpu_tracking(self) -> None:
        """Initialize GPU allocation tracking"""
        for gpu_id in self.model_loader.gpu_ids:
            self._gpu_allocations[gpu_id] = None
        
        logger.info(f"Tracking {len(self._gpu_allocations)} GPUs: {list(self._gpu_allocations.keys())}")
    
    def start(self) -> None:
        """Start the distributed orchestrator"""
        if self._running:
            logger.warning("Orchestrator is already running")
            return
        
        self._running = True
        self._orchestrator_thread = threading.Thread(target=self._orchestrator_loop, daemon=True)
        self._orchestrator_thread.start()
        
        logger.info("Distributed orchestrator started")
    
    def stop(self) -> None:
        """Stop the distributed orchestrator"""
        if not self._running:
            return
        
        self._running = False
        
        if self._orchestrator_thread:
            self._orchestrator_thread.join(timeout=10.0)
        
        # Cancel running tasks
        for task_id in list(self._running_tasks.keys()):
            self._cancel_task(task_id)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("Distributed orchestrator stopped")
    
    def submit_evaluation_request(self, 
                                request: EvaluationRequest,
                                priority: WorkloadPriority = WorkloadPriority.NORMAL,
                                callback: Optional[Callable] = None) -> str:
        """
        Submit an evaluation request for distributed processing
        
        Args:
            request: Evaluation request
            priority: Task priority level
            callback: Optional callback function for completion
            
        Returns:
            Task ID for tracking
        """
        
        task_id = str(uuid.uuid4())
        
        try:
            # Validate request
            if not self._validate_request(request):
                raise ValueError("Invalid evaluation request")
            
            # Check if model can be handled
            model_config = request.model_config
            if not self.model_loader.can_load_model(model_config):
                raise RuntimeError(f"Cannot handle model {model_config.model_name} with available resources")
            
            # Load model if not already loaded
            model_name = model_config.model_name
            model_info = self.model_loader.get_model_info(model_name)
            
            if model_info is None:
                logger.info(f"Loading model {model_name} for evaluation")
                model_info = self.model_loader.load_model_distributed(model_config)
            
            # Estimate task requirements
            estimated_duration = self._estimate_task_duration(request, model_info)
            gpu_requirements = self._determine_gpu_requirements(model_info)
            
            # Create task
            task = WorkloadTask(
                task_id=task_id,
                request=request,
                model_info=model_info,
                priority=priority,
                created_at=time.time(),
                estimated_duration=estimated_duration,
                gpu_requirements=gpu_requirements,
                callback=callback
            )
            
            # Add to queue
            self._task_queue.put(task)
            
            logger.info(f"Submitted evaluation task {task_id} for model {model_name}")
            logger.info(f"Priority: {priority.name}, Estimated duration: {estimated_duration:.1f}s")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit evaluation request: {e}")
            self._failed_tasks[task_id] = str(e)
            raise
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        with self._lock:
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                allocation = self._allocation_plans.get(task_id)
                
                return {
                    "status": "running",
                    "progress": self._estimate_task_progress(task_id),
                    "start_time": allocation.estimated_start_time if allocation else None,
                    "estimated_completion": allocation.estimated_completion_time if allocation else None,
                    "gpu_allocation": allocation.gpu_ids if allocation else [],
                    "memory_usage": allocation.memory_allocation if allocation else {}
                }
            
            elif task_id in self._completed_tasks:
                return {
                    "status": "completed",
                    "result": self._completed_tasks[task_id],
                    "progress": 100.0
                }
            
            elif task_id in self._failed_tasks:
                return {
                    "status": "failed",
                    "error": self._failed_tasks[task_id],
                    "progress": 0.0
                }
            
            else:
                # Check if in queue
                queue_items = list(self._task_queue.queue)
                for task in queue_items:
                    if task.task_id == task_id:
                        return {
                            "status": "queued",
                            "queue_position": queue_items.index(task) + 1,
                            "progress": 0.0
                        }
                
                return {"status": "not_found"}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status and metrics"""
        with self._lock:
            current_metrics = self._collect_cluster_metrics()
            
            return {
                "cluster_state": self._cluster_state.value,
                "metrics": current_metrics,
                "active_tasks": len(self._running_tasks),
                "queued_tasks": self._task_queue.qsize(),
                "gpu_allocations": {gpu_id: task_id for gpu_id, task_id in self._gpu_allocations.items()},
                "model_loader_status": self.model_loader.get_gpu_status()
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        return self._cancel_task(task_id)
    
    def _orchestrator_loop(self) -> None:
        """Main orchestrator loop for task scheduling and management"""
        logger.info("Orchestrator loop started")
        
        while self._running:
            try:
                # Update cluster metrics
                self._update_cluster_metrics()
                
                # Process task queue
                self._process_task_queue()
                
                # Monitor running tasks
                self._monitor_running_tasks()
                
                # Perform maintenance
                self._perform_maintenance()
                
                # Sleep for a short interval
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")
                time.sleep(5.0)  # Longer sleep on error
        
        logger.info("Orchestrator loop ended")
    
    def _process_task_queue(self) -> None:
        """Process queued tasks and allocate resources"""
        while not self._task_queue.empty() and len(self._running_tasks) < self.max_concurrent_tasks:
            try:
                task = self._task_queue.get_nowait()
                
                # Create allocation plan
                allocation_plan = self._create_allocation_plan(task)
                
                if allocation_plan is None:
                    # No resources available, put back in queue
                    self._task_queue.put(task)
                    break
                
                # Start task execution
                self._start_task_execution(task, allocation_plan)
                
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
                break
    
    def _create_allocation_plan(self, task: WorkloadTask) -> Optional[GPUAllocationPlan]:
        """Create GPU allocation plan for a task"""
        required_gpus = len(task.gpu_requirements)
        
        # Find available GPUs
        available_gpus = [gpu_id for gpu_id, allocated_task in self._gpu_allocations.items() 
                         if allocated_task is None]
        
        if len(available_gpus) < required_gpus:
            return None  # Not enough GPUs available
        
        # Select GPUs based on strategy
        selected_gpus = self._select_optimal_gpus(available_gpus, required_gpus, task)
        
        # Calculate memory allocation
        memory_allocation = {}
        model_memory = task.model_info.memory_usage_gb
        memory_per_gpu = model_memory / len(selected_gpus)
        
        for gpu_id in selected_gpus:
            memory_allocation[gpu_id] = memory_per_gpu
        
        return GPUAllocationPlan(
            task_id=task.task_id,
            gpu_ids=selected_gpus,
            memory_allocation=memory_allocation,
            estimated_start_time=time.time(),
            estimated_completion_time=time.time() + task.estimated_duration,
            communication_group=f"task_{task.task_id}"
        )
    
    def _select_optimal_gpus(self, available_gpus: List[int], required_count: int, task: WorkloadTask) -> List[int]:
        """Select optimal GPUs for a task based on strategy"""
        if self.scheduling_strategy == "round_robin":
            return available_gpus[:required_count]
        
        elif self.scheduling_strategy == "load_balanced":
            # Sort by utilization (prefer less utilized GPUs)
            gpu_utilization = [(gpu_id, self.model_loader._get_gpu_utilization(gpu_id)) 
                              for gpu_id in available_gpus]
            gpu_utilization.sort(key=lambda x: x[1])
            return [gpu_id for gpu_id, _ in gpu_utilization[:required_count]]
        
        else:  # priority_first or default
            # Use consecutive GPUs for better communication
            return available_gpus[:required_count]
    
    def _start_task_execution(self, task: WorkloadTask, allocation_plan: GPUAllocationPlan) -> None:
        """Start executing a task with allocated resources"""
        with self._lock:
            # Allocate GPUs
            for gpu_id in allocation_plan.gpu_ids:
                self._gpu_allocations[gpu_id] = task.task_id
            
            # Store allocation plan
            self._allocation_plans[task.task_id] = allocation_plan
            
            # Add to running tasks
            self._running_tasks[task.task_id] = task
        
        # Submit task to executor
        future = self._executor.submit(self._execute_task, task, allocation_plan)
        future.add_done_callback(lambda f: self._task_completed(task.task_id, f))
        
        logger.info(f"Started task {task.task_id} on GPUs {allocation_plan.gpu_ids}")
    
    def _execute_task(self, task: WorkloadTask, allocation_plan: GPUAllocationPlan) -> EvaluationResult:
        """Execute an evaluation task"""
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task.task_id} for model {task.model_info.model_name}")
            
            # Set up distributed environment if needed
            self._setup_distributed_environment(allocation_plan)
            
            # Perform distributed evaluation
            result = self._perform_distributed_evaluation(task, allocation_plan)
            
            # Update result with execution info
            result.execution_time_seconds = time.time() - start_time
            result.engine_used = EngineType.DISTRIBUTED
            result.completed_at = time.time()
            result.success = True
            
            logger.info(f"Task {task.task_id} completed successfully in {result.execution_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Create error result
            error_result = EvaluationResult(
                request_id=task.request.request_id,
                model_name=task.model_info.model_name,
                dataset_name=",".join(task.request.datasets),
                metrics={},
                engine_used=EngineType.DISTRIBUTED,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=time.time(),
                execution_time_seconds=time.time() - start_time
            )
            
            raise Exception(f"Task execution failed: {e}")
    
    def _perform_distributed_evaluation(self, task: WorkloadTask, allocation_plan: GPUAllocationPlan) -> EvaluationResult:
        """Perform the actual distributed evaluation"""
        
        # This is a simplified implementation
        # In production, this would coordinate across GPUs for actual model inference
        
        request = task.request
        model_info = task.model_info
        
        # Simulate distributed evaluation
        logger.info(f"Simulating distributed evaluation for {len(request.datasets)} datasets")
        
        # Mock evaluation metrics
        evaluation_metrics = {
            "accuracy": 0.87,
            "f1_score": 0.85,
            "perplexity": 3.2,
            "tokens_per_second": 120.0 * len(allocation_plan.gpu_ids),  # Scale with GPU count
            "memory_efficiency": 0.92,
            "gpu_utilization": 0.88,
            "cross_gpu_communication_ms": len(allocation_plan.gpu_ids) * 2.5
        }
        
        # Simulate processing time (shorter for more GPUs due to parallelism)
        base_time = 5.0
        parallel_efficiency = 0.8  # 80% parallel efficiency
        actual_time = base_time / (len(allocation_plan.gpu_ids) * parallel_efficiency)
        time.sleep(min(actual_time, 10.0))  # Cap at 10 seconds for testing
        
        # Create result
        result = EvaluationResult(
            request_id=request.request_id,
            model_name=model_info.model_name,
            dataset_name=",".join(request.datasets),
            metrics=evaluation_metrics,
            raw_outputs=[f"Distributed output {i}" for i in range(10)],
            performance_data={
                "gpu_count": len(allocation_plan.gpu_ids),
                "memory_per_gpu_gb": max(allocation_plan.memory_allocation.values()),
                "distribution_strategy": model_info.distribution_strategy.value,
                "communication_backend": model_info.communication_backend
            },
            resource_usage={
                "total_gpu_memory_gb": sum(allocation_plan.memory_allocation.values()),
                "average_gpu_utilization": evaluation_metrics["gpu_utilization"],
                "cross_gpu_bandwidth_gbps": 25.0 * len(allocation_plan.gpu_ids)
            },
            tokens_processed=1000,
            tokens_per_second=evaluation_metrics["tokens_per_second"],
            started_at=time.time() - actual_time
        )
        
        return result
    
    def _setup_distributed_environment(self, allocation_plan: GPUAllocationPlan) -> None:
        """Setup distributed computing environment for the task"""
        # This would set up distributed communication in production
        # For now, just log the setup
        logger.info(f"Setting up distributed environment for task {allocation_plan.task_id}")
        logger.info(f"Communication group: {allocation_plan.communication_group}")
        logger.info(f"GPU allocation: {allocation_plan.gpu_ids}")
    
    def _task_completed(self, task_id: str, future: Future) -> None:
        """Handle task completion"""
        with self._lock:
            if task_id not in self._running_tasks:
                return
            
            task = self._running_tasks[task_id]
            allocation_plan = self._allocation_plans[task_id]
            
            # Free allocated GPUs
            for gpu_id in allocation_plan.gpu_ids:
                self._gpu_allocations[gpu_id] = None
            
            # Remove from running tasks
            del self._running_tasks[task_id]
            del self._allocation_plans[task_id]
        
        try:
            # Get result
            result = future.result()
            self._completed_tasks[task_id] = result
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    logger.error(f"Callback failed for task {task_id}: {e}")
            
            logger.info(f"Task {task_id} completed and resources freed")
            
        except Exception as e:
            self._failed_tasks[task_id] = str(e)
            logger.error(f"Task {task_id} failed: {e}")
    
    def _monitor_running_tasks(self) -> None:
        """Monitor running tasks for health and progress"""
        current_time = time.time()
        
        with self._lock:
            for task_id, task in list(self._running_tasks.items()):
                allocation_plan = self._allocation_plans[task_id]
                
                # Check for timeout
                if current_time > allocation_plan.estimated_completion_time + 300:  # 5 minute grace period
                    logger.warning(f"Task {task_id} has exceeded expected completion time")
                    # Could implement timeout handling here
                
                # Check GPU health
                for gpu_id in allocation_plan.gpu_ids:
                    if not self._check_gpu_health(gpu_id):
                        logger.error(f"GPU {gpu_id} health check failed for task {task_id}")
                        # Could implement fault recovery here
    
    def _check_gpu_health(self, gpu_id: int) -> bool:
        """Check health of a specific GPU"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                # Simple health check - ensure GPU is responsive
                torch.cuda.synchronize()
                return True
            else:
                return True  # Assume healthy in non-CUDA environments
        except Exception as e:
            logger.warning(f"GPU {gpu_id} health check failed: {e}")
            return False
    
    def _estimate_task_progress(self, task_id: str) -> float:
        """Estimate progress of a running task"""
        if task_id not in self._running_tasks:
            return 0.0
        
        task = self._running_tasks[task_id]
        allocation_plan = self._allocation_plans[task_id]
        
        current_time = time.time()
        elapsed_time = current_time - allocation_plan.estimated_start_time
        
        if task.estimated_duration <= 0:
            return 50.0  # Unknown progress
        
        progress = (elapsed_time / task.estimated_duration) * 100.0
        return min(progress, 95.0)  # Cap at 95% until completion
    
    def _collect_cluster_metrics(self) -> ClusterMetrics:
        """Collect current cluster performance metrics"""
        current_time = time.time()
        
        # Count GPU states
        active_gpus = sum(1 for task_id in self._gpu_allocations.values() if task_id is not None)
        idle_gpus = len(self._gpu_allocations) - active_gpus
        fault_gpus = 0  # Would implement fault detection
        
        # Calculate memory usage
        total_memory = 0.0
        used_memory = 0.0
        total_utilization = 0.0
        
        for gpu_id in self.model_loader.gpu_ids:
            try:
                available_mem = self.model_loader._get_gpu_available_memory(gpu_id)
                utilization = self.model_loader._get_gpu_utilization(gpu_id)
                
                # Estimate total memory (assuming some is always used)
                estimated_total = available_mem / (1.0 - max(utilization / 100.0, 0.1))
                total_memory += estimated_total
                used_memory += estimated_total - available_mem
                total_utilization += utilization
                
            except Exception:
                continue
        
        average_utilization = total_utilization / len(self.model_loader.gpu_ids) if self.model_loader.gpu_ids else 0.0
        
        return ClusterMetrics(
            timestamp=current_time,
            total_gpus=len(self._gpu_allocations),
            active_gpus=active_gpus,
            idle_gpus=idle_gpus,
            fault_gpus=fault_gpus,
            total_memory_gb=total_memory,
            used_memory_gb=used_memory,
            average_utilization=average_utilization,
            pending_tasks=self._task_queue.qsize(),
            running_tasks=len(self._running_tasks),
            completed_tasks=len(self._completed_tasks),
            failed_tasks=len(self._failed_tasks)
        )
    
    def _update_cluster_metrics(self) -> None:
        """Update cluster metrics and determine cluster state"""
        current_time = time.time()
        
        # Update metrics every 30 seconds
        if current_time - self._last_metrics_update > 30.0:
            metrics = self._collect_cluster_metrics()
            self._metrics_history.append(metrics)
            
            # Keep only recent metrics (last 100 data points)
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]
            
            # Update cluster state
            self._update_cluster_state(metrics)
            
            self._last_metrics_update = current_time
    
    def _update_cluster_state(self, metrics: ClusterMetrics) -> None:
        """Update cluster state based on current metrics"""
        utilization = metrics.average_utilization
        
        if metrics.fault_gpus > 0:
            self._cluster_state = GPUClusterState.FAULT
        elif utilization > 90.0:
            self._cluster_state = GPUClusterState.OVERLOADED
        elif metrics.running_tasks > 0:
            self._cluster_state = GPUClusterState.BUSY
        else:
            self._cluster_state = GPUClusterState.IDLE
    
    def _perform_maintenance(self) -> None:
        """Perform periodic maintenance tasks"""
        current_time = time.time()
        
        # Clean up old completed/failed tasks (older than 1 hour)
        cutoff_time = current_time - 3600
        
        for task_id in list(self._completed_tasks.keys()):
            if hasattr(self._completed_tasks[task_id], 'completed_at') and \
               self._completed_tasks[task_id].completed_at < cutoff_time:
                del self._completed_tasks[task_id]
        
        for task_id in list(self._failed_tasks.keys()):
            # Failed tasks don't have completion time, so use current approach
            # This would be improved with proper timestamp tracking
            if len(self._failed_tasks) > 100:  # Keep only recent failures
                oldest_failed = list(self._failed_tasks.keys())[0]
                del self._failed_tasks[oldest_failed]
    
    def _validate_request(self, request: EvaluationRequest) -> bool:
        """Validate an evaluation request"""
        if not hasattr(request, 'model_config') or request.model_config is None:
            return False
        
        if not hasattr(request, 'datasets') or not request.datasets:
            return False
        
        return True
    
    def _estimate_task_duration(self, request: EvaluationRequest, model_info: DistributedModelInfo) -> float:
        """Estimate task duration based on request and model"""
        base_duration = 60.0  # Base 1 minute
        
        # Adjust for number of datasets
        dataset_factor = len(request.datasets)
        
        # Adjust for model size (larger models take longer)
        size_factor = model_info.memory_usage_gb / 30.0  # Normalize to 30GB
        
        # Adjust for distribution strategy (more parallelism = faster)
        if model_info.distribution_strategy == DistributionStrategy.TENSOR_PARALLEL:
            parallel_factor = 0.7  # 30% faster with tensor parallelism
        elif model_info.distribution_strategy == DistributionStrategy.HYBRID:
            parallel_factor = 0.5  # 50% faster with hybrid parallelism
        else:
            parallel_factor = 1.0
        
        estimated_duration = base_duration * dataset_factor * size_factor * parallel_factor
        
        return max(estimated_duration, 30.0)  # Minimum 30 seconds
    
    def _determine_gpu_requirements(self, model_info: DistributedModelInfo) -> List[int]:
        """Determine GPU requirements for a model"""
        return [allocation.gpu_id for allocation in model_info.gpu_allocations]
    
    def _cancel_task(self, task_id: str) -> bool:
        """Internal method to cancel a task"""
        with self._lock:
            # Check if task is running
            if task_id in self._running_tasks:
                # For running tasks, we would need to implement proper cancellation
                # For now, just mark as failed
                self._failed_tasks[task_id] = "Task cancelled by user"
                
                # Free resources
                if task_id in self._allocation_plans:
                    allocation_plan = self._allocation_plans[task_id]
                    for gpu_id in allocation_plan.gpu_ids:
                        self._gpu_allocations[gpu_id] = None
                    del self._allocation_plans[task_id]
                
                del self._running_tasks[task_id]
                
                logger.info(f"Cancelled running task {task_id}")
                return True
            
            # Check if task is queued
            queue_items = []
            found_and_removed = False
            
            while not self._task_queue.empty():
                task = self._task_queue.get_nowait()
                if task.task_id == task_id:
                    found_and_removed = True
                    self._failed_tasks[task_id] = "Task cancelled by user"
                    logger.info(f"Cancelled queued task {task_id}")
                else:
                    queue_items.append(task)
            
            # Put back non-cancelled tasks
            for task in queue_items:
                self._task_queue.put(task)
            
            return found_and_removed
    
    def cleanup(self) -> None:
        """Cleanup orchestrator resources"""
        logger.info("Cleaning up distributed orchestrator")
        
        # Stop orchestrator
        self.stop()
        
        # Clear all state
        self._completed_tasks.clear()
        self._failed_tasks.clear()
        self._allocation_plans.clear()
        self._metrics_history.clear()
        
        # Reset GPU allocations
        for gpu_id in self._gpu_allocations:
            self._gpu_allocations[gpu_id] = None
        
        logger.info("Distributed orchestrator cleanup completed")