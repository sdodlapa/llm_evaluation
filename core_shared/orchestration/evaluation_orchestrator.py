"""
Evaluation orchestrator for hybrid architecture

Central coordinator that routes evaluation requests to appropriate engines
based on model requirements and resource availability.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import uuid

from ..interfaces.evaluation_interfaces import (
    EvaluationRequest, EvaluationResult, EvaluationEngine, EngineType,
    ResourceRequirements, EngineRegistry
)
from ..model_registry.enhanced_model_config import EnhancedModelConfig
from ..model_registry.model_registry_enhanced import EnhancedModelRegistry


logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Status of evaluation requests"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Request priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class OrchestratorConfig:
    """Configuration for evaluation orchestrator"""
    # Engine selection strategy
    prefer_lightweight: bool = True
    auto_fallback: bool = True
    max_queue_size: int = 100
    
    # Timing and retries
    engine_selection_timeout_seconds: float = 30.0
    evaluation_timeout_seconds: float = 3600.0  # 1 hour default
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 60.0
    
    # Resource management
    enable_resource_monitoring: bool = True
    resource_check_interval_seconds: float = 30.0
    
    # Load balancing
    load_balance_strategy: str = "round_robin"  # "round_robin", "least_loaded", "resource_based"
    enable_request_batching: bool = True
    max_batch_size: int = 5
    batch_timeout_seconds: float = 10.0
    
    # Monitoring and logging
    enable_detailed_logging: bool = True
    log_performance_metrics: bool = True
    metrics_collection_interval: float = 60.0


@dataclass
class RequestContext:
    """Context for tracking evaluation requests"""
    request: EvaluationRequest
    status: RequestStatus = RequestStatus.PENDING
    priority: Priority = Priority.MEDIUM
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_engine: Optional[str] = None
    retry_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    result: Optional[EvaluationResult] = None


class EvaluationOrchestrator:
    """Central orchestrator for evaluation requests"""
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        
        # Core components
        self.engine_registry = EngineRegistry()
        self.model_registry = EnhancedModelRegistry()
        
        # Request management
        self._request_queue: Dict[str, RequestContext] = {}
        self._active_requests: Dict[str, RequestContext] = {}
        self._completed_requests: Dict[str, RequestContext] = {}
        
        # Engine management
        self._engine_status: Dict[str, Dict[str, Any]] = {}
        self._engine_load: Dict[str, int] = {}
        
        # Performance tracking
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "engine_usage": {},
            "average_processing_time": 0.0
        }
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {
            "request_submitted": [],
            "request_started": [],
            "request_completed": [],
            "request_failed": [],
            "engine_selected": []
        }
        
        self._running = False
        logger.info("Evaluation Orchestrator initialized")
    
    async def start(self):
        """Start the orchestrator"""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        logger.info("Starting Evaluation Orchestrator")
        
        # Start background tasks
        if self.config.enable_resource_monitoring:
            asyncio.create_task(self._resource_monitoring_loop())
        
        asyncio.create_task(self._request_processing_loop())
        asyncio.create_task(self._metrics_collection_loop())
    
    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping Evaluation Orchestrator")
        self._running = False
        
        # Wait for active requests to complete or timeout
        timeout = 30.0
        start_time = time.time()
        
        while self._active_requests and (time.time() - start_time) < timeout:
            await asyncio.sleep(1.0)
        
        # Cancel remaining requests
        for context in self._active_requests.values():
            context.status = RequestStatus.CANCELLED
        
        logger.info("Evaluation Orchestrator stopped")
    
    def register_engine(self, engine: EvaluationEngine):
        """Register an evaluation engine"""
        self.engine_registry.register_engine(engine)
        self._engine_status[engine.engine_id] = {
            "available": True,
            "last_checked": time.time(),
            "active_evaluations": 0
        }
        self._engine_load[engine.engine_id] = 0
        logger.info(f"Registered engine: {engine.engine_id}")
    
    def register_model(self, model_config: EnhancedModelConfig):
        """Register a model configuration"""
        success = self.model_registry.register_model(model_config)
        if success:
            logger.info(f"Registered model: {model_config.model_name}")
        return success
    
    async def submit_request(self, request: EvaluationRequest, priority: Priority = Priority.MEDIUM) -> str:
        """Submit an evaluation request
        
        Args:
            request: The evaluation request
            priority: Request priority
            
        Returns:
            str: Request ID for tracking
        """
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        
        # Create request context
        context = RequestContext(
            request=request,
            priority=priority,
            status=RequestStatus.PENDING
        )
        
        # Validate request
        validation_errors = await self._validate_request(request)
        if validation_errors:
            context.status = RequestStatus.FAILED
            context.error_messages = validation_errors
            self._completed_requests[request.request_id] = context
            await self._trigger_event("request_failed", context)
            return request.request_id
        
        # Queue the request
        self._request_queue[request.request_id] = context
        self._metrics["total_requests"] += 1
        
        logger.info(f"Submitted request {request.request_id} for model {request.model_config.model_name}")
        await self._trigger_event("request_submitted", context)
        
        return request.request_id
    
    async def get_request_status(self, request_id: str) -> Optional[RequestContext]:
        """Get status of a request"""
        # Check all request stores
        for store in [self._request_queue, self._active_requests, self._completed_requests]:
            if request_id in store:
                return store[request_id]
        return None
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or active request"""
        context = await self.get_request_status(request_id)
        if not context:
            return False
        
        if context.status in [RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED]:
            return False
        
        # Remove from queue if pending
        if request_id in self._request_queue:
            del self._request_queue[request_id]
        
        # Mark as cancelled
        context.status = RequestStatus.CANCELLED
        context.completed_at = time.time()
        self._completed_requests[request_id] = context
        
        # Remove from active if running
        if request_id in self._active_requests:
            del self._active_requests[request_id]
        
        logger.info(f"Cancelled request {request_id}")
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        current_metrics = self._metrics.copy()
        current_metrics.update({
            "queued_requests": len(self._request_queue),
            "active_requests": len(self._active_requests),
            "completed_requests": len(self._completed_requests),
            "engine_status": self._engine_status.copy(),
            "engine_load": self._engine_load.copy()
        })
        return current_metrics
    
    def add_event_handler(self, event_name: str, handler: Callable):
        """Add event handler"""
        if event_name in self._event_handlers:
            self._event_handlers[event_name].append(handler)
    
    async def _request_processing_loop(self):
        """Main request processing loop"""
        while self._running:
            try:
                # Process pending requests
                await self._process_pending_requests()
                
                # Check active requests for completion/timeout
                await self._check_active_requests()
                
                # Small delay to prevent busy loop
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in request processing loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_pending_requests(self):
        """Process pending requests in priority order"""
        if not self._request_queue:
            return
        
        # Sort by priority and submission time
        sorted_requests = sorted(
            self._request_queue.items(),
            key=lambda x: (-x[1].priority.value, x[1].submitted_at)
        )
        
        for request_id, context in sorted_requests:
            # Check if we can process more requests
            if len(self._active_requests) >= self.config.max_queue_size:
                break
            
            try:
                # Select engine for request
                engine = await self._select_engine(context.request)
                if not engine:
                    # No available engine, leave in queue
                    continue
                
                # Move to active and start processing
                del self._request_queue[request_id]
                context.status = RequestStatus.RUNNING
                context.started_at = time.time()
                context.assigned_engine = engine.engine_id
                self._active_requests[request_id] = context
                
                # Update engine load
                self._engine_load[engine.engine_id] = self._engine_load.get(engine.engine_id, 0) + 1
                
                # Start evaluation task
                asyncio.create_task(self._execute_evaluation(context, engine))
                
                await self._trigger_event("request_started", context)
                logger.info(f"Started processing request {request_id} on engine {engine.engine_id}")
                
            except Exception as e:
                logger.error(f"Failed to start request {request_id}: {e}")
                context.status = RequestStatus.FAILED
                context.error_messages.append(str(e))
                self._completed_requests[request_id] = context
                if request_id in self._request_queue:
                    del self._request_queue[request_id]
    
    async def _select_engine(self, request: EvaluationRequest) -> Optional[EvaluationEngine]:
        """Select optimal engine for request"""
        try:
            # Get compatible engines
            compatible_engines = self.engine_registry.get_compatible_engines(request)
            if not compatible_engines:
                logger.warning(f"No compatible engines for request {request.request_id}")
                return None
            
            # Filter by availability and load
            available_engines = []
            for engine in compatible_engines:
                engine_status = self._engine_status.get(engine.engine_id, {})
                if engine_status.get("available", False):
                    available_engines.append(engine)
            
            if not available_engines:
                return None
            
            # Select based on strategy
            selected = self._apply_selection_strategy(available_engines, request)
            
            if selected:
                await self._trigger_event("engine_selected", {
                    "request_id": request.request_id,
                    "engine_id": selected.engine_id,
                    "engine_type": selected.capabilities.engine_type
                })
            
            return selected
            
        except Exception as e:
            logger.error(f"Engine selection failed for request {request.request_id}: {e}")
            return None
    
    def _apply_selection_strategy(self, engines: List[EvaluationEngine], 
                                request: EvaluationRequest) -> Optional[EvaluationEngine]:
        """Apply engine selection strategy"""
        if not engines:
            return None
        
        strategy = self.config.load_balance_strategy
        
        if strategy == "round_robin":
            # Simple round-robin based on usage metrics
            engine_usage = {e.engine_id: self._metrics["engine_usage"].get(e.engine_id, 0) for e in engines}
            return min(engines, key=lambda e: engine_usage[e.engine_id])
        
        elif strategy == "least_loaded":
            # Select engine with lowest current load
            return min(engines, key=lambda e: self._engine_load.get(e.engine_id, 0))
        
        elif strategy == "resource_based":
            # Select based on resource requirements
            # For now, use least loaded as fallback
            return min(engines, key=lambda e: self._engine_load.get(e.engine_id, 0))
        
        else:
            # Default to first available
            return engines[0]
    
    async def _execute_evaluation(self, context: RequestContext, engine: EvaluationEngine):
        """Execute evaluation on selected engine"""
        request_id = context.request.request_id
        
        try:
            # Execute the evaluation
            result = engine.evaluate(context.request)
            
            # Store result
            context.result = result
            context.status = RequestStatus.COMPLETED
            context.completed_at = time.time()
            
            # Update metrics
            self._metrics["completed_requests"] += 1
            engine_id = engine.engine_id
            self._metrics["engine_usage"][engine_id] = self._metrics["engine_usage"].get(engine_id, 0) + 1
            
            # Calculate processing time
            processing_time = context.completed_at - context.started_at
            self._update_average_processing_time(processing_time)
            
            logger.info(f"Completed request {request_id} in {processing_time:.2f}s")
            await self._trigger_event("request_completed", context)
            
        except Exception as e:
            logger.error(f"Evaluation failed for request {request_id}: {e}")
            context.status = RequestStatus.FAILED
            context.error_messages.append(str(e))
            context.completed_at = time.time()
            self._metrics["failed_requests"] += 1
            await self._trigger_event("request_failed", context)
        
        finally:
            # Clean up
            if request_id in self._active_requests:
                del self._active_requests[request_id]
            
            # Move to completed
            self._completed_requests[request_id] = context
            
            # Update engine load
            engine_id = engine.engine_id
            self._engine_load[engine_id] = max(0, self._engine_load.get(engine_id, 0) - 1)
    
    async def _check_active_requests(self):
        """Check active requests for timeouts"""
        current_time = time.time()
        timeout_duration = self.config.evaluation_timeout_seconds
        
        timed_out = []
        for request_id, context in self._active_requests.items():
            if context.started_at and (current_time - context.started_at) > timeout_duration:
                timed_out.append(request_id)
        
        for request_id in timed_out:
            context = self._active_requests[request_id]
            context.status = RequestStatus.FAILED
            context.error_messages.append("Evaluation timeout")
            context.completed_at = current_time
            
            del self._active_requests[request_id]
            self._completed_requests[request_id] = context
            
            # Update engine load
            if context.assigned_engine:
                engine_id = context.assigned_engine
                self._engine_load[engine_id] = max(0, self._engine_load.get(engine_id, 0) - 1)
            
            logger.warning(f"Request {request_id} timed out")
            await self._trigger_event("request_failed", context)
    
    async def _validate_request(self, request: EvaluationRequest) -> List[str]:
        """Validate evaluation request"""
        errors = []
        
        # Check if model is registered
        model_config = self.model_registry.get_model(request.model_config.model_name)
        if not model_config:
            errors.append(f"Model {request.model_config.model_name} not registered")
        
        # Check if compatible engines exist
        compatible_engines = self.engine_registry.get_compatible_engines(request)
        if not compatible_engines:
            errors.append("No compatible engines available")
        
        return errors
    
    async def _resource_monitoring_loop(self):
        """Monitor engine resources"""
        while self._running:
            try:
                # Update engine status
                for engine_id in self._engine_status:
                    # This would check actual engine health/availability
                    # For now, keep engines available
                    self._engine_status[engine_id]["last_checked"] = time.time()
                
                await asyncio.sleep(self.config.resource_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(30.0)
    
    async def _metrics_collection_loop(self):
        """Collect performance metrics"""
        while self._running:
            try:
                # This would collect detailed metrics
                # For now, just log current state
                if self.config.log_performance_metrics:
                    metrics = self.get_metrics()
                    logger.debug(f"Orchestrator metrics: {metrics}")
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(60.0)
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time metric"""
        current_avg = self._metrics["average_processing_time"]
        completed_count = self._metrics["completed_requests"]
        
        if completed_count == 1:
            self._metrics["average_processing_time"] = processing_time
        else:
            # Running average
            self._metrics["average_processing_time"] = (
                (current_avg * (completed_count - 1) + processing_time) / completed_count
            )
    
    async def _trigger_event(self, event_name: str, data: Any):
        """Trigger event handlers"""
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")