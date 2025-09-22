"""
Strategy Selector: Intelligent Model Routing
Core component of Phase 4 optimization engine for selecting optimal evaluation strategies
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from .optimization_types import (
    StrategyType, 
    OptimizationGoal,
    ModelProfile,
    HardwareProfile,
    EvaluationProfile,
    PerformancePrediction,
    StrategyRecommendation
)

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Intelligent strategy selection for optimal evaluation performance
    
    Analyzes model characteristics, hardware capabilities, and evaluation requirements
    to recommend the best evaluation strategy (lightweight vs distributed variants)
    """
    
    def __init__(self):
        self.performance_history: Dict[str, List[Dict]] = {}
        self.strategy_success_rates: Dict[StrategyType, float] = {
            StrategyType.LIGHTWEIGHT: 0.95,
            StrategyType.DISTRIBUTED_TENSOR: 0.90,
            StrategyType.DISTRIBUTED_PIPELINE: 0.85,
            StrategyType.DISTRIBUTED_HYBRID: 0.88
        }
        
        # Strategy selection thresholds (can be tuned based on experience)
        self.lightweight_max_size_gb = 50.0  # Models <= 50GB use lightweight (simplified single threshold)
        self.distributed_min_gpus = 2       # Need >= 2 GPUs for distributed
        self.memory_safety_factor = 1.2     # 20% memory safety margin
        
        logger.info("StrategySelector initialized")
    
    def select_strategy(
        self,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile,
        evaluation_profile: EvaluationProfile
    ) -> StrategyRecommendation:
        """
        Select optimal evaluation strategy based on comprehensive analysis
        
        Args:
            model_profile: Characteristics of the model to evaluate
            hardware_profile: Available hardware resources
            evaluation_profile: Evaluation requirements and constraints
            
        Returns:
            StrategyRecommendation with optimal strategy and alternatives
        """
        logger.info(f"Selecting strategy for model: {model_profile.name} ({model_profile.size_gb:.1f}GB)")
        
        # Step 1: Analyze feasibility of each strategy
        feasible_strategies = self._analyze_strategy_feasibility(
            model_profile, hardware_profile, evaluation_profile
        )
        
        if not feasible_strategies:
            raise ValueError("No feasible evaluation strategies found for given constraints")
        
        # Step 2: Generate performance predictions for each feasible strategy
        predictions = []
        for strategy in feasible_strategies:
            prediction = self._predict_strategy_performance(
                strategy, model_profile, hardware_profile, evaluation_profile
            )
            predictions.append(prediction)
        
        # Step 3: Select optimal strategy based on optimization goal
        optimal_prediction = self._select_optimal_prediction(
            predictions, evaluation_profile.optimization_goal
        )
        
        # Step 4: Generate recommendation with alternatives
        alternatives = [p for p in predictions if p.strategy != optimal_prediction.strategy]
        alternatives.sort(key=lambda x: self._score_prediction(x, evaluation_profile.optimization_goal), reverse=True)
        
        recommendation = StrategyRecommendation(
            recommended_strategy=optimal_prediction.strategy,
            prediction=optimal_prediction,
            alternatives=alternatives[:3],  # Top 3 alternatives
            recommendation_reason=self._generate_recommendation_reason(optimal_prediction, model_profile, hardware_profile),
            trade_offs=self._analyze_trade_offs(optimal_prediction, alternatives),
            warnings=self._generate_warnings(optimal_prediction, model_profile, hardware_profile),
            optimal_batch_size=self._calculate_optimal_batch_size(optimal_prediction, evaluation_profile),
            suggested_settings=self._generate_suggested_settings(optimal_prediction, evaluation_profile)
        )
        
        logger.info(f"Selected strategy: {optimal_prediction.strategy.value} with {optimal_prediction.confidence:.2f} confidence")
        return recommendation
    
    def _analyze_strategy_feasibility(
        self,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile,
        evaluation_profile: EvaluationProfile
    ) -> List[StrategyType]:
        """Analyze which strategies are feasible given constraints"""
        feasible = []
        
        # Check lightweight strategy
        if self._is_lightweight_feasible(model_profile, hardware_profile):
            feasible.append(StrategyType.LIGHTWEIGHT)
        
        # Check distributed strategies (only if multi-GPU available and allowed)
        if hardware_profile.is_multi_gpu and evaluation_profile.allow_distributed:
            if self._is_distributed_tensor_feasible(model_profile, hardware_profile):
                feasible.append(StrategyType.DISTRIBUTED_TENSOR)
            
            if self._is_distributed_pipeline_feasible(model_profile, hardware_profile):
                feasible.append(StrategyType.DISTRIBUTED_PIPELINE)
            
            if self._is_distributed_hybrid_feasible(model_profile, hardware_profile):
                feasible.append(StrategyType.DISTRIBUTED_HYBRID)
        
        logger.debug(f"Feasible strategies: {[s.value for s in feasible]}")
        return feasible
    
    def _is_lightweight_feasible(self, model_profile: ModelProfile, hardware_profile: HardwareProfile) -> bool:
        """Check if lightweight strategy is feasible"""
        # Need enough memory on at least one GPU
        required_memory = model_profile.size_gb * self.memory_safety_factor
        max_available = max(hardware_profile.available_gpu_memory_gb) if hardware_profile.available_gpu_memory_gb else 0
        
        feasible = max_available >= required_memory
        logger.debug(f"Lightweight feasible: {feasible} (need {required_memory:.1f}GB, have {max_available:.1f}GB)")
        return feasible
    
    def _is_distributed_tensor_feasible(self, model_profile: ModelProfile, hardware_profile: HardwareProfile) -> bool:
        """Check if tensor parallel strategy is feasible"""
        if hardware_profile.num_gpus < 2:
            return False
        
        # Tensor parallel divides model across GPUs
        memory_per_gpu = (model_profile.size_gb * self.memory_safety_factor) / hardware_profile.num_gpus
        min_available = min(hardware_profile.available_gpu_memory_gb)
        
        feasible = min_available >= memory_per_gpu
        logger.debug(f"Tensor parallel feasible: {feasible} (need {memory_per_gpu:.1f}GB per GPU, min available {min_available:.1f}GB)")
        return feasible
    
    def _is_distributed_pipeline_feasible(self, model_profile: ModelProfile, hardware_profile: HardwareProfile) -> bool:
        """Check if pipeline parallel strategy is feasible"""
        if hardware_profile.num_gpus < 2:
            return False
        
        # Pipeline parallel needs model to be splittable into stages
        # Each stage needs to fit on one GPU
        stages = hardware_profile.num_gpus
        memory_per_stage = (model_profile.size_gb * self.memory_safety_factor) / stages
        min_available = min(hardware_profile.available_gpu_memory_gb)
        
        # Pipeline is most effective for larger models
        min_model_size_for_pipeline = 10.0  # GB
        
        feasible = (min_available >= memory_per_stage and 
                   model_profile.size_gb >= min_model_size_for_pipeline)
        logger.debug(f"Pipeline parallel feasible: {feasible} (need {memory_per_stage:.1f}GB per stage, model {model_profile.size_gb:.1f}GB)")
        return feasible
    
    def _is_distributed_hybrid_feasible(self, model_profile: ModelProfile, hardware_profile: HardwareProfile) -> bool:
        """Check if hybrid parallel strategy is feasible"""
        if hardware_profile.num_gpus < 4:  # Need at least 4 GPUs for effective hybrid
            return False
        
        # Hybrid combines tensor and pipeline parallelism
        # More complex memory calculation
        tensor_groups = 2  # Typical hybrid configuration
        pipeline_stages = hardware_profile.num_gpus // tensor_groups
        
        memory_per_gpu = (model_profile.size_gb * self.memory_safety_factor) / tensor_groups / pipeline_stages
        min_available = min(hardware_profile.available_gpu_memory_gb)
        
        # Hybrid is most effective for very large models
        min_model_size_for_hybrid = 20.0  # GB
        
        feasible = (min_available >= memory_per_gpu and 
                   model_profile.size_gb >= min_model_size_for_hybrid)
        logger.debug(f"Hybrid parallel feasible: {feasible} (need {memory_per_gpu:.1f}GB per GPU, model {model_profile.size_gb:.1f}GB)")
        return feasible
    
    def _predict_strategy_performance(
        self,
        strategy: StrategyType,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile,
        evaluation_profile: EvaluationProfile
    ) -> PerformancePrediction:
        """Predict performance metrics for a specific strategy"""
        
        # Base performance estimation
        base_time = self._estimate_base_evaluation_time(model_profile, evaluation_profile)
        base_memory = model_profile.size_gb
        base_cost = self._estimate_base_cost(base_time, hardware_profile)
        
        # Strategy-specific adjustments
        if strategy == StrategyType.LIGHTWEIGHT:
            time_multiplier = 1.0
            memory_multiplier = 1.2  # Some overhead
            cost_multiplier = 1.0
            required_gpus = 1
            confidence = 0.9
            
        elif strategy == StrategyType.DISTRIBUTED_TENSOR:
            time_multiplier = 0.6 / hardware_profile.num_gpus * 2  # Parallel speedup with communication overhead
            memory_multiplier = 1.1  # Less memory per GPU but some overhead
            cost_multiplier = hardware_profile.num_gpus * 0.8  # Multi-GPU but more efficient
            required_gpus = hardware_profile.num_gpus
            confidence = 0.8
            
        elif strategy == StrategyType.DISTRIBUTED_PIPELINE:
            time_multiplier = 0.7 / hardware_profile.num_gpus * 2.5  # Pipeline has more overhead
            memory_multiplier = 1.0  # Good memory efficiency
            cost_multiplier = hardware_profile.num_gpus * 0.85
            required_gpus = hardware_profile.num_gpus
            confidence = 0.75
            
        elif strategy == StrategyType.DISTRIBUTED_HYBRID:
            time_multiplier = 0.5 / hardware_profile.num_gpus * 3  # Best parallel efficiency but most overhead
            memory_multiplier = 1.05  # Most memory efficient
            cost_multiplier = hardware_profile.num_gpus * 0.7  # Most cost effective for large models
            required_gpus = hardware_profile.num_gpus
            confidence = 0.7  # Most complex, lower confidence
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply multipliers
        estimated_time = base_time * time_multiplier
        estimated_memory = base_memory * memory_multiplier
        estimated_cost = base_cost * cost_multiplier
        
        # Calculate memory distribution
        gpu_memory_per_device = self._calculate_memory_distribution(
            strategy, estimated_memory, required_gpus, hardware_profile
        )
        
        # Adjust confidence based on historical success rate
        confidence *= self.strategy_success_rates.get(strategy, 0.8)
        
        # Generate potential issues
        bottlenecks, risks = self._analyze_strategy_risks(strategy, model_profile, hardware_profile)
        
        return PerformancePrediction(
            strategy=strategy,
            estimated_time_minutes=estimated_time,
            estimated_memory_gb=estimated_memory,
            estimated_cost_usd=estimated_cost,
            confidence=confidence,
            required_gpus=required_gpus,
            gpu_memory_per_device=gpu_memory_per_device,
            peak_memory_gb=max(gpu_memory_per_device) if gpu_memory_per_device else estimated_memory,
            expected_success_rate=self.strategy_success_rates.get(strategy, 0.8),
            potential_bottlenecks=bottlenecks,
            risk_factors=risks
        )
    
    def _estimate_base_evaluation_time(self, model_profile: ModelProfile, evaluation_profile: EvaluationProfile) -> float:
        """Estimate base evaluation time in minutes"""
        # Use historical data if available
        if model_profile.average_eval_time:
            base_time = model_profile.average_eval_time
        else:
            # Rough estimation based on model size and dataset size
            # This would be refined with actual performance data
            size_factor = math.log(model_profile.size_gb + 1)  # Logarithmic scaling
            dataset_factor = evaluation_profile.dataset_size / 1000  # Linear with dataset size
            base_time = size_factor * dataset_factor * 2.0  # Base 2 minutes per log(GB) per 1K samples
        
        return max(base_time, 0.5)  # Minimum 30 seconds
    
    def _estimate_base_cost(self, time_minutes: float, hardware_profile: HardwareProfile) -> float:
        """Estimate base cost in USD"""
        # Rough cloud GPU pricing (would be configurable in production)
        gpu_hour_cost = 2.50  # USD per GPU per hour (rough A100 pricing)
        cost = (time_minutes / 60) * gpu_hour_cost * hardware_profile.num_gpus
        return cost
    
    def _calculate_memory_distribution(
        self,
        strategy: StrategyType,
        total_memory: float,
        required_gpus: int,
        hardware_profile: HardwareProfile
    ) -> List[float]:
        """Calculate memory distribution across GPUs for strategy"""
        if strategy == StrategyType.LIGHTWEIGHT:
            return [total_memory]
        
        elif strategy == StrategyType.DISTRIBUTED_TENSOR:
            # Memory evenly distributed
            memory_per_gpu = total_memory / required_gpus
            return [memory_per_gpu] * required_gpus
        
        elif strategy == StrategyType.DISTRIBUTED_PIPELINE:
            # Memory might be uneven across pipeline stages
            # For simplicity, assume even distribution (can be refined)
            memory_per_stage = total_memory / required_gpus
            return [memory_per_stage] * required_gpus
        
        elif strategy == StrategyType.DISTRIBUTED_HYBRID:
            # Complex hybrid distribution (simplified)
            memory_per_gpu = total_memory / required_gpus * 0.9  # Some efficiency gain
            return [memory_per_gpu] * required_gpus
        
        return [total_memory / required_gpus] * required_gpus
    
    def _analyze_strategy_risks(
        self,
        strategy: StrategyType,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile
    ) -> Tuple[List[str], List[str]]:
        """Analyze potential bottlenecks and risk factors for strategy"""
        bottlenecks = []
        risks = []
        
        if strategy == StrategyType.LIGHTWEIGHT:
            if model_profile.size_gb > 5.0:
                bottlenecks.append("Large model on single GPU may cause memory pressure")
            if hardware_profile.gpu_memory_gb[0] < model_profile.size_gb * 1.5:
                risks.append("Memory utilization will be high, risk of OOM")
        
        elif strategy in [StrategyType.DISTRIBUTED_TENSOR, StrategyType.DISTRIBUTED_PIPELINE, StrategyType.DISTRIBUTED_HYBRID]:
            if hardware_profile.inter_gpu_bandwidth_gbps and hardware_profile.inter_gpu_bandwidth_gbps < 100:
                bottlenecks.append("Low inter-GPU bandwidth may limit parallel efficiency")
            
            if hardware_profile.num_gpus > 4:
                risks.append("Large GPU count increases communication overhead")
            
            if strategy == StrategyType.DISTRIBUTED_PIPELINE:
                bottlenecks.append("Pipeline parallelism may have bubble overhead")
            
            if strategy == StrategyType.DISTRIBUTED_HYBRID:
                risks.append("Hybrid parallelism is complex and may have tuning challenges")
        
        return bottlenecks, risks
    
    def _select_optimal_prediction(
        self,
        predictions: List[PerformancePrediction],
        goal: OptimizationGoal
    ) -> PerformancePrediction:
        """Select optimal prediction based on optimization goal"""
        if not predictions:
            raise ValueError("No predictions to select from")
        
        scored_predictions = [
            (self._score_prediction(pred, goal), pred) for pred in predictions
        ]
        scored_predictions.sort(key=lambda x: x[0], reverse=True)
        
        return scored_predictions[0][1]
    
    def _score_prediction(self, prediction: PerformancePrediction, goal: OptimizationGoal) -> float:
        """Score a prediction based on optimization goal"""
        base_score = prediction.confidence * prediction.expected_success_rate
        
        if goal == OptimizationGoal.MINIMIZE_TIME:
            time_score = max(0, 1 - prediction.estimated_time_minutes / 60)  # Normalize to 1 hour
            return base_score * 0.7 + time_score * 0.3
        
        elif goal == OptimizationGoal.MINIMIZE_MEMORY:
            memory_score = max(0, 1 - prediction.estimated_memory_gb / 100)  # Normalize to 100GB
            return base_score * 0.7 + memory_score * 0.3
        
        elif goal == OptimizationGoal.MINIMIZE_COST:
            cost_score = max(0, 1 - prediction.estimated_cost_usd / 50)  # Normalize to $50
            return base_score * 0.7 + cost_score * 0.3
        
        elif goal == OptimizationGoal.MAXIMIZE_THROUGHPUT:
            # Inverse of time, weighted by GPU utilization
            throughput_score = 1 / (prediction.estimated_time_minutes + 1) * prediction.required_gpus
            return base_score * 0.5 + throughput_score * 0.5
        
        else:  # BALANCED
            # Balanced scoring considers all factors
            time_score = max(0, 1 - prediction.estimated_time_minutes / 30)
            memory_score = max(0, 1 - prediction.estimated_memory_gb / 50)
            cost_score = max(0, 1 - prediction.estimated_cost_usd / 20)
            
            return base_score * 0.4 + (time_score + memory_score + cost_score) / 3 * 0.6
    
    def _calculate_optimal_batch_size(
        self,
        prediction: PerformancePrediction,
        evaluation_profile: EvaluationProfile
    ) -> Optional[int]:
        """Calculate optimal batch size for the selected strategy"""
        if evaluation_profile.batch_size:
            return evaluation_profile.batch_size
        
        # Estimate optimal batch size based on available memory
        available_memory = min(prediction.gpu_memory_per_device) * 0.8  # 80% utilization target
        
        # Rough estimation: 1GB memory can handle batch size of 16 for medium models
        # This would be refined with actual profiling data
        memory_per_sample_gb = 0.06  # 60MB per sample (rough estimate)
        optimal_batch = int(available_memory / memory_per_sample_gb)
        
        # Clamp to reasonable values
        optimal_batch = max(1, min(optimal_batch, 64))
        
        return optimal_batch
    
    def _generate_suggested_settings(
        self,
        prediction: PerformancePrediction,
        evaluation_profile: EvaluationProfile
    ) -> Dict[str, Any]:
        """Generate suggested configuration settings"""
        settings = {}
        
        if prediction.strategy == StrategyType.LIGHTWEIGHT:
            settings.update({
                "use_half_precision": True,
                "gradient_checkpointing": prediction.estimated_memory_gb > 10,
                "max_memory_fraction": 0.9
            })
        
        elif prediction.strategy == StrategyType.DISTRIBUTED_TENSOR:
            settings.update({
                "tensor_parallel_size": prediction.required_gpus,
                "use_half_precision": True,
                "gradient_checkpointing": False,
                "communication_backend": "nccl"
            })
        
        elif prediction.strategy == StrategyType.DISTRIBUTED_PIPELINE:
            settings.update({
                "pipeline_parallel_size": prediction.required_gpus,
                "micro_batch_size": 1,
                "use_half_precision": True,
                "gradient_checkpointing": True
            })
        
        elif prediction.strategy == StrategyType.DISTRIBUTED_HYBRID:
            tensor_parallel_size = min(4, prediction.required_gpus // 2)
            pipeline_parallel_size = prediction.required_gpus // tensor_parallel_size
            
            settings.update({
                "tensor_parallel_size": tensor_parallel_size,
                "pipeline_parallel_size": pipeline_parallel_size,
                "use_half_precision": True,
                "gradient_checkpointing": True,
                "communication_backend": "nccl"
            })
        
        return settings
    
    def _generate_recommendation_reason(
        self,
        prediction: PerformancePrediction,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile
    ) -> str:
        """Generate human-readable reason for recommendation"""
        strategy_name = prediction.strategy.value.replace('_', ' ').title()
        
        if prediction.strategy == StrategyType.LIGHTWEIGHT:
            return f"{strategy_name} selected for {model_profile.size_gb:.1f}GB model on single GPU. Optimal for models under {self.lightweight_max_size_gb}GB."
        
        elif prediction.strategy == StrategyType.DISTRIBUTED_TENSOR:
            return f"{strategy_name} selected to distribute {model_profile.size_gb:.1f}GB model across {hardware_profile.num_gpus} GPUs. Good parallel efficiency for tensor operations."
        
        elif prediction.strategy == StrategyType.DISTRIBUTED_PIPELINE:
            return f"{strategy_name} selected for {model_profile.size_gb:.1f}GB model across {hardware_profile.num_gpus} GPUs. Effective for memory-bound large models."
        
        elif prediction.strategy == StrategyType.DISTRIBUTED_HYBRID:
            return f"{strategy_name} selected for {model_profile.size_gb:.1f}GB model across {hardware_profile.num_gpus} GPUs. Optimal parallel efficiency for very large models."
        
        return f"{strategy_name} selected based on model size and hardware configuration."
    
    def _analyze_trade_offs(
        self,
        optimal: PerformancePrediction,
        alternatives: List[PerformancePrediction]
    ) -> List[str]:
        """Analyze trade-offs between optimal and alternative strategies"""
        trade_offs = []
        
        for alt in alternatives[:2]:  # Top 2 alternatives
            if alt.estimated_time_minutes < optimal.estimated_time_minutes:
                time_diff = optimal.estimated_time_minutes - alt.estimated_time_minutes
                trade_offs.append(f"{alt.strategy.value} could be {time_diff:.1f} minutes faster but has {optimal.confidence - alt.confidence:.0%} lower confidence")
            
            if alt.estimated_cost_usd < optimal.estimated_cost_usd:
                cost_diff = optimal.estimated_cost_usd - alt.estimated_cost_usd
                trade_offs.append(f"{alt.strategy.value} could save ${cost_diff:.2f} but may have reliability concerns")
        
        return trade_offs
    
    def _generate_warnings(
        self,
        prediction: PerformancePrediction,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile
    ) -> List[str]:
        """Generate warnings for the selected strategy"""
        warnings = []
        
        if prediction.confidence < 0.7:
            warnings.append(f"Low confidence ({prediction.confidence:.0%}) in prediction - monitor performance closely")
        
        if prediction.peak_memory_gb > max(hardware_profile.available_gpu_memory_gb) * 0.9:
            warnings.append("High memory utilization predicted - risk of out-of-memory errors")
        
        if prediction.estimated_time_minutes > 60:
            warnings.append("Long evaluation time predicted - consider model optimization or smaller datasets")
        
        if len(prediction.potential_bottlenecks) > 0:
            warnings.append(f"Potential bottlenecks: {', '.join(prediction.potential_bottlenecks)}")
        
        return warnings
    
    def update_performance_history(self, model_name: str, strategy: StrategyType, metrics: Dict[str, Any]):
        """Update performance history for learning"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            'strategy': strategy,
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        # Update success rates based on actual performance
        if 'success_rate' in metrics:
            current_rate = self.strategy_success_rates[strategy]
            new_rate = metrics['success_rate']
            # Exponentially weighted moving average
            self.strategy_success_rates[strategy] = current_rate * 0.9 + new_rate * 0.1
        
        logger.info(f"Updated performance history for {model_name} with {strategy.value}")
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get strategy selection statistics"""
        return {
            'success_rates': dict(self.strategy_success_rates),
            'history_count': {model: len(history) for model, history in self.performance_history.items()},
            'total_evaluations': sum(len(history) for history in self.performance_history.values())
        }