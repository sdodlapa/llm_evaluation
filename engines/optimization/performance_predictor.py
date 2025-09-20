"""
Performance Predictor: ML-Based Performance Estimation
Advanced component for predicting evaluation performance using machine learning
"""

import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import math
import time
from pathlib import Path
import numpy as np

from .optimization_types import (
    StrategyType,
    ModelProfile,
    HardwareProfile, 
    EvaluationProfile,
    PerformancePrediction,
    OptimizationMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceFeatures:
    """Feature vector for performance prediction"""
    # Model features
    model_size_gb: float
    parameter_count: float
    context_length: float
    
    # Hardware features
    num_gpus: int
    total_gpu_memory_gb: float
    gpu_compute_capability: float
    inter_gpu_bandwidth_gbps: float
    
    # Evaluation features
    dataset_size: int
    batch_size: int
    
    # Strategy features
    strategy_type: int  # Encoded strategy
    
    def to_vector(self) -> List[float]:
        """Convert to numerical feature vector"""
        return [
            self.model_size_gb,
            math.log(self.parameter_count + 1),
            math.log(self.context_length + 1),
            self.num_gpus,
            self.total_gpu_memory_gb,
            self.gpu_compute_capability,
            self.inter_gpu_bandwidth_gbps or 0.0,
            math.log(self.dataset_size + 1),
            self.batch_size,
            self.strategy_type
        ]


class PerformancePredictor:
    """
    ML-based performance prediction system
    
    Uses historical performance data to predict evaluation metrics:
    - Execution time
    - Memory usage
    - Cost estimation
    - Success probability
    """
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        self.model_cache_dir = Path(model_cache_dir or "model_cache/performance_models")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Training data storage
        self.training_data: List[Tuple[PerformanceFeatures, OptimizationMetrics]] = []
        self.feature_stats: Optional[Dict[str, Tuple[float, float]]] = None  # (mean, std) for normalization
        
        # Simple regression models (would use sklearn in production)
        self.time_model: Optional[Dict] = None
        self.memory_model: Optional[Dict] = None
        self.cost_model: Optional[Dict] = None
        self.success_model: Optional[Dict] = None
        
        # Strategy encoding
        self.strategy_encoding = {
            StrategyType.LIGHTWEIGHT: 1,
            StrategyType.DISTRIBUTED_TENSOR: 2,
            StrategyType.DISTRIBUTED_PIPELINE: 3,
            StrategyType.DISTRIBUTED_HYBRID: 4
        }
        
        # Load existing models if available
        self._load_models()
        
        logger.info("PerformancePredictor initialized")
    
    def predict_performance(
        self,
        strategy: StrategyType,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile,
        evaluation_profile: EvaluationProfile
    ) -> PerformancePrediction:
        """
        Predict performance metrics for given configuration
        
        Args:
            strategy: Evaluation strategy to predict for
            model_profile: Model characteristics
            hardware_profile: Hardware configuration
            evaluation_profile: Evaluation requirements
            
        Returns:
            PerformancePrediction with estimated metrics and confidence
        """
        # Extract features
        features = self._extract_features(strategy, model_profile, hardware_profile, evaluation_profile)
        
        # Use ML models if trained, otherwise fall back to heuristics
        if self._models_trained():
            prediction = self._predict_with_ml(features, strategy, hardware_profile)
        else:
            prediction = self._predict_with_heuristics(features, strategy, model_profile, hardware_profile, evaluation_profile)
        
        logger.debug(f"Predicted performance for {strategy.value}: {prediction.estimated_time_minutes:.1f}min, {prediction.estimated_memory_gb:.1f}GB")
        return prediction
    
    def _extract_features(
        self,
        strategy: StrategyType,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile,
        evaluation_profile: EvaluationProfile
    ) -> PerformanceFeatures:
        """Extract feature vector from profiles"""
        # Estimate batch size if not provided
        batch_size = evaluation_profile.batch_size or self._estimate_batch_size(model_profile, hardware_profile)
        
        # Convert GPU compute capability to numerical value
        gpu_compute = float(hardware_profile.gpu_compute_capability[0].replace('.', '')) if hardware_profile.gpu_compute_capability else 80.0
        
        return PerformanceFeatures(
            model_size_gb=model_profile.size_gb,
            parameter_count=model_profile.parameter_count or (model_profile.size_gb * 0.5e9),  # Rough estimate
            context_length=model_profile.context_length or 2048,
            num_gpus=hardware_profile.num_gpus,
            total_gpu_memory_gb=hardware_profile.total_gpu_memory_gb,
            gpu_compute_capability=gpu_compute,
            inter_gpu_bandwidth_gbps=hardware_profile.inter_gpu_bandwidth_gbps or 0.0,
            dataset_size=evaluation_profile.dataset_size,
            batch_size=batch_size,
            strategy_type=self.strategy_encoding[strategy]
        )
    
    def _predict_with_ml(self, features: PerformanceFeatures, strategy: StrategyType, hardware_profile: HardwareProfile) -> PerformancePrediction:
        """Predict using trained ML models"""
        feature_vector = self._normalize_features(features.to_vector())
        
        # Predict using simple linear models (would use sklearn in production)
        estimated_time = self._predict_with_model(self.time_model, feature_vector)
        estimated_memory = self._predict_with_model(self.memory_model, feature_vector)
        estimated_cost = self._predict_with_model(self.cost_model, feature_vector)
        success_probability = self._predict_with_model(self.success_model, feature_vector)
        
        # Calculate confidence based on training data size and feature similarity
        confidence = self._calculate_confidence(features)
        
        # Calculate memory distribution
        gpu_memory_per_device = self._distribute_memory(estimated_memory, strategy, hardware_profile.num_gpus)
        
        return PerformancePrediction(
            strategy=strategy,
            estimated_time_minutes=max(0.5, estimated_time),
            estimated_memory_gb=max(0.1, estimated_memory),
            estimated_cost_usd=max(0.01, estimated_cost),
            confidence=confidence,
            required_gpus=hardware_profile.num_gpus if strategy != StrategyType.LIGHTWEIGHT else 1,
            gpu_memory_per_device=gpu_memory_per_device,
            peak_memory_gb=max(gpu_memory_per_device) if gpu_memory_per_device else estimated_memory,
            expected_success_rate=max(0.1, min(1.0, success_probability)),
            potential_bottlenecks=self._predict_bottlenecks(features, strategy),
            risk_factors=self._predict_risks(features, strategy)
        )
    
    def _predict_with_heuristics(
        self,
        features: PerformanceFeatures,
        strategy: StrategyType,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile,
        evaluation_profile: EvaluationProfile
    ) -> PerformancePrediction:
        """Predict using heuristic models when ML models aren't trained"""
        
        # Heuristic time estimation
        base_time_per_sample = 0.1  # 100ms per sample base
        model_complexity_factor = math.log(features.model_size_gb + 1)
        dataset_factor = features.dataset_size / features.batch_size  # Number of batches
        
        if strategy == StrategyType.LIGHTWEIGHT:
            strategy_factor = 1.0
        elif strategy == StrategyType.DISTRIBUTED_TENSOR:
            strategy_factor = max(0.3, 1.0 / features.num_gpus * 1.2)  # Parallel efficiency with overhead
        elif strategy == StrategyType.DISTRIBUTED_PIPELINE:
            strategy_factor = max(0.4, 1.0 / features.num_gpus * 1.5)  # More overhead
        else:  # HYBRID
            strategy_factor = max(0.2, 1.0 / features.num_gpus * 1.8)  # Best parallel, most overhead
        
        estimated_time = base_time_per_sample * model_complexity_factor * dataset_factor * strategy_factor / 60  # Convert to minutes
        
        # Heuristic memory estimation
        base_memory = features.model_size_gb
        overhead_factor = 1.2 if strategy == StrategyType.LIGHTWEIGHT else 1.1
        estimated_memory = base_memory * overhead_factor
        
        # Heuristic cost estimation (cloud pricing)
        gpu_hour_cost = 2.50  # USD per GPU hour
        num_gpus = 1 if strategy == StrategyType.LIGHTWEIGHT else features.num_gpus
        estimated_cost = (estimated_time / 60) * gpu_hour_cost * num_gpus
        
        # Heuristic success rate
        memory_pressure = estimated_memory / features.total_gpu_memory_gb
        if memory_pressure > 0.9:
            success_rate = 0.7
        elif memory_pressure > 0.8:
            success_rate = 0.85
        else:
            success_rate = 0.95
        
        # Lower success rate for more complex strategies
        strategy_reliability = {
            StrategyType.LIGHTWEIGHT: 0.95,
            StrategyType.DISTRIBUTED_TENSOR: 0.90,
            StrategyType.DISTRIBUTED_PIPELINE: 0.85,
            StrategyType.DISTRIBUTED_HYBRID: 0.80
        }
        success_rate *= strategy_reliability[strategy]
        
        # Calculate memory distribution
        gpu_memory_per_device = self._distribute_memory(estimated_memory, strategy, hardware_profile.num_gpus)
        
        # Confidence based on heuristic reliability
        confidence = 0.6  # Lower confidence for heuristics
        
        return PerformancePrediction(
            strategy=strategy,
            estimated_time_minutes=max(0.5, estimated_time),
            estimated_memory_gb=max(0.1, estimated_memory),
            estimated_cost_usd=max(0.01, estimated_cost),
            confidence=confidence,
            required_gpus=1 if strategy == StrategyType.LIGHTWEIGHT else features.num_gpus,
            gpu_memory_per_device=gpu_memory_per_device,
            peak_memory_gb=max(gpu_memory_per_device) if gpu_memory_per_device else estimated_memory,
            expected_success_rate=success_rate,
            potential_bottlenecks=self._predict_bottlenecks(features, strategy),
            risk_factors=self._predict_risks(features, strategy)
        )
    
    def _distribute_memory(self, total_memory: float, strategy: StrategyType, num_gpus: int) -> List[float]:
        """Distribute memory across GPUs based on strategy"""
        if strategy == StrategyType.LIGHTWEIGHT:
            return [total_memory]
        elif strategy == StrategyType.DISTRIBUTED_TENSOR:
            memory_per_gpu = total_memory / num_gpus
            return [memory_per_gpu] * num_gpus
        elif strategy == StrategyType.DISTRIBUTED_PIPELINE:
            # Pipeline might have uneven distribution, but use even for simplicity
            memory_per_gpu = total_memory / num_gpus
            return [memory_per_gpu] * num_gpus
        else:  # HYBRID
            memory_per_gpu = total_memory / num_gpus * 0.95  # Slight efficiency gain
            return [memory_per_gpu] * num_gpus
    
    def _predict_bottlenecks(self, features: PerformanceFeatures, strategy: StrategyType) -> List[str]:
        """Predict potential bottlenecks"""
        bottlenecks = []
        
        # Memory bottlenecks
        memory_ratio = features.model_size_gb / features.total_gpu_memory_gb
        if memory_ratio > 0.8:
            bottlenecks.append("High memory utilization")
        
        # Communication bottlenecks for distributed strategies
        if strategy != StrategyType.LIGHTWEIGHT and features.num_gpus > 1:
            if features.inter_gpu_bandwidth_gbps < 100:
                bottlenecks.append("Limited inter-GPU bandwidth")
            if features.num_gpus > 4:
                bottlenecks.append("Communication overhead with many GPUs")
        
        # Compute bottlenecks
        if features.gpu_compute_capability < 75:  # Older GPUs
            bottlenecks.append("Limited GPU compute capability")
        
        # Batch size bottlenecks
        if features.batch_size < 4:
            bottlenecks.append("Small batch size may underutilize GPU")
        elif features.batch_size > 32:
            bottlenecks.append("Large batch size may cause memory pressure")
        
        return bottlenecks
    
    def _predict_risks(self, features: PerformanceFeatures, strategy: StrategyType) -> List[str]:
        """Predict risk factors"""
        risks = []
        
        # Memory risks
        memory_ratio = features.model_size_gb / features.total_gpu_memory_gb
        if memory_ratio > 0.9:
            risks.append("High risk of out-of-memory errors")
        
        # Strategy complexity risks
        if strategy == StrategyType.DISTRIBUTED_HYBRID and features.num_gpus < 4:
            risks.append("Hybrid parallelism may not be effective with few GPUs")
        
        if strategy == StrategyType.DISTRIBUTED_PIPELINE and features.model_size_gb < 10:
            risks.append("Pipeline parallelism may have poor efficiency for smaller models")
        
        # Hardware risks
        if features.num_gpus > 8:
            risks.append("Large GPU count increases failure probability")
        
        return risks
    
    def add_training_data(self, features: PerformanceFeatures, metrics: OptimizationMetrics):
        """Add new training data point"""
        self.training_data.append((features, metrics))
        logger.debug(f"Added training data point, total: {len(self.training_data)}")
        
        # Retrain models if we have enough data
        if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
            self._train_models()
    
    def record_actual_performance(
        self,
        strategy: StrategyType,
        model_profile: ModelProfile,
        hardware_profile: HardwareProfile,
        evaluation_profile: EvaluationProfile,
        actual_metrics: OptimizationMetrics
    ):
        """Record actual performance for model training"""
        features = self._extract_features(strategy, model_profile, hardware_profile, evaluation_profile)
        self.add_training_data(features, actual_metrics)
    
    def _train_models(self):
        """Train ML models on accumulated data"""
        if len(self.training_data) < 5:
            logger.warning("Not enough training data to train models")
            return
        
        logger.info(f"Training performance models with {len(self.training_data)} data points")
        
        # Prepare training data
        X = []
        y_time = []
        y_memory = []
        y_cost = []
        y_success = []
        
        for features, metrics in self.training_data:
            X.append(features.to_vector())
            y_time.append(metrics.actual_eval_time_minutes or 0)
            y_memory.append(metrics.actual_memory_gb or 0)
            y_cost.append(metrics.actual_cost_usd or 0)
            y_success.append(metrics.success_rate)
        
        X = np.array(X)
        
        # Calculate feature statistics for normalization
        self.feature_stats = {
            'mean': np.mean(X, axis=0).tolist(),
            'std': np.std(X, axis=0).tolist()
        }
        
        # Train simple linear models (would use sklearn in production)
        self.time_model = self._train_linear_model(X, np.array(y_time))
        self.memory_model = self._train_linear_model(X, np.array(y_memory))
        self.cost_model = self._train_linear_model(X, np.array(y_cost))
        self.success_model = self._train_linear_model(X, np.array(y_success))
        
        # Save models
        self._save_models()
        
        logger.info("Performance models trained and saved")
    
    def _train_linear_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train simple linear regression model"""
        # Normalize features
        X_norm = self._normalize_features_array(X)
        
        # Add bias term
        X_bias = np.column_stack([np.ones(X_norm.shape[0]), X_norm])
        
        # Solve normal equations: w = (X^T X)^-1 X^T y
        try:
            XtX = X_bias.T @ X_bias
            Xty = X_bias.T @ y
            weights = np.linalg.solve(XtX, Xty)
            
            # Calculate R-squared
            y_pred = X_bias @ weights
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'weights': weights.tolist(),
                'r_squared': r_squared,
                'feature_count': X_norm.shape[1]
            }
        except np.linalg.LinAlgError:
            logger.warning("Linear algebra error in model training, using fallback")
            return {
                'weights': [0.0] * (X_norm.shape[1] + 1),
                'r_squared': 0.0,
                'feature_count': X_norm.shape[1]
            }
    
    def _predict_with_model(self, model: Dict[str, Any], features: List[float]) -> float:
        """Predict using trained linear model"""
        if not model or 'weights' not in model:
            return 0.0
        
        # Add bias term
        features_bias = [1.0] + features
        weights = model['weights']
        
        if len(features_bias) != len(weights):
            logger.warning("Feature dimension mismatch in prediction")
            return 0.0
        
        return max(0.0, sum(f * w for f, w in zip(features_bias, weights)))
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """Normalize features using stored statistics"""
        if not self.feature_stats:
            return features
        
        normalized = []
        for i, (feature, mean, std) in enumerate(zip(features, self.feature_stats['mean'], self.feature_stats['std'])):
            if std > 0:
                normalized.append((feature - mean) / std)
            else:
                normalized.append(0.0)
        
        return normalized
    
    def _normalize_features_array(self, X: np.ndarray) -> np.ndarray:
        """Normalize feature array"""
        if not self.feature_stats:
            # Use current data statistics
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (X - mean) / std
        else:
            mean = np.array(self.feature_stats['mean'])
            std = np.array(self.feature_stats['std'])
            std[std == 0] = 1
            return (X - mean) / std
    
    def _calculate_confidence(self, features: PerformanceFeatures) -> float:
        """Calculate prediction confidence based on training data similarity"""
        if len(self.training_data) < 3:
            return 0.6  # Low confidence with little data
        
        # Simple confidence based on data size and feature coverage
        data_confidence = min(1.0, len(self.training_data) / 50)  # More data = higher confidence
        
        # Check if we have similar examples in training data
        feature_vector = features.to_vector()
        similarities = []
        
        for train_features, _ in self.training_data[-20:]:  # Check last 20 examples
            train_vector = train_features.to_vector()
            # Simple cosine similarity
            similarity = self._cosine_similarity(feature_vector, train_vector)
            similarities.append(similarity)
        
        similarity_confidence = max(similarities) if similarities else 0.5
        
        # Combine confidences
        confidence = (data_confidence * 0.6 + similarity_confidence * 0.4)
        return max(0.3, min(0.95, confidence))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between feature vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _estimate_batch_size(self, model_profile: ModelProfile, hardware_profile: HardwareProfile) -> int:
        """Estimate reasonable batch size"""
        # Simple heuristic based on available memory
        available_memory = max(hardware_profile.available_gpu_memory_gb) if hardware_profile.available_gpu_memory_gb else 16.0
        memory_per_sample = model_profile.size_gb * 0.01  # Rough estimate
        
        if memory_per_sample > 0:
            batch_size = int(available_memory * 0.5 / memory_per_sample)  # Use 50% of memory
            return max(1, min(batch_size, 32))
        
        return 8  # Default batch size
    
    def _models_trained(self) -> bool:
        """Check if ML models are trained"""
        return (self.time_model is not None and 
                self.memory_model is not None and 
                self.cost_model is not None and 
                self.success_model is not None)
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_data = {
                'time_model': self.time_model,
                'memory_model': self.memory_model,
                'cost_model': self.cost_model,
                'success_model': self.success_model,
                'feature_stats': self.feature_stats,
                'training_data_count': len(self.training_data),
                'timestamp': time.time()
            }
            
            model_file = self.model_cache_dir / "performance_models.json"
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Saved performance models to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_file = self.model_cache_dir / "performance_models.json"
            if model_file.exists():
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
                
                self.time_model = model_data.get('time_model')
                self.memory_model = model_data.get('memory_model')
                self.cost_model = model_data.get('cost_model')
                self.success_model = model_data.get('success_model')
                self.feature_stats = model_data.get('feature_stats')
                
                logger.info(f"Loaded performance models from {model_file}")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about trained models"""
        stats = {
            'training_data_count': len(self.training_data),
            'models_trained': self._models_trained(),
            'feature_dimensions': len(PerformanceFeatures(0,0,0,0,0,0,0,0,0,0).to_vector())
        }
        
        if self._models_trained():
            stats.update({
                'time_model_r2': self.time_model.get('r_squared', 0),
                'memory_model_r2': self.memory_model.get('r_squared', 0),
                'cost_model_r2': self.cost_model.get('r_squared', 0),
                'success_model_r2': self.success_model.get('r_squared', 0)
            })
        
        return stats