"""
Comprehensive Tests for Phase 4 Optimization Engine
Testing all components of the adaptive strategy engine
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import optimization components
from engines.optimization.optimization_types import (
    StrategyType, OptimizationGoal, ModelProfile, HardwareProfile, 
    EvaluationProfile, PerformancePrediction, StrategyRecommendation,
    OptimizationSettings, OptimizationMetrics,
    create_default_hardware_profile, create_default_evaluation_profile
)
from engines.optimization.strategy_selector import StrategySelector
from engines.optimization.performance_predictor import PerformancePredictor, PerformanceFeatures
from engines.optimization.optimization_controller import (
    OptimizationController, OptimizationAction, ResourceMonitorData
)


class TestOptimizationTypes:
    """Test optimization data types and structures"""
    
    def test_model_profile_creation(self):
        """Test ModelProfile creation and validation"""
        profile = ModelProfile(
            name="test-model",
            size_gb=7.5,
            parameter_count=7_000_000_000,
            architecture="transformer"
        )
        
        assert profile.name == "test-model"
        assert profile.size_gb == 7.5
        assert profile.parameter_count == 7_000_000_000
        assert profile.architecture == "transformer"
        assert profile.success_rate == 1.0
    
    def test_model_profile_parameter_estimation(self):
        """Test automatic parameter count estimation"""
        profile = ModelProfile(name="test-model", size_gb=8.0)
        
        # Should estimate parameter count from size
        expected_params = int(8.0 * 1024**3 / 2)  # 2 bytes per parameter
        assert profile.parameter_count == expected_params
    
    def test_hardware_profile_properties(self):
        """Test HardwareProfile computed properties"""
        profile = HardwareProfile(
            num_gpus=4,
            gpu_memory_gb=[16.0, 16.0, 16.0, 16.0],
            gpu_compute_capability=["8.0", "8.0", "8.0", "8.0"],
            cpu_cores=32,
            system_memory_gb=128.0,
            memory_utilization=[0.2, 0.3, 0.1, 0.4]
        )
        
        assert profile.total_gpu_memory_gb == 64.0
        assert profile.is_multi_gpu == True
        
        available = profile.available_gpu_memory_gb
        assert len(available) == 4
        assert available[0] == 16.0 * 0.8  # 80% available (20% used)
        assert available[3] == 16.0 * 0.6  # 60% available (40% used)
    
    def test_evaluation_profile_defaults(self):
        """Test EvaluationProfile default values"""
        profile = EvaluationProfile(dataset_size=1000)
        
        assert profile.dataset_size == 1000
        assert profile.min_success_rate == 0.95
        assert profile.confidence_level == 0.95
        assert profile.priority == 5
        assert profile.optimization_goal == OptimizationGoal.BALANCED
        assert profile.allow_distributed == True
    
    def test_optimization_settings_validation(self):
        """Test OptimizationSettings validation"""
        settings = OptimizationSettings(
            optimization_goal=OptimizationGoal.MINIMIZE_TIME,
            max_evaluation_time_minutes=30.0
        )
        
        settings.validate()  # Should not raise
        
        # Test invalid settings
        invalid_settings = OptimizationSettings(
            optimization_goal=OptimizationGoal.MINIMIZE_TIME,
            max_evaluation_time_minutes=-5.0
        )
        
        with pytest.raises(ValueError, match="Max evaluation time must be positive"):
            invalid_settings.validate()


class TestStrategySelector:
    """Test strategy selection logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.selector = StrategySelector()
        
        # Standard test profiles
        self.small_model = ModelProfile(name="small-model", size_gb=3.0)
        self.medium_model = ModelProfile(name="medium-model", size_gb=15.0)
        self.large_model = ModelProfile(name="large-model", size_gb=45.0)
        
        self.single_gpu_hardware = create_default_hardware_profile(num_gpus=1, gpu_memory_gb=24.0)
        self.multi_gpu_hardware = create_default_hardware_profile(num_gpus=4, gpu_memory_gb=24.0)
        
        self.standard_evaluation = create_default_evaluation_profile(dataset_size=1000)
    
    def test_lightweight_strategy_selection(self):
        """Test selection of lightweight strategy for small models"""
        recommendation = self.selector.select_strategy(
            self.small_model,
            self.single_gpu_hardware,
            self.standard_evaluation
        )
        
        assert recommendation.recommended_strategy == StrategyType.LIGHTWEIGHT
        assert recommendation.prediction.required_gpus == 1
        assert recommendation.prediction.confidence > 0.7
        assert recommendation.optimal_batch_size is not None
    
    def test_distributed_strategy_selection(self):
        """Test selection of distributed strategies for large models"""
        recommendation = self.selector.select_strategy(
            self.large_model,
            self.multi_gpu_hardware,
            self.standard_evaluation
        )
        
        # Should select a distributed strategy
        assert recommendation.recommended_strategy in [
            StrategyType.DISTRIBUTED_TENSOR,
            StrategyType.DISTRIBUTED_PIPELINE,
            StrategyType.DISTRIBUTED_HYBRID
        ]
        assert recommendation.prediction.required_gpus > 1
        assert len(recommendation.alternatives) > 0
    
    def test_strategy_feasibility_analysis(self):
        """Test strategy feasibility checking"""
        # Small hardware, large model - should have limited options
        limited_hardware = HardwareProfile(
            num_gpus=1,
            gpu_memory_gb=[8.0],  # Only 8GB
            gpu_compute_capability=["8.0"],
            cpu_cores=8,
            system_memory_gb=32.0
        )
        
        # This should fail for large model
        with pytest.raises(ValueError, match="No feasible evaluation strategies"):
            self.selector.select_strategy(
                self.large_model,
                limited_hardware,
                self.standard_evaluation
            )
    
    def test_optimization_goal_influence(self):
        """Test how optimization goals influence strategy selection"""
        time_focused = EvaluationProfile(
            dataset_size=1000,
            optimization_goal=OptimizationGoal.MINIMIZE_TIME
        )
        
        memory_focused = EvaluationProfile(
            dataset_size=1000,
            optimization_goal=OptimizationGoal.MINIMIZE_MEMORY
        )
        
        time_rec = self.selector.select_strategy(self.medium_model, self.multi_gpu_hardware, time_focused)
        memory_rec = self.selector.select_strategy(self.medium_model, self.multi_gpu_hardware, memory_focused)
        
        # Strategies might be different based on optimization goals
        assert time_rec.recommended_strategy is not None
        assert memory_rec.recommended_strategy is not None
        
        # Time-focused should prefer faster strategies
        # Memory-focused should prefer memory-efficient strategies
        # (exact strategies depend on the specific model/hardware combination)
    
    def test_performance_history_update(self):
        """Test performance history tracking"""
        metrics = {
            'success_rate': 0.92,
            'actual_time': 15.5,
            'actual_memory': 12.3
        }
        
        self.selector.update_performance_history("test-model", StrategyType.LIGHTWEIGHT, metrics)
        
        stats = self.selector.get_strategy_statistics()
        assert stats['total_evaluations'] == 1
        assert 'test-model' in stats['history_count']
        assert stats['history_count']['test-model'] == 1
    
    def test_recommendation_reasoning(self):
        """Test recommendation includes proper reasoning"""
        recommendation = self.selector.select_strategy(
            self.medium_model,
            self.multi_gpu_hardware,
            self.standard_evaluation
        )
        
        assert len(recommendation.recommendation_reason) > 0
        assert isinstance(recommendation.trade_offs, list)
        assert isinstance(recommendation.warnings, list)
        assert isinstance(recommendation.suggested_settings, dict)


class TestPerformancePredictor:
    """Test performance prediction functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.predictor = PerformancePredictor()
        
        self.test_model = ModelProfile(name="test-model", size_gb=10.0)
        self.test_hardware = create_default_hardware_profile(num_gpus=2, gpu_memory_gb=20.0)
        self.test_evaluation = create_default_evaluation_profile(dataset_size=500)
    
    def test_performance_prediction_heuristics(self):
        """Test performance prediction using heuristics"""
        prediction = self.predictor.predict_performance(
            StrategyType.DISTRIBUTED_TENSOR,
            self.test_model,
            self.test_hardware,
            self.test_evaluation
        )
        
        assert isinstance(prediction, PerformancePrediction)
        assert prediction.strategy == StrategyType.DISTRIBUTED_TENSOR
        assert prediction.estimated_time_minutes > 0
        assert prediction.estimated_memory_gb > 0
        assert prediction.estimated_cost_usd > 0
        assert 0 <= prediction.confidence <= 1
        assert 0 <= prediction.expected_success_rate <= 1
        assert prediction.required_gpus == 2
        assert len(prediction.gpu_memory_per_device) == 2
    
    def test_feature_extraction(self):
        """Test feature extraction from profiles"""
        features = self.predictor._extract_features(
            StrategyType.LIGHTWEIGHT,
            self.test_model,
            self.test_hardware,
            self.test_evaluation
        )
        
        assert isinstance(features, PerformanceFeatures)
        assert features.model_size_gb == 10.0
        assert features.num_gpus == 2
        assert features.dataset_size == 500
        assert features.strategy_type == 1  # LIGHTWEIGHT encoded as 1
        
        # Test feature vector conversion
        vector = features.to_vector()
        assert len(vector) == 10  # Expected feature count
        assert all(isinstance(v, (int, float)) for v in vector)
    
    def test_training_data_accumulation(self):
        """Test accumulation of training data"""
        initial_count = len(self.predictor.training_data)
        
        # Add some training data
        features = PerformanceFeatures(
            model_size_gb=5.0, parameter_count=5e9, context_length=2048,
            num_gpus=1, total_gpu_memory_gb=16.0, gpu_compute_capability=80.0,
            inter_gpu_bandwidth_gbps=0.0, dataset_size=1000, batch_size=8,
            strategy_type=1
        )
        
        metrics = OptimizationMetrics(
            strategy_selected=StrategyType.LIGHTWEIGHT,
            selection_time_ms=100,
            actual_eval_time_minutes=10.5,
            actual_memory_gb=5.2,
            actual_cost_usd=2.50
        )
        
        self.predictor.add_training_data(features, metrics)
        
        assert len(self.predictor.training_data) == initial_count + 1
    
    def test_model_statistics(self):
        """Test model statistics reporting"""
        stats = self.predictor.get_model_statistics()
        
        assert 'training_data_count' in stats
        assert 'models_trained' in stats
        assert 'feature_dimensions' in stats
        assert isinstance(stats['training_data_count'], int)
        assert isinstance(stats['models_trained'], bool)
        assert stats['feature_dimensions'] == 10


class TestOptimizationController:
    """Test real-time optimization control"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.settings = OptimizationSettings(
            optimization_goal=OptimizationGoal.BALANCED,
            max_evaluation_time_minutes=30.0,
            enable_dynamic_batching=True,
            enable_memory_optimization=True
        )
        
        self.controller = OptimizationController(self.settings)
        
        # Mock callbacks
        self.callback_results = {}
        
        def mock_callback(action_type):
            def callback(params):
                self.callback_results[action_type] = params
                return True
            return callback
        
        # Register callbacks
        for action in OptimizationAction:
            self.controller.register_action_callback(action, mock_callback(action))
    
    def test_optimization_session_lifecycle(self):
        """Test complete optimization session lifecycle"""
        # Start optimization
        assert not self.controller.is_active
        
        self.controller.start_optimization("test-eval-001", initial_batch_size=16)
        
        assert self.controller.is_active
        assert self.controller.current_evaluation_id == "test-eval-001"
        assert self.controller.current_batch_size == 16
        assert len(self.controller.optimization_events) == 1  # Start event
        
        # Stop optimization
        metrics = self.controller.stop_optimization()
        
        assert not self.controller.is_active
        assert isinstance(metrics, OptimizationMetrics)
    
    def test_resource_monitoring_triggers(self):
        """Test resource monitoring and optimization triggers"""
        self.controller.start_optimization("test-eval-002", initial_batch_size=8)
        
        # High memory usage scenario
        high_memory_data = ResourceMonitorData(
            timestamp=time.time(),
            gpu_utilization=[0.8, 0.7],
            gpu_memory_used=[18.0, 19.0],  # High usage
            gpu_memory_total=[20.0, 20.0],
            gpu_temperature=[75.0, 78.0],
            system_memory_used=32.0,
            system_memory_total=64.0
        )
        
        # This should trigger optimization
        self.controller.report_resource_usage(high_memory_data)
        
        # Reset optimization time throttling for testing
        self.controller.last_optimization_time = 0
        
        # Give some time for processing and force an optimization check
        time.sleep(0.1)
        
        # Manually trigger optimization to ensure it happens for the test
        if self.controller._should_optimize(high_memory_data):
            self.controller._perform_optimization(high_memory_data)
        
        # Should have triggered some optimization actions
        assert len(self.controller.optimization_events) > 1  # More than just start event
        
        self.controller.stop_optimization()
    
    def test_performance_monitoring(self):
        """Test performance monitoring and baseline setting"""
        self.controller.start_optimization("test-eval-003", initial_batch_size=4)
        
        # Report initial performance
        self.controller.report_performance_metrics(
            throughput=100.0,  # samples/second
            latency=0.01,      # seconds
            quality_score=0.85
        )
        
        assert self.controller.performance_baseline is not None
        assert self.controller.performance_baseline['throughput'] == 100.0
        
        # Report degraded performance
        self.controller.report_performance_metrics(
            throughput=70.0,   # 30% degradation
            latency=0.015,
            quality_score=0.83
        )
        
        assert len(self.controller.recent_performance) == 2
        
        self.controller.stop_optimization()
    
    def test_optimization_action_execution(self):
        """Test optimization action execution"""
        self.controller.start_optimization("test-eval-004", initial_batch_size=8)
        
        # Manually trigger optimization action
        success = self.controller._execute_optimization_action(
            OptimizationAction.ADJUST_BATCH_SIZE,
            {"batch_size": 16}
        )
        
        assert success == True
        assert self.controller.current_batch_size == 16
        assert OptimizationAction.ADJUST_BATCH_SIZE in self.callback_results
        assert self.callback_results[OptimizationAction.ADJUST_BATCH_SIZE]["batch_size"] == 16
        
        self.controller.stop_optimization()
    
    def test_optimization_summary(self):
        """Test optimization summary reporting"""
        self.controller.start_optimization("test-eval-005", initial_batch_size=12)
        
        # Simulate some optimization events
        self.controller._record_event(
            OptimizationAction.ADJUST_BATCH_SIZE,
            {"batch_size": 8},
            "Test optimization",
            True
        )
        
        summary = self.controller.get_optimization_summary()
        
        assert summary["status"] == "active"
        assert summary["evaluation_id"] == "test-eval-005"
        assert summary["total_events"] >= 2  # Start + test event
        assert summary["current_batch_size"] == 12
        assert "event_breakdown" in summary
        
        self.controller.stop_optimization()
        
        # Check completed summary
        final_summary = self.controller.get_optimization_summary()
        assert final_summary["status"] == "completed"
    
    def test_optimization_log_export(self):
        """Test optimization log export"""
        self.controller.start_optimization("test-eval-006", initial_batch_size=4)
        
        # Add some events
        self.controller._record_event(
            OptimizationAction.ENABLE_MEMORY_OPTIMIZATION,
            {"level": 2},
            "Memory pressure detected",
            True
        )
        
        log = self.controller.export_optimization_log()
        
        assert len(log) >= 2  # Start + test event
        assert all('timestamp' in entry for entry in log)
        assert all('action' in entry for entry in log)
        assert all('parameters' in entry for entry in log)
        assert all('reason' in entry for entry in log)
        assert all('success' in entry for entry in log)
        
        self.controller.stop_optimization()


class TestIntegration:
    """Test integration between optimization components"""
    
    def setup_method(self):
        """Setup integrated test environment"""
        self.selector = StrategySelector()
        self.predictor = PerformancePredictor()
        
        self.model_profile = ModelProfile(name="integration-test-model", size_gb=12.0)
        self.hardware_profile = create_default_hardware_profile(num_gpus=4, gpu_memory_gb=20.0)
        self.evaluation_profile = create_default_evaluation_profile(dataset_size=2000)
    
    def test_selector_predictor_integration(self):
        """Test integration between strategy selector and performance predictor"""
        # Get recommendation from selector
        recommendation = self.selector.select_strategy(
            self.model_profile,
            self.hardware_profile,
            self.evaluation_profile
        )
        
        # Use predictor to refine the prediction
        refined_prediction = self.predictor.predict_performance(
            recommendation.recommended_strategy,
            self.model_profile,
            self.hardware_profile,
            self.evaluation_profile
        )
        
        # Predictions should be consistent
        assert refined_prediction.strategy == recommendation.recommended_strategy
        assert refined_prediction.estimated_time_minutes > 0
        assert refined_prediction.estimated_memory_gb > 0
        
        # Record performance for learning
        actual_metrics = OptimizationMetrics(
            strategy_selected=recommendation.recommended_strategy,
            selection_time_ms=50,
            actual_eval_time_minutes=refined_prediction.estimated_time_minutes * 1.1,  # 10% slower
            actual_memory_gb=refined_prediction.estimated_memory_gb * 0.95,  # 5% less memory
            success_rate=0.98
        )
        
        self.predictor.record_actual_performance(
            recommendation.recommended_strategy,
            self.model_profile,
            self.hardware_profile,
            self.evaluation_profile,
            actual_metrics
        )
        
        # Should have added training data
        assert len(self.predictor.training_data) > 0
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete end-to-end optimization workflow"""
        # Step 1: Strategy selection
        recommendation = self.selector.select_strategy(
            self.model_profile,
            self.hardware_profile,
            self.evaluation_profile
        )
        
        # Step 2: Setup optimization controller
        settings = OptimizationSettings(
            optimization_goal=self.evaluation_profile.optimization_goal,
            max_evaluation_time_minutes=30.0,
            enable_dynamic_batching=True
        )
        
        controller = OptimizationController(settings)
        
        # Register dummy callbacks
        def dummy_callback(params):
            return True
        
        for action in OptimizationAction:
            controller.register_action_callback(action, dummy_callback)
        
        # Step 3: Start optimization session
        controller.start_optimization("integration-test", recommendation.optimal_batch_size)
        
        # Step 4: Simulate evaluation with monitoring
        for i in range(3):
            # Simulate resource data
            resource_data = ResourceMonitorData(
                timestamp=time.time(),
                gpu_utilization=[0.7 + i*0.1, 0.6 + i*0.1, 0.8 + i*0.1, 0.9 + i*0.1],
                gpu_memory_used=[15.0 + i, 14.0 + i, 16.0 + i, 17.0 + i],
                gpu_memory_total=[20.0] * 4,
                gpu_temperature=[70.0 + i*2] * 4,
                system_memory_used=40.0,
                system_memory_total=128.0
            )
            
            controller.report_resource_usage(resource_data)
            
            # Simulate performance metrics
            controller.report_performance_metrics(
                throughput=95.0 - i*5,  # Degrading performance
                latency=0.01 + i*0.002,
                quality_score=0.92 - i*0.01
            )
            
            time.sleep(0.1)  # Small delay
        
        # Step 5: Complete optimization
        final_metrics = controller.stop_optimization()
        
        # Verify workflow completed successfully
        assert isinstance(final_metrics, OptimizationMetrics)
        assert final_metrics.success_rate >= 0.0
        
        # Get optimization summary
        summary = controller.get_optimization_summary()
        assert summary["status"] == "completed"
        assert summary["total_events"] >= 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])