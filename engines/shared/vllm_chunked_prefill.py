"""
vLLM Chunked Prefill Integration for Enhanced AOT Compilation

This module provides clean integration of vLLM's built-in chunked prefill
with our enhanced AOT compilation system. It automatically enables chunked
prefill only for sequences that benefit from it.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import vLLM components
from vllm import EngineArgs
from vllm.engine.llm_engine import LLMEngine

# Import our enhanced compilation components
from .enhanced_aot_compiler import EnhancedAOTModelCompiler
from .performance_monitor import performance_monitor, monitor_compilation

logger = logging.getLogger(__name__)

@dataclass
class VLLMChunkedPrefillConfig:
    """Configuration for vLLM's built-in chunked prefill optimization"""
    
    # Core chunked prefill settings
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 2048  # Tokens to trigger chunking
    max_num_partial_prefills: int = 4         # Max chunks per request
    max_long_partial_prefills: int = 2        # Max long prefills in batch
    
    # Additional performance settings
    disable_chunked_mm_input: bool = False    # Keep chunked matrix mult
    auto_detect_threshold: bool = True        # Auto-adjust threshold based on data
    
    def to_engine_args(self) -> Dict[str, Any]:
        """Convert to vLLM EngineArgs parameters"""
        return {
            'enable_chunked_prefill': self.enable_chunked_prefill,
            'long_prefill_token_threshold': self.long_prefill_token_threshold,
            'max_num_partial_prefills': self.max_num_partial_prefills,
            'max_long_partial_prefills': self.max_long_partial_prefills,
            'disable_chunked_mm_input': self.disable_chunked_mm_input
        }

class VLLMChunkedPrefillOptimizer:
    """
    Clean wrapper around vLLM's built-in chunked prefill functionality
    
    This class provides intelligent threshold detection and seamless integration
    with our existing enhanced AOT compilation system.
    """
    
    def __init__(self, config: VLLMChunkedPrefillConfig = None):
        self.config = config or VLLMChunkedPrefillConfig()
        self.stats = {
            "sequences_analyzed": 0,
            "chunked_sequences": 0,
            "threshold_adjustments": 0,
            "optimal_threshold": self.config.long_prefill_token_threshold
        }
        
        logger.info(f"Initialized vLLM chunked prefill optimizer")
        logger.info(f"  Threshold: {self.config.long_prefill_token_threshold} tokens")
        logger.info(f"  Max chunks per request: {self.config.max_num_partial_prefills}")
    
    def analyze_dataset_for_optimal_threshold(self, dataset_stats: List[Dict]) -> int:
        """
        Analyze dataset statistics to determine optimal chunking threshold
        
        Args:
            dataset_stats: List of dataset statistics with max_tokens_est
            
        Returns:
            Optimal threshold in tokens
        """
        if not self.config.auto_detect_threshold:
            return self.config.long_prefill_token_threshold
        
        # Extract max token estimates from all datasets
        max_tokens = [
            stats.get('max_tokens_est', 0) 
            for stats in dataset_stats 
            if 'max_tokens_est' in stats
        ]
        
        if not max_tokens:
            return self.config.long_prefill_token_threshold
        
        # Calculate statistics
        import statistics
        max_overall = max(max_tokens)
        percentile_95 = statistics.quantiles(max_tokens, n=20)[18]  # 95th percentile
        
        # Set threshold based on data characteristics
        if max_overall < 1000:
            # Short sequences - disable chunking
            optimal_threshold = max_overall + 1000  # Never trigger
            logger.info("Dataset has only short sequences - chunked prefill will be disabled")
        elif percentile_95 < 2048:
            # Most sequences short, few long ones - conservative threshold
            optimal_threshold = 2048
            logger.info("Mixed sequence lengths - using conservative 2K token threshold")
        else:
            # Many long sequences - aggressive chunking
            optimal_threshold = min(2048, int(percentile_95 * 0.8))
            logger.info(f"Many long sequences detected - using aggressive {optimal_threshold} token threshold")
        
        # Update configuration
        if optimal_threshold != self.config.long_prefill_token_threshold:
            self.stats["threshold_adjustments"] += 1
            self.stats["optimal_threshold"] = optimal_threshold
            self.config.long_prefill_token_threshold = optimal_threshold
            
            logger.info(f"Auto-adjusted chunking threshold to {optimal_threshold} tokens")
        
        return optimal_threshold
    
    def should_enable_for_model(self, example_inputs, dataset_info=None) -> bool:
        """
        Determine if chunked prefill should be enabled for a specific model
        
        Args:
            example_inputs: Example input tensors
            dataset_info: Optional dataset statistics
            
        Returns:
            Whether to enable chunked prefill for this model
        """
        max_input_length = 0
        
        # Check example input lengths
        for inp in example_inputs:
            if hasattr(inp, '__len__'):
                max_input_length = max(max_input_length, len(inp))
        
        # Check dataset statistics if provided
        if dataset_info and 'max_tokens_est' in dataset_info:
            max_input_length = max(max_input_length, dataset_info['max_tokens_est'])
        
        should_enable = max_input_length > self.config.long_prefill_token_threshold
        
        logger.debug(f"Chunked prefill decision: max_length={max_input_length}, "
                    f"threshold={self.config.long_prefill_token_threshold}, "
                    f"enable={should_enable}")
        
        return should_enable
    
    def get_vllm_engine_config(self, base_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get vLLM engine configuration with chunked prefill settings
        
        Args:
            base_config: Base engine configuration to extend
            
        Returns:
            Complete engine configuration with chunked prefill
        """
        config = base_config.copy() if base_config else {}
        
        # Add chunked prefill parameters
        chunked_params = self.config.to_engine_args()
        config.update(chunked_params)
        
        logger.info(f"vLLM engine config with chunked prefill: {chunked_params}")
        
        return config
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.stats,
            "current_threshold": self.config.long_prefill_token_threshold,
            "chunking_enabled": self.config.enable_chunked_prefill
        }

class EnhancedAOTWithVLLMChunking(EnhancedAOTModelCompiler):
    """
    Enhanced AOT compiler with vLLM's built-in chunked prefill integration
    
    This class seamlessly integrates vLLM's chunked prefill with our enhanced
    compilation system, providing automatic optimization for long sequences.
    """
    
    def __init__(self, 
                 base_compiler=None,
                 graph_config=None,
                 vllm_chunking_config: VLLMChunkedPrefillConfig = None):
        super().__init__(base_compiler, graph_config)
        
        self.vllm_chunking = VLLMChunkedPrefillOptimizer(vllm_chunking_config)
        self.model_chunking_decisions: Dict[str, bool] = {}
        
        logger.info("Initialized Enhanced AOT with vLLM chunked prefill integration")
    
    @monitor_compilation
    def compile_model(self, 
                     model, 
                     example_inputs, 
                     model_id: str,
                     dataset_info: Dict = None,
                     **compile_kwargs) -> Any:
        """
        Compile model with automatic vLLM chunked prefill optimization
        
        Args:
            model: PyTorch model to compile
            example_inputs: Example input tensors
            model_id: Unique model identifier
            dataset_info: Optional dataset statistics for threshold optimization
            **compile_kwargs: Additional compilation arguments
            
        Returns:
            Compiled model with optimal chunked prefill configuration
        """
        
        # Determine if chunked prefill should be enabled for this model
        should_chunk = self.vllm_chunking.should_enable_for_model(
            example_inputs, dataset_info
        )
        
        self.model_chunking_decisions[model_id] = should_chunk
        
        if should_chunk:
            logger.info(f"Enabling vLLM chunked prefill for model {model_id}")
            
            # Add vLLM chunked prefill configuration to compile kwargs
            vllm_config = self.vllm_chunking.get_vllm_engine_config()
            compile_kwargs.update(vllm_config)
        else:
            logger.info(f"Chunked prefill not beneficial for model {model_id} (sequences too short)")
        
        # Use standard enhanced compilation with chunked prefill config
        compiled_model = super().compile_model(
            model, example_inputs, model_id, **compile_kwargs
        )
        
        # Wrap with chunking statistics tracking
        if should_chunk and compiled_model:
            compiled_model = VLLMChunkedModelWrapper(
                compiled_model, 
                self.vllm_chunking,
                model_id
            )
        
        return compiled_model
    
    def auto_optimize_for_datasets(self, dataset_stats: List[Dict]) -> None:
        """
        Automatically optimize chunked prefill settings based on dataset analysis
        
        Args:
            dataset_stats: List of dataset statistics from evaluation data
        """
        logger.info("Auto-optimizing vLLM chunked prefill for dataset characteristics")
        
        optimal_threshold = self.vllm_chunking.analyze_dataset_for_optimal_threshold(
            dataset_stats
        )
        
        logger.info(f"Optimal chunking threshold determined: {optimal_threshold} tokens")
    
    def get_chunking_report(self) -> Dict[str, Any]:
        """Get comprehensive chunked prefill optimization report"""
        
        return {
            "vllm_chunking_stats": self.vllm_chunking.get_statistics(),
            "model_decisions": self.model_chunking_decisions,
            "models_with_chunking": sum(self.model_chunking_decisions.values()),
            "total_models": len(self.model_chunking_decisions)
        }

class VLLMChunkedModelWrapper:
    """Wrapper that tracks vLLM chunked prefill usage statistics"""
    
    def __init__(self, compiled_model, chunking_optimizer, model_id):
        self.compiled_model = compiled_model
        self.chunking_optimizer = chunking_optimizer
        self.model_id = model_id
        self.execution_stats = {
            "total_inferences": 0,
            "long_sequence_inferences": 0,
            "chunking_triggered": 0
        }
    
    def __call__(self, *args, **kwargs):
        """Execute with chunking statistics tracking"""
        self.execution_stats["total_inferences"] += 1
        
        # Check if this inference likely triggered chunking
        if args and hasattr(args[0], '__len__'):
            input_length = len(args[0])
            if input_length > self.chunking_optimizer.config.long_prefill_token_threshold:
                self.execution_stats["long_sequence_inferences"] += 1
                self.execution_stats["chunking_triggered"] += 1
        
        return self.compiled_model(*args, **kwargs)
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get chunked prefill execution statistics"""
        total = self.execution_stats["total_inferences"]
        return {
            **self.execution_stats,
            "chunking_rate": (
                self.execution_stats["chunking_triggered"] / total 
                if total > 0 else 0.0
            )
        }
    
    # Delegate all other attributes to compiled model
    def __getattr__(self, name):
        return getattr(self.compiled_model, name)

def create_vllm_enhanced_compiler(
    enable_chunked_prefill: bool = True,
    chunking_threshold: int = None,  # Auto-detect if None
    max_chunks_per_request: int = 4,
    **other_kwargs
) -> EnhancedAOTWithVLLMChunking:
    """
    Factory function to create enhanced compiler with vLLM chunked prefill
    
    Args:
        enable_chunked_prefill: Whether to enable vLLM's chunked prefill
        chunking_threshold: Token threshold (auto-detected if None)
        max_chunks_per_request: Maximum chunks per request
        **other_kwargs: Other arguments for enhanced compiler
        
    Returns:
        Enhanced compiler with vLLM chunked prefill integration
    """
    
    # Create chunked prefill configuration
    chunking_config = VLLMChunkedPrefillConfig(
        enable_chunked_prefill=enable_chunked_prefill,
        long_prefill_token_threshold=chunking_threshold or 2048,
        max_num_partial_prefills=max_chunks_per_request,
        auto_detect_threshold=(chunking_threshold is None)
    )
    
    return EnhancedAOTWithVLLMChunking(
        vllm_chunking_config=chunking_config,
        **other_kwargs
    )