#!/usr/bin/env python3
"""
Focused test to verify vLLM configuration fixes.
"""

import sys
import os
sys.path.append('/home/sdodl001_odu_edu/llm_evaluation')

import logging
from pathlib import Path
from models.registry import ModelRegistry
from configs.model_configs import MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vllm_configs():
    """Test that vLLM configurations are compatible."""
    
    logger.info("🧪 Testing vLLM configuration compatibility...")
    
    # Test configurations to validate
    test_configs = [
        ("qwen3_8b", "memory_optimized"),  # Previously failed with block_size=8
        ("qwen2.5_14b", "performance"),   # Previously failed with use_v2_block_manager
    ]
    
    registry = ModelRegistry()
    
    for model_name, preset in test_configs:
        logger.info(f"\n🔍 Testing {model_name} with {preset} preset...")
        
        try:
            # Create model instance (this validates config)
            model = registry.create_model(model_name, preset=preset)
            logger.info(f"✅ {model_name}/{preset} - Configuration valid")
            
            # Check specific config values
            config = MODEL_CONFIGS[model_name]["presets"][preset]
            
            # Verify block_size is multiple of 16
            if "block_size" in config:
                block_size = config["block_size"]
                if block_size % 16 != 0:
                    logger.error(f"❌ Block size {block_size} not multiple of 16")
                else:
                    logger.info(f"✅ Block size {block_size} is valid")
            
            # Verify no deprecated parameters
            deprecated_params = ["use_v2_block_manager"]
            for param in deprecated_params:
                if param in config:
                    logger.error(f"❌ Deprecated parameter found: {param}")
                else:
                    logger.info(f"✅ No deprecated parameter: {param}")
            
        except Exception as e:
            logger.error(f"❌ {model_name}/{preset} - Configuration error: {e}")
    
    logger.info("\n✅ vLLM configuration testing completed!")

if __name__ == "__main__":
    test_vllm_configs()
