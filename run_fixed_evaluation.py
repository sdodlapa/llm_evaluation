#!/usr/bin/env python3
"""
Improved Comprehensive Model Evaluation with All Fixes Applied
Runs evaluation on 7 new models with proper error handling and authentication
"""

import json
import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging for the evaluation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/comprehensive_evaluation_fixed_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    """Check environment setup and authentication"""
    logger = logging.getLogger(__name__)
    
    # Check HuggingFace token
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        logger.info("✓ HuggingFace token found")
    else:
        logger.warning("⚠ No HuggingFace token found - gated models may fail")
    
    # Check CUDA
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        logger.info(f"✓ CUDA devices: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    else:
        logger.warning("⚠ No CUDA_VISIBLE_DEVICES set")
    
    # Check cache directories
    cache_dirs = ['vllm_evaluation_cache', 'model_cache', 'logs']
    for cache_dir in cache_dirs:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"✓ Created cache directory: {cache_dir}")
    
    return True

def run_evaluation():
    """Run the comprehensive evaluation with all fixes"""
    logger = logging.getLogger(__name__)
    
    # Define new models to test (the ones that failed in original test)
    models_to_test = [
        "llama31_70b",           # Gated model - should work with auth fix
        "mixtral_8x7b",          # Gated model - should work with auth fix  
        "gemma2_27b",            # Gated model - should work with auth fix
        "deepseek_r1_distill_llama_70b",  # Should work with reasoning dataset fix
        "llama32_vision_90b",    # Multimodal - may need special handling
        "deepseek_r1_lite_8b",   # Standard model - should work
        "yi_lightning_7b"        # Standard model - should work
    ]
    
    results = {}
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE EVALUATION WITH FIXES APPLIED")
    logger.info("="*60)
    logger.info(f"Testing {len(models_to_test)} models:")
    for model in models_to_test:
        logger.info(f"  - {model}")
    logger.info("="*60)
    
    try:
        # Import evaluation modules
        from evaluation.category_evaluation import ModelEvaluator
        from evaluation.improved_discovery import load_configs_and_datasets
        
        logger.info("✓ Successfully imported evaluation modules")
        
        # Load configurations
        logger.info("Loading model configurations and datasets...")
        model_configs, dataset_configs = load_configs_and_datasets()
        logger.info(f"✓ Loaded {len(model_configs)} model configs")
        logger.info(f"✓ Loaded {len(dataset_configs)} dataset configs")
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        logger.info("✓ ModelEvaluator initialized")
        
        # Test each model
        for i, model_name in enumerate(models_to_test, 1):
            logger.info(f"\n[{i}/{len(models_to_test)}] Testing model: {model_name}")
            logger.info("-" * 40)
            
            try:
                if model_name not in model_configs:
                    logger.error(f"❌ Model {model_name} not found in configurations")
                    results[model_name] = {"status": "error", "error": "Model not found in configs"}
                    continue
                
                model_config = model_configs[model_name]
                logger.info(f"Model config: {model_config}")
                
                # Run evaluation for this model
                logger.info(f"Starting evaluation for {model_name}...")
                
                evaluation_result = evaluator.evaluate_model(
                    model_name=model_name,
                    model_config=model_config,
                    limit_samples=2,  # Small test to verify functionality
                    save_results=True
                )
                
                if evaluation_result:
                    logger.info(f"✅ {model_name} evaluation completed successfully")
                    results[model_name] = {
                        "status": "success", 
                        "categories": len(evaluation_result),
                        "result": evaluation_result
                    }
                else:
                    logger.error(f"❌ {model_name} evaluation returned None")
                    results[model_name] = {"status": "error", "error": "Evaluation returned None"}
                
            except Exception as e:
                logger.error(f"❌ Error evaluating {model_name}: {str(e)}")
                logger.exception("Full traceback:")
                results[model_name] = {"status": "error", "error": str(e)}
            
            logger.info(f"Completed {model_name} (Status: {results[model_name]['status']})")
    
    except Exception as e:
        logger.error(f"❌ Critical error in evaluation setup: {str(e)}")
        logger.exception("Full traceback:")
        return {"status": "critical_error", "error": str(e)}
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    error_count = len(results) - success_count
    
    logger.info(f"Total models tested: {len(results)}")
    logger.info(f"Successful evaluations: {success_count}")
    logger.info(f"Failed evaluations: {error_count}")
    
    for model_name, result in results.items():
        status_emoji = "✅" if result.get('status') == 'success' else "❌"
        logger.info(f"  {status_emoji} {model_name}: {result.get('status')}")
        if result.get('error'):
            logger.info(f"    Error: {result['error']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_results/fixed_evaluation_{timestamp}.json"
    os.makedirs("comprehensive_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_models": len(results),
            "successful": success_count,
            "failed": error_count,
            "results": results
        }, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {results_file}")
    logger.info("="*60)
    
    return results

def main():
    """Main function"""
    logger = setup_logging()
    
    logger.info("Starting comprehensive evaluation with fixes applied")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        return 1
    
    # Run evaluation
    results = run_evaluation()
    
    if isinstance(results, dict) and results.get('status') == 'critical_error':
        logger.error("Critical error occurred")
        return 1
    
    # Count successes
    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    total_count = len(results)
    
    logger.info(f"\nFinal result: {success_count}/{total_count} models evaluated successfully")
    
    if success_count > 0:
        logger.info("🎉 Evaluation completed with at least some successes!")
        return 0
    else:
        logger.error("😞 No models evaluated successfully")
        return 1

if __name__ == "__main__":
    exit(main())