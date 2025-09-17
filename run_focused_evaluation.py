#!/usr/bin/env python3
"""
CLEAN Focused Evaluation Script
- Only remaining datasets (6 unimplemented)
- Only optimal presets (balanced for both models)
- No redundancy, maximum efficiency
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from evaluation.enhanced_dataset_manager import EnhancedDatasetManager
from evaluation.simple_evaluator import evaluate_model
from evaluation.performance_monitor import LivePerformanceMonitor
from models.registry import create_model

def setup_logging():
    """Simple focused logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'focused_evaluation_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def get_focused_evaluation_plan():
    """Simple, clean evaluation plan"""
    return {
        "models": ["qwen3_8b", "qwen3_14b"],
        "preset": "balanced",  # Single optimal preset
        "remaining_datasets": [
            "math", "bfcl", "toolllama", 
            "mmlu", "ifeval", "winogrande"
        ],
        "samples_per_dataset": 200,  # Consistent 200 samples
        "total_combinations": 12    # 2 models × 1 preset × 6 datasets
    }

def run_single_evaluation(model_name, dataset_name, samples):
    """Clean single evaluation without complexity"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create model with balanced preset
        model = create_model(model_name, preset="balanced")
        if not model:
            return None, f"Failed to create model {model_name}"
        
        # Run evaluation with performance monitoring
        monitor = LivePerformanceMonitor(model_name, "balanced", dataset_name)
        monitor.start_monitoring()
        
        result = evaluate_model(
            model=model,
            dataset_name=dataset_name,
            samples=samples
        )
        
        performance_metrics = monitor.stop_monitoring()
        
        return {
            "model": model_name,
            "dataset": dataset_name,
            "samples_processed": len(samples),
            "evaluation_result": result,
            "performance_metrics": performance_metrics.to_dict() if performance_metrics else None,
            "timestamp": datetime.now().isoformat()
        }, None
        
    except Exception as e:
        return None, str(e)

def main():
    """Clean main execution"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 FOCUSED EVALUATION: Remaining Datasets Only")
    logger.info("=" * 50)
    
    plan = get_focused_evaluation_plan()
    dataset_manager = EnhancedDatasetManager()
    
    logger.info(f"Models: {plan['models']}")
    logger.info(f"Preset: {plan['preset']} (optimal)")
    logger.info(f"Datasets: {plan['remaining_datasets']}")
    logger.info(f"Total combinations: {plan['total_combinations']}")
    
    results = []
    completed = 0
    
    for model_name in plan["models"]:
        logger.info(f"\n🤖 Processing {model_name}")
        
        for dataset_name in plan["remaining_datasets"]:
            completed += 1
            logger.info(f"\n📊 [{completed}/{plan['total_combinations']}] {model_name} on {dataset_name}")
            
            try:
                # Check if dataset is implemented
                if dataset_name not in dataset_manager.get_implemented_datasets():
                    logger.warning(f"⚠️ {dataset_name} not implemented - creating placeholder")
                    
                    result = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "status": "not_implemented",
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    continue
                
                # Load dataset
                samples = dataset_manager.load_dataset(dataset_name, plan["samples_per_dataset"])
                logger.info(f"📋 Loaded {len(samples)} samples")
                
                # Run evaluation
                result, error = run_single_evaluation(model_name, dataset_name, samples)
                
                if result:
                    result["status"] = "success"
                    logger.info(f"✅ Completed successfully")
                else:
                    result = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "status": "error",
                        "error": error,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.error(f"❌ Failed: {error}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"💥 Unexpected error: {e}")
                result = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"focused_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "evaluation_plan": plan,
            "results": results,
            "summary": {
                "total_attempted": len(results),
                "successful": len([r for r in results if r.get("status") == "success"]),
                "failed": len([r for r in results if r.get("status") == "error"]),
                "not_implemented": len([r for r in results if r.get("status") == "not_implemented"])
            }
        }, f, indent=2)
    
    logger.info(f"\n🏁 Focused evaluation completed!")
    logger.info(f"📁 Results saved to: {results_file}")
    
    # Print summary
    successful = len([r for r in results if r.get("status") == "success"])
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {len(results) - successful}")

if __name__ == "__main__":
    main()