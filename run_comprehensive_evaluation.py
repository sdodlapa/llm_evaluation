#!/usr/bin/env python3
"""
Main script for running comprehensive LLM evaluation
Tests all models across all datasets with live performance monitoring
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from evaluation.comprehensive_runner import ComprehensiveEvaluationRunner
from evaluation.enhanced_dataset_manager import EnhancedDatasetManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('comprehensive_evaluation.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive LLM Evaluation with Live Performance Monitoring")
    
    parser.add_argument("--mode", choices=["validate", "quick", "comprehensive"], default="validate",
                       help="Evaluation mode: validate (single test), quick (subset), comprehensive (all)")
    
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to test (default: all)")
    
    parser.add_argument("--presets", nargs="+", 
                       choices=["memory_optimized", "balanced", "performance"],
                       default=["memory_optimized", "balanced", "performance"],
                       help="Model presets to test")
    
    parser.add_argument("--datasets", nargs="+",
                       help="Specific datasets to test (default: all implemented)")
    
    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples per dataset (default: recommended counts)")
    
    parser.add_argument("--output-dir", default="comprehensive_results",
                       help="Output directory for results")
    
    parser.add_argument("--continue-on-failure", action="store_true", default=True,
                       help="Continue evaluation even if some runs fail")
    
    parser.add_argument("--show-datasets", action="store_true",
                       help="Show available datasets and exit")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Show datasets if requested
    if args.show_datasets:
        dataset_manager = EnhancedDatasetManager()
        summary = dataset_manager.get_dataset_summary()
        
        print("\\n📊 Available Datasets:")
        print(f"Total: {summary['total_datasets']}")
        print(f"Implemented: {summary['implemented_datasets']}")
        print(f"Unimplemented: {summary['unimplemented_datasets']}")
        
        print("\\n📋 Dataset Details:")
        for name, info in summary['datasets'].items():
            status = "✅" if info['implemented'] else "❌"
            print(f"  {status} {name}: {info['task_type']} ({info['sample_count']} samples) - {info['description']}")
        
        print("\\n🔧 Task Type Distribution:")
        for task_type, count in summary['task_type_distribution'].items():
            print(f"  {task_type}: {count} datasets")
        
        return
    
    # Create runner
    runner = ComprehensiveEvaluationRunner(args.output_dir)
    
    # Determine samples per dataset
    samples_per_dataset = None
    if args.samples:
        # Use same number for all datasets
        dataset_manager = EnhancedDatasetManager()
        datasets = args.datasets if args.datasets else dataset_manager.get_implemented_datasets()
        samples_per_dataset = {dataset: args.samples for dataset in datasets}
    
    # Run evaluation based on mode
    if args.mode == "validate":
        logger.info("🔍 Running validation test...")
        result = runner.run_quick_validation(
            model_name="qwen3_8b",
            preset="memory_optimized",
            dataset="humaneval",
            num_samples=5
        )
        
        print("\\n✅ Validation completed successfully!")
        print(f"Throughput: {result['performance_metrics']['avg_throughput_tokens_per_second']:.1f} tokens/sec")
        print(f"Peak Memory: {result['performance_metrics']['peak_gpu_memory_gb']:.1f} GB")
        
    elif args.mode == "quick":
        logger.info("⚡ Running quick evaluation...")
        
        # Quick mode: 2 models, 2 presets, 3 datasets, 25 samples each
        quick_models = args.models if args.models else ["qwen3_8b", "qwen3_14b"]
        quick_presets = ["memory_optimized", "balanced"]
        quick_datasets = args.datasets if args.datasets else ["humaneval", "gsm8k", "hellaswag"]
        quick_samples = {dataset: 25 for dataset in quick_datasets}
        
        result = runner.run_comprehensive_evaluation(
            models=quick_models,
            presets=quick_presets,
            datasets=quick_datasets,
            samples_per_dataset=quick_samples,
            continue_on_failure=args.continue_on_failure
        )
        
        print(f"\\n⚡ Quick evaluation completed!")
        print(f"Total runs: {result['evaluation_summary']['total_runs']}")
        print(f"Successful: {result['evaluation_summary']['successful_runs']}")
        print(f"Failed: {result['evaluation_summary']['failed_runs']}")
        
    elif args.mode == "comprehensive":
        logger.info("🚀 Running comprehensive evaluation...")
        
        result = runner.run_comprehensive_evaluation(
            models=args.models,
            presets=args.presets,
            datasets=args.datasets,
            samples_per_dataset=samples_per_dataset,
            continue_on_failure=args.continue_on_failure
        )
        
        print(f"\\n🚀 Comprehensive evaluation completed!")
        print(f"Total runs: {result['evaluation_summary']['total_runs']}")
        print(f"Successful: {result['evaluation_summary']['successful_runs']}")
        print(f"Failed: {result['evaluation_summary']['failed_runs']}")
        print(f"Duration: {result['evaluation_summary']['total_duration_seconds']/3600:.1f} hours")
        
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())