#!/usr/bin/env python3
"""
Category-Based Model Evaluation CLI
===================================

Command-line interface for running systematic evaluations of model categories
on their appropriate datasets using the existing pipeline infrastructure.

Usage Examples:
  # Evaluate all coding specialists on all coding datasets with 5 samples (balanced preset)
  python category_evaluation.py --category coding_specialists --samples 5
  
  # Evaluate with performance preset for maximum throughput
  python category_evaluation.py --category coding_specialists --samples 5 --preset performance
  
  # Evaluate with memory-optimized preset for efficiency
  python category_evaluation.py --category coding_specialists --samples 5 --preset memory_optimized
  
  # Evaluate specific model on its category datasets
  python category_evaluation.py --model qwen3_8b --samples 10 --preset balanced
  
  # Evaluate specific model on specific dataset
  python category_evaluation.py --model qwen3_8b --dataset humaneval --samples 5 --preset performance
  
  # Dry run to see what would be evaluated
  python category_evaluation.py --category coding_specialists --dry-run --preset balanced
"""

import argparse
import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom JSON serialization framework
from evaluation.json_serializer import safe_json_dump, MLObjectEncoder

# OPTIMIZATION: Lazy imports for faster CLI startup
# Only import when needed to reduce 20s -> 3s startup time

def lazy_import_category_system():
    """Lazy import for category system - only when needed"""
    # Try improved discovery system first
    try:
        from evaluation.improved_discovery import CategoryMappingManager
        from evaluation.mappings import CATEGORY_REGISTRY
        from evaluation.mappings.model_categories import get_category_for_model
        from configs.model_configs import MODEL_CONFIGS
        return CategoryMappingManager, CATEGORY_REGISTRY, get_category_for_model, MODEL_CONFIGS, "improved"
    except ImportError:
        # Fallback to legacy system
        from evaluation.mappings import CategoryMappingManager, CATEGORY_REGISTRY
        from evaluation.mappings.model_categories import get_category_for_model
        from configs.model_configs import MODEL_CONFIGS
        return CategoryMappingManager, CATEGORY_REGISTRY, get_category_for_model, MODEL_CONFIGS, "legacy"

def lazy_import_pipeline_components():
    """Lazy import for heavy pipeline components - only when evaluation starts"""
    from evaluation.orchestrator import EvaluationOrchestrator
    from evaluation.dataset_manager import EnhancedDatasetManager  
    from evaluation.performance_monitor import LivePerformanceMonitor
    from models.registry import ModelRegistry
    return EvaluationOrchestrator, EnhancedDatasetManager, LivePerformanceMonitor, ModelRegistry

def get_category_registry_keys():
    """Get category registry keys for argument parser - lightweight operation"""
    try:
        from evaluation.mappings.model_categories import CATEGORY_REGISTRY
        return list(CATEGORY_REGISTRY.keys())
    except ImportError:
        return ['coding_specialists', 'mathematical_reasoning']  # Fallback

def get_model_configs_keys():
    """Get model config keys for argument parser - lightweight operation"""
    try:
        from configs.model_configs import MODEL_CONFIGS
        return list(MODEL_CONFIGS.keys())
    except ImportError:
        return []  # Fallback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CategoryEvaluationCLI:
    """Command-line interface for category-based model evaluation"""
    
    def __init__(self):
        # OPTIMIZATION: Delay heavy component initialization until actually needed
        self.manager = None
        self.orchestrator = None
        self.results_dir = Path("category_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _ensure_components_loaded(self):
        """Load heavy components only when needed for evaluation"""
        if self.manager is None:
            CategoryMappingManager, CATEGORY_REGISTRY, get_category_for_model, MODEL_CONFIGS, system_type = lazy_import_category_system()
            print(f"Using {system_type} discovery system...")
            self.manager = CategoryMappingManager()
            # Store these as instance variables for later use
            self._category_registry = CATEGORY_REGISTRY
            self._model_configs = MODEL_CONFIGS
            self._get_category_for_model = get_category_for_model
            
        if self.orchestrator is None:
            EvaluationOrchestrator, EnhancedDatasetManager, LivePerformanceMonitor, ModelRegistry = lazy_import_pipeline_components()
            self.orchestrator = EvaluationOrchestrator()
        
    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Category-based LLM evaluation system",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__
        )
        
        # Evaluation modes (mutually exclusive)
        mode_group = parser.add_mutually_exclusive_group(required=False)
        mode_group.add_argument(
            "--category",
            choices=get_category_registry_keys(),
            help="Evaluate all models in a category on category datasets"
        )
        mode_group.add_argument(
            "--model",
            choices=get_model_configs_keys(),
            help="Evaluate specific model (on its category datasets or specified dataset)"
        )
        
        # Dataset specification
        parser.add_argument(
            "--dataset",
            help="Specific dataset to evaluate (requires --model or --category)"
        )
        
        # Evaluation parameters
        parser.add_argument(
            "--samples",
            type=int,
            default=5,
            help="Number of samples per dataset (default: 5)"
        )
        
        parser.add_argument(
            "--preset",
            choices=["balanced", "performance", "memory_optimized"],
            default="balanced",
            help="Model preset configuration (default: balanced)"
        )
        
        parser.add_argument(
            "--include-optional",
            action="store_true",
            help="Include optional datasets for category evaluation"
        )
        
        # Model filtering (for category mode)
        parser.add_argument(
            "--models",
            nargs="+",
            help="Specific models to include in category evaluation (space-separated)"
        )
        
        parser.add_argument(
            "--exclude-models", 
            nargs="+",
            help="Models to exclude from category evaluation (space-separated)"
        )
        
        # Execution control
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be evaluated without running"
        )
        
        parser.add_argument(
            "--max-concurrent",
            type=int,
            default=1,
            help="Maximum concurrent evaluations (default: 1)"
        )
        
        parser.add_argument(
            "--output-dir",
            default="category_evaluation_results",
            help="Output directory for results (default: category_evaluation_results)"
        )
        
        # Validation and info
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Validate category readiness and exit"
        )
        
        parser.add_argument(
            "--list-categories",
            action="store_true",
            help="List all available categories and exit"
        )
        
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List all available models and exit"
        )
        
        parser.add_argument(
            "--category-info",
            help="Show detailed info about a specific category"
        )
        
        return parser.parse_args()
    
    def list_categories(self):
        """List all available categories"""
        self._ensure_components_loaded()
        
        print("\n" + "="*60)
        print("AVAILABLE MODEL CATEGORIES")
        print("="*60)
        
        for name, category in self._category_registry.items():
            # Use get_category_status for legacy-compatible format
            if hasattr(self.manager, 'get_category_status'):
                status = self.manager.get_category_status(name)
            else:
                status = self.manager.validate_category_readiness(name)
            
            ready_status = "✅ READY" if status['ready'] else "❌ NOT READY"
            
            print(f"\n{name.upper()}: {ready_status}")
            print(f"  Models ({len(category['models'])}): {', '.join(category['models'])}")
            print(f"  Primary Datasets ({len(category['primary_datasets'])}): {', '.join(category['primary_datasets'])}")
            
            if status['ready'] and 'primary_datasets' in status:
                available_count = len(status['primary_datasets']['available'])
                total_count = len(status['primary_datasets']['required'])
                print(f"  Available Datasets: {available_count}/{total_count}")
            elif 'available_datasets' in status:
                # Handle different status format
                available_count = len(status['available_datasets'])
                required_count = len(status['required_datasets'])
                print(f"  Available Datasets: {available_count}/{required_count}")
    
    def list_models(self):
        """List all available models with category information"""
        self._ensure_components_loaded()
        
        print("\n" + "="*60)
        print("AVAILABLE MODELS")
        print("="*60)
        
        models_by_category = {}
        uncategorized = []
        
        for model_name in sorted(self._model_configs.keys()):
            category_name = self._get_category_for_model(model_name)
            if category_name:
                if category_name not in models_by_category:
                    models_by_category[category_name] = []
                models_by_category[category_name].append(model_name)
            else:
                uncategorized.append(model_name)
        
        # Show categorized models
        for category_name, models in models_by_category.items():
            print(f"\n{category_name.upper()}:")
            for model in sorted(models):
                print(f"  - {model}")
        
        # Show uncategorized models
        if uncategorized:
            print(f"\nUNCATEGORIZED ({len(uncategorized)} models):")
            for model in sorted(uncategorized):
                print(f"  - {model}")
        
        print(f"\nTOTAL: {len(self._model_configs)} models")
    
    def show_category_info(self, category_name: str):
        """Show detailed information about a category"""
        self._ensure_components_loaded()
        
        if category_name not in self._category_registry:
            print(f"❌ Category '{category_name}' not found")
            return
        
        summary = self.manager.get_category_summary(category_name)
        
        print(f"\n" + "="*60)
        print(f"CATEGORY: {category_name.upper()}")
        print("="*60)
        
        category = self._category_registry[category_name]
        print(f"Description: {category.description}")
        print(f"Priority: {category.priority}")
        print(f"Ready: {'✅ YES' if summary['ready'] else '❌ NO'}")
        
        print(f"\nMODELS ({len(category.models)}):")
        for model in category.models:
            print(f"  - {model}")
        
        print(f"\nPRIMARY DATASETS:")
        primary = summary['primary_datasets']
        print(f"  Available: {primary['available']}/{primary['total']}")
        for dataset in primary['available_list']:
            print(f"    ✅ {dataset}")
        for dataset in primary['missing']:
            print(f"    ❌ {dataset} (missing)")
        
        if category.optional_datasets:
            print(f"\nOPTIONAL DATASETS:")
            optional = summary['optional_datasets']
            print(f"  Available: {optional['available']}/{optional['total']}")
            for dataset in optional['available_list']:
                print(f"    ✅ {dataset}")
            for dataset in optional['missing']:
                print(f"    ❌ {dataset} (missing)")
        
        if summary['ready']:
            breakdown = summary['task_breakdown']
            print(f"\nEVALUATION POTENTIAL:")
            print(f"  Total model-dataset combinations: {breakdown['total_combinations']}")
            print(f"  Estimated tasks (5 samples): {breakdown['total_combinations'] * 5}")
            
            # Show evaluation plan
            plan = self.manager.suggest_evaluation_plan(category_name)
            if plan['feasible']:
                rec = plan['recommendation']
                print(f"  Recommended sample limit: {rec['sample_limit']}")
                print(f"  Estimated evaluation time: {plan['execution_estimate']['estimated_time_minutes']} minutes")
    
    def validate_categories(self):
        """Validate all categories"""
        self._ensure_components_loaded()
        
        print("\n" + "="*60)
        print("CATEGORY VALIDATION REPORT")
        print("="*60)
        
        all_status = self.manager.get_all_categories_status()
        
        ready_count = 0
        for name, status in all_status.items():
            ready_indicator = "✅" if status['ready'] else "❌"
            print(f"\n{ready_indicator} {name.upper()}")
            
            if status['ready']:
                ready_count += 1
                primary = status['primary_datasets']
                print(f"   All {primary['total']} primary datasets available")
                print(f"   {status['potential_tasks']} potential evaluation tasks")
            else:
                if 'primary_datasets' in status:
                    missing = status['primary_datasets']['missing']
                    print(f"   Missing primary datasets: {missing}")
                else:
                    print(f"   Error: {status.get('error', 'Unknown error')}")
        
        print(f"\nSUMMARY: {ready_count}/{len(all_status)} categories ready")
    
    def generate_evaluation_tasks(self, args) -> List[Dict[str, Any]]:
        """Generate evaluation tasks based on arguments"""
        self._ensure_components_loaded()
        
        tasks = []
        
        if args.category:
            # Category-based evaluation with optional model filtering
            specific_models = None
            
            # Apply model filtering if specified
            if args.models or args.exclude_models:
                from evaluation.mappings.model_categories import get_models_in_category
                all_category_models = get_models_in_category(args.category)
                
                if args.models:
                    # Include only specified models
                    specific_models = [m for m in args.models if m in all_category_models]
                    if len(specific_models) != len(args.models):
                        missing = set(args.models) - set(specific_models)
                        logger.warning(f"Models not in category '{args.category}': {missing}")
                else:
                    specific_models = all_category_models.copy()
                
                if args.exclude_models:
                    # Handle both space-separated and comma-separated exclude models
                    exclude_list = []
                    for item in args.exclude_models:
                        if ',' in item:
                            exclude_list.extend([m.strip() for m in item.split(',')])
                        else:
                            exclude_list.append(item.strip())
                    
                    # Remove excluded models
                    specific_models = [m for m in specific_models if m not in exclude_list]
                    logger.info(f"Excluding models: {exclude_list}")
                
                if not specific_models:
                    logger.error("No models remaining after filtering")
                    return []
                
                logger.info(f"Filtered models for evaluation: {specific_models}")
            
            if args.dataset:
                # Category + specific dataset: evaluate all category models on one dataset
                from evaluation.mappings.model_categories import get_models_in_category
                if specific_models is None:
                    specific_models = get_models_in_category(args.category)
                
                for model in specific_models:
                    tasks.append({
                        'model': model,
                        'dataset': args.dataset,
                        'category': args.category,
                        'samples': args.samples,
                        'preset': args.preset,
                        'config': {}
                    })
            else:
                # Full category evaluation
                evaluation_tasks = self.manager.generate_evaluation_tasks(
                    args.category,
                    sample_limit=args.samples,
                    include_optional=args.include_optional,
                    specific_models=specific_models
                )
                
                for task in evaluation_tasks:
                    tasks.append({
                        'model': task.model_name,
                        'dataset': task.dataset_name,
                        'category': task.category,
                        'samples': task.sample_limit,
                        'preset': args.preset,
                        'config': task.evaluation_config
                    })
        
        elif args.model:
            if args.dataset:
                # Specific model + dataset
                tasks.append({
                    'model': args.model,
                    'dataset': args.dataset,
                    'category': None,
                    'samples': args.samples,
                    'preset': args.preset,
                    'config': {}
                })
            else:
                # Model on its category datasets
                self._ensure_components_loaded()
                category = self._get_category_for_model(args.model)
                if category:
                    evaluation_tasks = self.manager.generate_evaluation_tasks(
                        category,
                        sample_limit=args.samples,
                        include_optional=args.include_optional,
                        specific_models=[args.model]
                    )
                    
                    for task in evaluation_tasks:
                        tasks.append({
                            'model': task.model_name,
                            'dataset': task.dataset_name,
                            'category': task.category,
                            'samples': task.sample_limit,
                            'preset': args.preset,
                            'config': task.evaluation_config
                        })
                else:
                    logger.error(f"Model '{args.model}' not found in any category")
                    return []
        
        return tasks
    
    def show_dry_run(self, tasks: List[Dict[str, Any]]):
        """Show what would be evaluated in dry run mode"""
        print("\n" + "="*60)
        print("DRY RUN - EVALUATION PLAN")
        print("="*60)
        
        if not tasks:
            print("❌ No tasks would be executed")
            return
        
        print(f"Total tasks: {len(tasks)}")
        print(f"Total samples: {sum(task['samples'] for task in tasks)}")
        print(f"Preset mode: {tasks[0]['preset'] if tasks else 'N/A'}")
        
        # Group by category
        by_category = {}
        for task in tasks:
            category = task['category'] or 'uncategorized'
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(task)
        
        for category, category_tasks in by_category.items():
            print(f"\n{category.upper()}:")
            
            # Group by model
            by_model = {}
            for task in category_tasks:
                model = task['model']
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(task)
            
            for model, model_tasks in by_model.items():
                datasets = [task['dataset'] for task in model_tasks]
                total_samples = sum(task['samples'] for task in model_tasks)
                print(f"  {model}: {len(datasets)} datasets, {total_samples} samples")
                for task in model_tasks:
                    print(f"    - {task['dataset']} ({task['samples']} samples)")
        
        estimated_time = len(tasks) * 2  # Rough estimate: 2 minutes per evaluation
        print(f"\nEstimated execution time: {estimated_time} minutes")
    
    def run_evaluations(self, tasks: List[Dict[str, Any]], output_dir: str):
        """Execute the evaluation tasks"""
        self._ensure_components_loaded()
        
        if not tasks:
            logger.error("No tasks to execute")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create session log
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log = output_path / f"evaluation_session_{session_id}.json"
        
        session_data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "tasks": tasks,
            "results": []
        }
        
        print(f"\n" + "="*60)
        print(f"EXECUTING CATEGORY-BASED EVALUATION")
        print(f"Session: {session_id}")
        print(f"Tasks: {len(tasks)}")
        print("="*60)
        
        success_count = 0
        
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Evaluating {task['model']} on {task['dataset']}")
            print(f"Samples: {task['samples']}, Preset: {task['preset']}")
            
            try:
                # Use existing pipeline for evaluation
                result = self.orchestrator.run_single_evaluation(
                    model_name=task['model'],
                    dataset_name=task['dataset'],
                    preset=task['preset'],
                    sample_limit=task['samples']
                )
                
                if result and not result.get('error'):
                    print(f"✅ Success: {result.get('summary', 'Completed')}")
                    success_count += 1
                    
                    # Extract nested data from orchestrator result structure
                    eval_result = result.get('evaluation_result', {})
                    
                    # Save detailed predictions for debugging during development
                    if 'predictions' in eval_result and 'execution_details' in eval_result:
                        predictions_file = output_path / f"predictions_{session_id}_task{i}_{task['model']}_{task['dataset']}.json"
                        
                        # Convert evaluation metrics to JSON-serializable format
                        serializable_metrics = {}
                        eval_metrics = eval_result.get('evaluation_metrics', {})
                        for key, value in eval_metrics.items():
                            if hasattr(value, '__dict__'):
                                # Convert object to dictionary
                                serializable_metrics[key] = value.__dict__
                            else:
                                serializable_metrics[key] = value
                        
                        prediction_data = {
                            "task_info": task,
                            "predictions": eval_result['predictions'],
                            "ground_truth": eval_result.get('ground_truth', []),
                            "execution_details": eval_result['execution_details'],
                            "evaluation_metrics": serializable_metrics,
                            "timestamp": datetime.now().isoformat()
                        }
                        if not safe_json_dump(prediction_data, predictions_file, indent=2):
                            logger.error(f"Failed to save predictions: {predictions_file}")
                        else:
                            print(f"📄 Predictions saved: {predictions_file}")
                    
                    # Add result to session data (without predictions to keep session file smaller)
                    result_copy = result.copy()
                    eval_result_copy = result_copy.get('evaluation_result', {}).copy()
                    eval_result_copy.pop('predictions', None)  # Remove predictions from session log
                    eval_result_copy.pop('execution_details', None)  # Remove detailed execution from session log
                    
                    # Convert evaluation metrics to JSON-serializable format in session data too
                    if 'evaluation_metrics' in eval_result_copy:
                        serializable_session_metrics = {}
                        for key, value in eval_result_copy['evaluation_metrics'].items():
                            if hasattr(value, '__dict__'):
                                serializable_session_metrics[key] = value.__dict__
                            else:
                                serializable_session_metrics[key] = value
                        eval_result_copy['evaluation_metrics'] = serializable_session_metrics
                    
                    result_copy['evaluation_result'] = eval_result_copy
                    
                    session_data['results'].append({
                        "task_index": i,
                        "model": task['model'],
                        "dataset": task['dataset'],
                        "category": task['category'],
                        "samples": task['samples'],
                        "preset": task['preset'],
                        "success": True,
                        "result": result_copy,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    session_data['results'].append({
                        "task_index": i,
                        "model": task['model'],
                        "dataset": task['dataset'],
                        "category": task['category'],
                        "samples": task['samples'],
                        "preset": task['preset'],
                        "success": False,
                        "error": result.get('error', 'Unknown error'),
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Exception during evaluation: {e}")
                print(f"❌ Exception: {e}")
                session_data['results'].append({
                    "task_index": i,
                    "model": task['model'],
                    "dataset": task['dataset'],
                    "category": task['category'],
                    "samples": task['samples'],
                    "preset": task['preset'],
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Save session data after each task
            session_data['end_time'] = datetime.now().isoformat()
            if not safe_json_dump(session_data, session_log, indent=2):
                logger.error(f"Failed to save session log: {session_log}")
                # Fallback: save minimal session data
                minimal_session = {
                    "session_id": session_data["session_id"],
                    "start_time": session_data["start_time"],
                    "end_time": session_data.get("end_time"),
                    "total_tasks": session_data["total_tasks"],
                    "success_count": len([r for r in session_data["results"] if r.get("success")])
                }
                with open(str(session_log).replace('.json', '.minimal.json'), 'w') as f:
                    json.dump(minimal_session, f, indent=2)
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"EVALUATION COMPLETE")
        print(f"Success: {success_count}/{len(tasks)} tasks")
        print(f"Session log: {session_log}")
        print("="*60)
    
    def run(self):
        """Main CLI execution"""
        args = self.parse_args()
        
        # Handle info commands
        if args.list_categories:
            self.list_categories()
            return
        
        if args.list_models:
            self.list_models()
            return
        
        if args.category_info:
            self.show_category_info(args.category_info)
            return
        
        if args.validate:
            self.validate_categories()
            return
        
        # Validate arguments
        if not any([args.list_categories, args.list_models, args.category_info, args.validate]):
            if not args.category and not args.model:
                logger.error("Either --category or --model is required for evaluation")
                sys.exit(1)
        
        if args.dataset and not args.model and not args.category:
            logger.error("--dataset requires either --model or --category")
            sys.exit(1)
        
        # Generate evaluation tasks
        tasks = self.generate_evaluation_tasks(args)
        
        if args.dry_run:
            self.show_dry_run(tasks)
            return
        
        # Execute evaluations
        self.run_evaluations(tasks, args.output_dir)


if __name__ == "__main__":
    cli = CategoryEvaluationCLI()
    cli.run()