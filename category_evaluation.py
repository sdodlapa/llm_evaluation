#!/usr/bin/env python3
"""
Category-Based Model Evaluation CLI
===================================

Command-line interface for running systematic evaluations of model categories
on their appropriate datasets using the existing pipeline infrastructure.

Usage Examples:
  # Evaluate all coding specialists on all coding datasets with 5 samples
  python category_evaluation.py --category coding_specialists --samples 5
  
  # Evaluate specific model on its category datasets
  python category_evaluation.py --model qwen3_8b --samples 10
  
  # Evaluate specific model on specific dataset
  python category_evaluation.py --model qwen3_8b --dataset humaneval --samples 5
  
  # Dry run to see what would be evaluated
  python category_evaluation.py --category coding_specialists --dry-run
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

# Import our new category system
from evaluation.mappings import (
    CategoryMappingManager,
    CATEGORY_REGISTRY,
    get_category_for_model,
    validate_coding_readiness
)

# Import existing pipeline components
from evaluation.orchestrator import EvaluationOrchestrator
from configs.model_configs import MODEL_CONFIGS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CategoryEvaluationCLI:
    """Command-line interface for category-based model evaluation"""
    
    def __init__(self):
        self.manager = CategoryMappingManager()
        self.orchestrator = EvaluationOrchestrator()
        self.results_dir = Path("category_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
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
            choices=list(CATEGORY_REGISTRY.keys()),
            help="Evaluate all models in a category on category datasets"
        )
        mode_group.add_argument(
            "--model",
            choices=list(MODEL_CONFIGS.keys()),
            help="Evaluate specific model (on its category datasets or specified dataset)"
        )
        
        # Dataset specification
        parser.add_argument(
            "--dataset",
            help="Specific dataset to evaluate (requires --model)"
        )
        
        # Evaluation parameters
        parser.add_argument(
            "--samples",
            type=int,
            default=5,
            help="Number of samples per dataset (default: 5)"
        )
        
        parser.add_argument(
            "--include-optional",
            action="store_true",
            help="Include optional datasets for category evaluation"
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
        print("\n" + "="*60)
        print("AVAILABLE MODEL CATEGORIES")
        print("="*60)
        
        for name, category in CATEGORY_REGISTRY.items():
            status = self.manager.validate_category_readiness(name)
            ready_status = "✅ READY" if status['ready'] else "❌ NOT READY"
            
            print(f"\n{name.upper()}: {ready_status}")
            print(f"  Description: {category.description}")
            print(f"  Models ({len(category.models)}): {', '.join(category.models)}")
            print(f"  Primary Datasets ({len(category.primary_datasets)}): {', '.join(category.primary_datasets)}")
            
            if status['ready']:
                available_primary = status['primary_datasets']['available']
                total_primary = status['primary_datasets']['total']
                print(f"  Available Datasets: {available_primary}/{total_primary}")
    
    def list_models(self):
        """List all available models with category information"""
        print("\n" + "="*60)
        print("AVAILABLE MODELS")
        print("="*60)
        
        models_by_category = {}
        uncategorized = []
        
        for model_name in sorted(MODEL_CONFIGS.keys()):
            category = get_category_for_model(model_name)
            if category:
                if category.name not in models_by_category:
                    models_by_category[category.name] = []
                models_by_category[category.name].append(model_name)
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
        
        print(f"\nTOTAL: {len(MODEL_CONFIGS)} models")
    
    def show_category_info(self, category_name: str):
        """Show detailed information about a category"""
        if category_name not in CATEGORY_REGISTRY:
            print(f"❌ Category '{category_name}' not found")
            return
        
        summary = self.manager.get_category_summary(category_name)
        
        print(f"\n" + "="*60)
        print(f"CATEGORY: {category_name.upper()}")
        print("="*60)
        
        category = CATEGORY_REGISTRY[category_name]
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
        tasks = []
        
        if args.category:
            # Category-based evaluation
            evaluation_tasks = self.manager.generate_evaluation_tasks(
                args.category,
                sample_limit=args.samples,
                include_optional=args.include_optional
            )
            
            for task in evaluation_tasks:
                tasks.append({
                    'model': task.model_name,
                    'dataset': task.dataset_name,
                    'category': task.category,
                    'samples': task.sample_limit,
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
                    'config': {}
                })
            else:
                # Model on its category datasets
                category = get_category_for_model(args.model)
                if category:
                    evaluation_tasks = self.manager.generate_evaluation_tasks(
                        category.name,
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
            print(f"Samples: {task['samples']}")
            
            try:
                # Use existing pipeline for evaluation
                result = self.orchestrator.run_single_evaluation(
                    model_name=task['model'],
                    dataset_name=task['dataset'],
                    sample_limit=task['samples']
                )
                
                if result and result.get('success', False):
                    print(f"✅ Success: {result.get('summary', 'Completed')}")
                    success_count += 1
                    
                    # Add result to session data
                    session_data['results'].append({
                        "task_index": i,
                        "model": task['model'],
                        "dataset": task['dataset'],
                        "category": task['category'],
                        "samples": task['samples'],
                        "success": True,
                        "result": result,
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
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Save session data after each task
            session_data['end_time'] = datetime.now().isoformat()
            with open(session_log, 'w') as f:
                json.dump(session_data, f, indent=2)
        
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
        
        if args.dataset and not args.model:
            logger.error("--dataset requires --model")
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