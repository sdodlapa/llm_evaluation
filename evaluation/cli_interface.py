#!/usr/bin/env python3
"""
CLI Interface for LLM Evaluation
Handles command-line argument parsing and user interface
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.model_configs import MODEL_CONFIGS, get_high_priority_models, get_agent_optimized_models

logger = logging.getLogger(__name__)

class EvaluationCLI:
    """Command-line interface for LLM evaluation"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the command-line argument parser"""
        parser = argparse.ArgumentParser(
            description='🚀 LLM Evaluation Framework - Comprehensive model testing',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Single model evaluation
  %(prog)s --model qwen3_8b --preset balanced --dataset humaneval --samples 10
  
  # Multi-model comparison
  %(prog)s --model qwen3_8b,qwen3_14b --preset balanced --dataset humaneval,mbpp
  
  # Full evaluation suite
  %(prog)s --model qwen3_8b --preset balanced
  
  # Preset comparison
  %(prog)s --model qwen3_8b --compare-presets
  
  # Quick test run
  %(prog)s --model qwen3_8b --quick-test
            """
        )
        
        # Model selection
        parser.add_argument(
            '--model', '--models',
            type=str,
            help='Model(s) to evaluate (comma-separated). Available: ' + 
                 ', '.join(list(MODEL_CONFIGS.keys())[:10]) + '...'
        )
        
        # Configuration
        parser.add_argument(
            '--preset',
            choices=['memory_optimized', 'balanced', 'performance'],
            default='balanced',
            help='Evaluation preset (default: balanced)'
        )
        
        # Dataset selection
        parser.add_argument(
            '--dataset', '--datasets',
            type=str,
            help='Dataset(s) to use (comma-separated). If not specified, uses comprehensive suite'
        )
        
        # Sample control
        parser.add_argument(
            '--samples',
            type=int,
            help='Number of samples per dataset (default: varies by dataset)'
        )
        
        # Evaluation modes
        parser.add_argument(
            '--quick-test',
            action='store_true',
            help='Quick test with minimal samples (5 per dataset)'
        )
        
        parser.add_argument(
            '--compare-presets',
            action='store_true',
            help='Compare all presets for specified model'
        )
        
        parser.add_argument(
            '--full-evaluation',
            action='store_true',
            help='Run comprehensive evaluation on all high-priority models'
        )
        
        parser.add_argument(
            '--comparison-analysis',
            action='store_true',
            help='Run comparison analysis between models'
        )
        
        # Output control
        parser.add_argument(
            '--output-dir',
            type=str,
            default='results',
            help='Output directory for results (default: results)'
        )
        
        parser.add_argument(
            '--cache-dir',
            type=str,
            help='Cache directory for model storage'
        )
        
        parser.add_argument(
            '--data-cache-dir',
            type=str,
            help='Cache directory for dataset storage'
        )
        
        # Advanced options
        parser.add_argument(
            '--skip-performance',
            action='store_true',
            help='Skip performance benchmarking'
        )
        
        parser.add_argument(
            '--skip-datasets',
            action='store_true',
            help='Skip dataset evaluation'
        )
        
        parser.add_argument(
            '--continue-on-error',
            action='store_true',
            help='Continue evaluation even if individual tests fail'
        )
        
        # Debugging
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
        )
        
        parser.add_argument(
            '--list-models',
            action='store_true',
            help='List available models and exit'
        )
        
        parser.add_argument(
            '--list-datasets',
            action='store_true',
            help='List available datasets and exit'
        )
        
        return parser
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments"""
        return self.parser.parse_args(args)
    
    def validate_arguments(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Validate and process arguments into evaluation parameters"""
        params = {}
        
        # Process model selection
        if args.model:
            models = [m.strip() for m in args.model.split(',')]
            invalid_models = [m for m in models if m not in MODEL_CONFIGS]
            if invalid_models:
                raise ValueError(f"Invalid models: {invalid_models}. "
                               f"Available: {list(MODEL_CONFIGS.keys())}")
            params['models'] = models
        else:
            if args.compare_presets or args.full_evaluation:
                params['models'] = list(get_high_priority_models().keys())
            else:
                # Default to Qwen models if no specific selection
                params['models'] = ['qwen3_8b']
        
        # Process dataset selection
        if args.dataset:
            params['datasets'] = [d.strip() for d in args.dataset.split(',')]
        else:
            params['datasets'] = None  # Use default comprehensive suite
        
        # Process sample configuration
        if args.quick_test:
            params['sample_limits'] = {'default': 5}
        elif args.samples:
            params['sample_limits'] = {'default': args.samples}
        else:
            params['sample_limits'] = None
        
        # Process evaluation modes
        params['preset'] = args.preset
        params['skip_performance'] = args.skip_performance
        params['skip_datasets'] = args.skip_datasets
        params['continue_on_error'] = args.continue_on_error
        
        # Output configuration
        params['output_dir'] = args.output_dir
        params['cache_dir'] = args.cache_dir
        params['data_cache_dir'] = args.data_cache_dir
        
        return params
    
    def setup_logging(self, args: argparse.Namespace) -> None:
        """Setup logging based on CLI arguments"""
        if args.debug:
            level = logging.DEBUG
        elif args.verbose:
            level = logging.INFO
        else:
            level = logging.WARNING
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def print_available_models(self) -> None:
        """Print available models and exit"""
        print("🤖 Available Models:")
        print("=" * 50)
        
        for category in ['text_generation', 'code_generation', 'mathematics', 'biomedical']:
            models = [name for name, config in MODEL_CONFIGS.items() 
                     if config.specialization_category == category]
            if models:
                print(f"\n{category.replace('_', ' ').title()}:")
                for model in models:
                    config = MODEL_CONFIGS[model]
                    print(f"  {model}: {config.model_name} ({config.size_gb}GB)")
        
        # Show general models
        general_models = [name for name, config in MODEL_CONFIGS.items() 
                         if config.specialization_category == 'general']
        if general_models:
            print(f"\nGeneral Purpose:")
            for model in general_models:
                config = MODEL_CONFIGS[model]
                print(f"  {model}: {config.model_name} ({config.size_gb}GB)")
    
    def print_available_datasets(self) -> None:
        """Print available datasets and exit"""
        try:
            from evaluation.dataset_manager import EnhancedDatasetManager
            dm = EnhancedDatasetManager()
            
            print("📊 Available Datasets:")
            print("=" * 50)
            
            by_task = {}
            for name, info in dm.datasets.items():
                if info.task_type not in by_task:
                    by_task[info.task_type] = []
                by_task[info.task_type].append((name, info))
            
            for task_type, datasets in by_task.items():
                print(f"\n{task_type.replace('_', ' ').title()}:")
                for name, info in datasets:
                    status = "✅" if info.implemented else "⚠️"
                    print(f"  {status} {name}: {info.sample_count:,} samples - {info.description}")
        
        except ImportError:
            print("Dataset manager not available")
    
    def handle_list_commands(self, args: argparse.Namespace) -> bool:
        """Handle list commands that should exit after displaying info"""
        if args.list_models:
            self.print_available_models()
            return True
        
        if args.list_datasets:
            self.print_available_datasets()
            return True
        
        return False


def create_cli() -> EvaluationCLI:
    """Factory function to create CLI instance"""
    return EvaluationCLI()