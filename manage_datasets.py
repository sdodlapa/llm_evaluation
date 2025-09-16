#!/usr/bin/env python3
"""
Dataset management CLI for LLM evaluation
Handles downloading, analyzing, and managing evaluation datasets
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.dataset_manager import EvaluationDatasetManager

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_dataset_summary(manager: EvaluationDatasetManager):
    """Print comprehensive dataset summary"""
    summary = manager.get_dataset_summary()
    
    print("\n" + "="*60)
    print("📊 EVALUATION DATASETS SUMMARY")
    print("="*60)
    
    print(f"\n📈 Overview:")
    print(f"  Available datasets: {summary['available_datasets']}")
    print(f"  Cached datasets: {len(summary['cached_datasets'])}")
    print(f"  Total available size: {summary['total_available_size_mb']/1024:.2f}GB")
    print(f"  Total cached size: {summary['total_cached_size_mb']/1024:.2f}GB")
    
    print(f"\n📋 By Task Type:")
    for task_type, info in summary['by_task_type'].items():
        cached_count = sum(1 for ds in info['datasets'] if ds in summary['cached_datasets'])
        status = f"({cached_count}/{info['count']} cached)" if cached_count > 0 else "(none cached)"
        print(f"  📁 {task_type.replace('_', ' ').title()}: {info['count']} datasets {status}")
        print(f"     Size: {info['total_size_mb']/1024:.2f}GB")
        
        for dataset_name in info['datasets']:
            cached_marker = "✅" if dataset_name in summary['cached_datasets'] else "⬜"
            dataset_info = summary['datasets'][dataset_name]['info']
            print(f"     {cached_marker} {dataset_name}: {dataset_info['size_mb']}MB - {dataset_info['description'][:50]}...")
    
    if summary['cached_datasets']:
        print(f"\n💾 Cached Datasets:")
        for dataset_name in summary['cached_datasets']:
            dataset_info = summary['datasets'][dataset_name]['info']
            print(f"  ✅ {dataset_name} ({dataset_info['size_mb']}MB)")
    else:
        print(f"\n💾 No datasets cached yet")
        print(f"  Run: python manage_datasets.py --download-recommended")
    
    print("\n" + "="*60)

def download_datasets(manager: EvaluationDatasetManager, dataset_names: list = None, 
                     task_types: list = None, recommended: bool = False):
    """Download specified datasets"""
    if recommended:
        print("\n📥 Downloading recommended datasets...")
        results = manager.download_recommended_datasets(task_types)
        
        successful = 0
        failed = 0
        total_size = 0
        
        for dataset_name, result in results.items():
            if 'error' in result:
                print(f"❌ {dataset_name}: {result['error']}")
                failed += 1
            else:
                print(f"✅ {dataset_name}: Downloaded successfully")
                successful += 1
                if dataset_name in manager.datasets_info:
                    total_size += manager.datasets_info[dataset_name].size_mb
        
        print(f"\n📊 Download Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total downloaded: {total_size/1024:.2f}GB")
        
    elif dataset_names:
        print(f"\n📥 Downloading specific datasets: {', '.join(dataset_names)}")
        
        for dataset_name in dataset_names:
            try:
                print(f"\n📥 Downloading {dataset_name}...")
                result = manager.download_dataset(dataset_name)
                
                if 'error' in result:
                    print(f"❌ {dataset_name}: {result['error']}")
                else:
                    samples = len(result.get('samples', []))
                    task_type = result.get('task_type', 'unknown')
                    print(f"✅ {dataset_name}: {samples} samples ({task_type})")
                    
            except Exception as e:
                print(f"❌ Failed to download {dataset_name}: {e}")
    else:
        print("❌ Please specify either --recommended or specific dataset names")

def analyze_dataset(manager: EvaluationDatasetManager, dataset_name: str):
    """Analyze a specific dataset"""
    print(f"\n🔍 Analyzing dataset: {dataset_name}")
    
    # Load dataset
    dataset = manager.load_cached_dataset(dataset_name)
    if not dataset:
        print(f"❌ Dataset {dataset_name} not cached. Download it first.")
        return
    
    # Print analysis
    metadata = dataset.get('metadata', {})
    samples = dataset.get('samples', [])
    
    print(f"\n📊 Dataset Analysis:")
    print(f"  Name: {dataset.get('name', dataset_name)}")
    print(f"  Task Type: {dataset.get('task_type', 'unknown')}")
    print(f"  Total Samples: {metadata.get('total_samples', 0)}")
    print(f"  Processed Samples: {len(samples)}")
    print(f"  Has Labels: {metadata.get('has_labels', False)}")
    print(f"  Source: {metadata.get('source', 'unknown')}")
    print(f"  License: {metadata.get('license', 'unknown')}")
    print(f"  Downloaded: {dataset.get('downloaded_at', 'unknown')}")
    
    if samples:
        print(f"\n📝 Sample Analysis:")
        sample = samples[0]
        print(f"  Sample Keys: {list(sample.keys())}")
        
        # Show example sample
        print(f"\n📄 Example Sample:")
        for key, value in list(sample.items())[:3]:  # First 3 keys
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {value}")

def list_available_datasets(manager: EvaluationDatasetManager, task_type: str = None):
    """List all available datasets"""
    summary = manager.get_dataset_summary()
    
    print(f"\n📋 Available Datasets:")
    
    if task_type:
        if task_type in summary['by_task_type']:
            info = summary['by_task_type'][task_type]
            print(f"\n📁 {task_type.replace('_', ' ').title()} ({info['count']} datasets):")
            
            for dataset_name in info['datasets']:
                dataset_info = summary['datasets'][dataset_name]['info']
                cached = "✅" if summary['datasets'][dataset_name]['cached'] else "⬜"
                print(f"  {cached} {dataset_name}")
                print(f"     📏 Size: {dataset_info['size_mb']}MB")
                print(f"     📝 Description: {dataset_info['description']}")
                print(f"     🏷️  License: {dataset_info['license']}")
                print()
        else:
            print(f"❌ Unknown task type: {task_type}")
            print(f"Available types: {list(summary['by_task_type'].keys())}")
    else:
        for task_type, info in summary['by_task_type'].items():
            print(f"\n📁 {task_type.replace('_', ' ').title()} ({info['count']} datasets, {info['total_size_mb']/1024:.2f}GB):")
            
            for dataset_name in info['datasets']:
                dataset_info = summary['datasets'][dataset_name]['info']
                cached = "✅" if summary['datasets'][dataset_name]['cached'] else "⬜"
                print(f"  {cached} {dataset_name} ({dataset_info['size_mb']}MB)")

def main():
    parser = argparse.ArgumentParser(
        description="Manage evaluation datasets for LLM testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_datasets.py --summary                    # Show dataset overview
  python manage_datasets.py --download-recommended       # Download recommended datasets  
  python manage_datasets.py --download humaneval gsm8k   # Download specific datasets
  python manage_datasets.py --list                       # List all available datasets
  python manage_datasets.py --list --task-type coding    # List coding datasets only
  python manage_datasets.py --analyze humaneval          # Analyze cached dataset
        """
    )
    
    parser.add_argument("--cache-dir", default="evaluation_data", 
                       help="Directory to cache datasets (default: evaluation_data)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--summary", action="store_true",
                             help="Show comprehensive dataset summary")
    action_group.add_argument("--download-recommended", action="store_true",
                             help="Download all recommended datasets")
    action_group.add_argument("--download", nargs="+", metavar="DATASET",
                             help="Download specific datasets by name")
    action_group.add_argument("--list", action="store_true",
                             help="List all available datasets")
    action_group.add_argument("--analyze", metavar="DATASET",
                             help="Analyze a specific cached dataset")
    
    # Filters
    parser.add_argument("--task-type", 
                       choices=["function_calling", "coding", "reasoning", "instruction_following", "qa"],
                       help="Filter by task type")
    parser.add_argument("--max-size", type=float, default=10.0,
                       help="Maximum total size in GB (default: 10.0)")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    manager = EvaluationDatasetManager(cache_dir=args.cache_dir, max_total_size_gb=args.max_size)
    
    # Execute actions
    try:
        if args.summary:
            print_dataset_summary(manager)
            
        elif args.download_recommended:
            task_types = [args.task_type] if args.task_type else None
            download_datasets(manager, recommended=True, task_types=task_types)
            
        elif args.download:
            download_datasets(manager, dataset_names=args.download)
            
        elif args.list:
            list_available_datasets(manager, args.task_type)
            
        elif args.analyze:
            analyze_dataset(manager, args.analyze)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()