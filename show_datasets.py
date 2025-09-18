#!/usr/bin/env python3
"""
Dataset Summary Generator
Quick utility to display comprehensive dataset information
"""

from evaluation.dataset_manager import EnhancedDatasetManager
import json
from collections import defaultdict

def print_dataset_summary():
    """Print comprehensive dataset summary"""
    dm = EnhancedDatasetManager()
    summary = dm.get_dataset_summary()
    
    print("📊 LLM EVALUATION FRAMEWORK - DATASET SUMMARY")
    print("=" * 60)
    
    # Overall stats
    print(f"\n🔢 OVERALL STATISTICS:")
    print(f"   • Total datasets: {summary['total_datasets']}")
    print(f"   • Ready for evaluation: {summary['implemented_datasets']}")
    print(f"   • Pending implementation: {summary['unimplemented_datasets']}")
    
    # Task type distribution
    print(f"\n📋 TASK TYPE DISTRIBUTION:")
    for task_type, count in sorted(summary['task_type_distribution'].items()):
        implemented_count = len([name for name, info in summary['datasets'].items() 
                               if info['task_type'] == task_type and info['implemented']])
        print(f"   • {task_type.upper()}: {count} datasets ({implemented_count} ready)")
    
    # Sample count summary
    total_samples = sum(info['sample_count'] for info in summary['datasets'].values())
    ready_samples = sum(info['sample_count'] for info in summary['datasets'].values() 
                       if info['implemented'])
    
    print(f"\n📊 SAMPLE COUNTS:")
    print(f"   • Total available samples: {total_samples:,}")
    print(f"   • Ready for evaluation: {ready_samples:,}")
    print(f"   • Pending implementation: {total_samples - ready_samples:,}")
    
    # Ready datasets detail
    print(f"\n✅ READY FOR EVALUATION ({summary['implemented_datasets']} datasets):")
    ready_datasets = [(name, info) for name, info in summary['datasets'].items() 
                     if info['implemented']]
    
    for name, info in sorted(ready_datasets):
        print(f"   • {name} [{info['task_type']}] - {info['sample_count']:,} samples")
    
    print(f"\n🎯 USAGE:")
    print(f"   • Use dm.get_recommended_datasets() for smart dataset selection")
    print(f"   • See DATASET_SUMMARY_TABLE.md for complete catalog")
    print(f"   • Run evaluations with sample_limit parameter for testing")

def print_category_summary(category: str):
    """Print summary for specific category"""
    dm = EnhancedDatasetManager()
    summary = dm.get_dataset_summary()
    
    category_datasets = [(name, info) for name, info in summary['datasets'].items() 
                        if info['task_type'].lower() == category.lower()]
    
    if not category_datasets:
        print(f"❌ No datasets found for category: {category}")
        available_categories = sorted(set(info['task_type'] for info in summary['datasets'].values()))
        print(f"Available categories: {', '.join(available_categories)}")
        return
    
    print(f"📊 {category.upper()} CATEGORY SUMMARY")
    print("=" * 50)
    
    implemented = [item for item in category_datasets if item[1]['implemented']]
    pending = [item for item in category_datasets if not item[1]['implemented']]
    
    print(f"Total datasets: {len(category_datasets)}")
    print(f"Ready: {len(implemented)}")
    print(f"Pending: {len(pending)}")
    
    if implemented:
        print(f"\n✅ READY DATASETS:")
        for name, info in implemented:
            print(f"   • {name} - {info['sample_count']:,} samples - {info['description']}")
    
    if pending:
        print(f"\n🔄 PENDING DATASETS:")
        for name, info in pending:
            print(f"   • {name} - {info['sample_count']:,} samples - {info['description']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        category = sys.argv[1]
        print_category_summary(category)
    else:
        print_dataset_summary()
        
        print(f"\n💡 TIP: Run 'python show_datasets.py [category]' for category-specific info")
        print(f"   Example: python show_datasets.py coding")