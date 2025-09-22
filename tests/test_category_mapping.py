#!/usr/bin/env python3
"""
Test script for the category mapping system.
Validates the coding specialists category and mapping functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.mappings import (
    validate_coding_readiness,
    quick_coding_evaluation_plan,
    get_coding_specialists_manager,
    CODING_SPECIALISTS
)

def test_category_mapping():
    """Test the category mapping system functionality"""
    
    print("=" * 60)
    print("CATEGORY MAPPING SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Basic category information
    print("\n1. CODING SPECIALISTS CATEGORY INFO:")
    print(f"   Name: {CODING_SPECIALISTS.name}")
    print(f"   Models: {CODING_SPECIALISTS.models}")
    print(f"   Primary Datasets: {CODING_SPECIALISTS.primary_datasets}")
    print(f"   Optional Datasets: {CODING_SPECIALISTS.optional_datasets}")
    print(f"   Priority: {CODING_SPECIALISTS.priority}")
    
    # Test 2: Dataset discovery
    print("\n2. DATASET DISCOVERY:")
    manager = get_coding_specialists_manager()
    available_datasets = manager.get_available_datasets()
    print(f"   Discovered datasets: {len(available_datasets)}")
    print(f"   Datasets: {available_datasets}")
    
    # Test 3: Category readiness validation
    print("\n3. CODING CATEGORY READINESS:")
    readiness = validate_coding_readiness()
    print(f"   Ready: {readiness['ready']}")
    if 'primary_datasets' in readiness:
        print(f"   Primary datasets available: {readiness['primary_datasets']['available']}/{readiness['primary_datasets']['total']}")
        print(f"   Available primary: {readiness['primary_datasets']['available_list']}")
        if readiness['primary_datasets']['missing']:
            print(f"   Missing primary: {readiness['primary_datasets']['missing']}")
    
    # Test 4: Evaluation plan
    print("\n4. EVALUATION PLAN:")
    plan = quick_coding_evaluation_plan(sample_limit=5)
    print(f"   Feasible: {plan['feasible']}")
    if plan['feasible']:
        rec = plan['recommendation']
        print(f"   Recommended sample limit: {rec['sample_limit']}")
        print(f"   Estimated tasks: {rec['estimated_tasks']}")
        print(f"   Estimated total samples: {rec['estimated_total_samples']}")
        print(f"   Available datasets: {plan['breakdown']['dataset_list']}")
    else:
        print(f"   Reason: {plan.get('reason', 'Unknown')}")
    
    # Test 5: Task generation
    print("\n5. TASK GENERATION:")
    tasks = manager.generate_evaluation_tasks("coding_specialists", sample_limit=3)
    print(f"   Generated tasks: {len(tasks)}")
    if tasks:
        print(f"   Sample task: {tasks[0].model_name} on {tasks[0].dataset_name}")
        print(f"   Task config keys: {list(tasks[0].evaluation_config.keys())}")
    
    # Test 6: Missing datasets analysis
    print("\n6. MISSING DATASETS ANALYSIS:")
    missing_primary, missing_optional = manager.get_missing_datasets_for_category("coding_specialists")
    print(f"   Missing primary: {missing_primary}")
    print(f"   Missing optional: {missing_optional}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return {
        "datasets_found": len(available_datasets),
        "category_ready": readiness['ready'],
        "tasks_generated": len(tasks),
        "missing_primary": len(missing_primary),
        "missing_optional": len(missing_optional)
    }

if __name__ == "__main__":
    results = test_category_mapping()
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"- Datasets discovered: {results['datasets_found']}")
    print(f"- Category ready: {results['category_ready']}")
    print(f"- Tasks generated: {results['tasks_generated']}")
    print(f"- Missing primary datasets: {results['missing_primary']}")
    print(f"- Missing optional datasets: {results['missing_optional']}")