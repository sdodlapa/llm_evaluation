#!/usr/bin/env python3
"""
Consolidated debugging utilities for the evaluation framework
Combines the most useful debugging functions from previous scattered files
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import EvaluationMetrics, evaluate_dataset_predictions

def test_humaneval_fix():
    """Test the HumanEval format fix with sample data"""
    print("=== Testing HumanEval Fix ===")
    
    # Sample response with code
    sample_response = """Here's the solution:

```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    \"\"\"
    # Sort the list of numbers
    numbers.sort()
    
    # Check the difference between consecutive elements
    for i in range(len(numbers) - 1):
        if numbers[i + 1] - numbers[i] < threshold:
            return True
    
    return False
```

This function works by sorting the numbers first."""
    
    # Actual HumanEval test case
    test_code = """
METADATA = {
    'author': 'jt',
    'dataset': 'test'
}

def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
"""
    
    # Test code extraction
    extracted_code = EvaluationMetrics._extract_code_from_response(sample_response)
    print(f"✅ Code extracted ({len(extracted_code)} chars)")
    
    # Test HumanEval execution
    result = EvaluationMetrics._execute_humaneval_tests(extracted_code, test_code)
    print(f"✅ HumanEval execution: {result['passed']} ({result['tests_passed']}/{result['tests_run']})")
    
    return result['passed']

def test_evaluation_pipeline():
    """Test the complete evaluation pipeline with mock data"""
    print("\n=== Testing Complete Pipeline ===")
    
    # Mock HumanEval data
    samples = [{
        "id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers, threshold):\n    pass",
        "test_cases": """
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
""",
        "expected_output": "some_output"
    }]
    
    predictions = ["""
```python
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```
"""]
    
    # Test pipeline
    results = evaluate_dataset_predictions("coding", predictions, samples)
    
    print(f"✅ Pipeline results:")
    for metric_name, result in results.items():
        print(f"   {metric_name}: {result.score:.1%}")
    
    return "code_execution" in results and results["code_execution"].score > 0

def debug_dataset_format(dataset_name):
    """Debug a specific dataset format"""
    print(f"\n=== Debugging {dataset_name} Format ===")
    
    try:
        with open(f'evaluation_data/coding/{dataset_name.lower()}.json', 'r') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        if not samples:
            print("❌ No samples found")
            return
        
        sample = samples[0]
        print(f"✅ Dataset: {data.get('name')}")
        print(f"✅ Sample ID: {sample.get('id')}")
        print(f"✅ Has prompt: {'prompt' in sample}")
        print(f"✅ Has test_cases: {'test_cases' in sample}")
        print(f"✅ Test cases type: {type(sample.get('test_cases'))}")
        
        if sample.get('test_cases'):
            tc = sample['test_cases']
            if isinstance(tc, str):
                print(f"✅ Test cases preview: {tc[:100]}...")
            else:
                print(f"✅ Test cases: {tc}")
        else:
            print("⚠️  Empty test_cases")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def run_all_debug_tests():
    """Run all debugging tests"""
    print("=" * 60)
    print("🔧 EVALUATION FRAMEWORK DEBUG SUITE")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: HumanEval fix
    if test_humaneval_fix():
        tests_passed += 1
        print("✅ HumanEval fix test: PASSED")
    else:
        print("❌ HumanEval fix test: FAILED")
    
    # Test 2: Pipeline test
    if test_evaluation_pipeline():
        tests_passed += 1
        print("✅ Pipeline test: PASSED")
    else:
        print("❌ Pipeline test: FAILED")
    
    print(f"\n📊 Debug Results: {tests_passed}/{total_tests} tests passed")
    
    # Debug dataset formats
    for dataset in ["humaneval", "mbpp"]:
        debug_dataset_format(dataset)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    run_all_debug_tests()