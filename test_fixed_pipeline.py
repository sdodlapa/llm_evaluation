#!/usr/bin/env python3
"""Test the fixed evaluation pipeline with real HumanEval data"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))

from metrics import evaluate_dataset_predictions

def test_fixed_evaluation_pipeline():
    """Test that the fixed pipeline works with real HumanEval data"""
    
    print("=== Testing Fixed Evaluation Pipeline ===")
    print()
    
    # Load real HumanEval data (first 2 samples)
    with open('evaluation_data/coding/humaneval.json', 'r') as f:
        humaneval_data = json.load(f)
    
    # Get first 2 samples
    samples = humaneval_data['samples'][:2]
    
    # Create mock predictions (same working solutions from before)
    predictions = [
        """Here's the solution:

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

This function works by sorting the numbers first.""",
        
        """Here's my solution:

```python
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    \"\"\"
    result = []
    current_string = ""
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string += c
        elif c == ')':
            current_depth -= 1
            current_string += c

            if current_depth == 0:
                result.append(current_string)
                current_string = ""

    return result
```

This processes each character and tracks nesting depth."""
    ]
    
    print(f"Testing with {len(samples)} HumanEval samples")
    print(f"Sample IDs: {[s['id'] for s in samples]}")
    print()
    
    # Run evaluation
    try:
        results = evaluate_dataset_predictions("coding", predictions, samples)
        
        print("Evaluation Results:")
        for metric_name, result in results.items():
            print(f"  {metric_name}:")
            print(f"    Score: {result.score:.2%}")
            print(f"    Total samples: {result.total_samples}")
            print(f"    Successful: {result.successful_samples}")
            
            # Show detailed test results for code execution
            if metric_name == "code_execution" and hasattr(result, 'details') and result.details:
                test_results = result.details.get("test_results", [])
                for i, test_result in enumerate(test_results):
                    print(f"    Sample {i+1}:")
                    print(f"      Passed: {test_result['passed']}")
                    print(f"      Tests run: {test_result['tests_run']}")
                    print(f"      Tests passed: {test_result['tests_passed']}")
                    if test_result.get('error'):
                        print(f"      Error: {test_result['error']}")
            print()
        
        # Check if we got good results
        if "code_execution" in results:
            score = results["code_execution"].score
            if score > 0:
                print(f"✅ SUCCESS! Fixed pipeline shows {score:.1%} accuracy")
                print("   This is a vast improvement from the previous 0%!")
            else:
                print("❌ STILL FAILING: Score is still 0%")
        else:
            print("❌ ERROR: No code execution results returned")
            
    except Exception as e:
        print(f"❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_evaluation_pipeline()