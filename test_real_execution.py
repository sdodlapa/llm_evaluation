#!/usr/bin/env python3
"""Test code execution with real HumanEval data"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))

from metrics import EvaluationMetrics

def test_real_evaluation():
    """Test code execution with realistic model responses"""
    
    # Sample realistic responses that models might generate (with markdown)
    mock_responses = [
        """Here's a solution for the problem:

```python
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

This function checks if any two elements in the list are closer than the threshold.""",
        
        """def separate_paren_groups(paren_string):
    result = []
    current_group = ""
    depth = 0
    
    for char in paren_string:
        if char == '(':
            depth += 1
            current_group += char
        elif char == ')':
            depth -= 1
            current_group += char
            if depth == 0:
                result.append(current_group)
                current_group = ""
    
    return result""",
        
        """Looking at this problem, I need to truncate a number to a given precision:

```python
def truncate_number(number):
    return number - int(number)
```

The function simply subtracts the integer part from the number."""
    ]
    
    # Sample test cases (simplified from HumanEval format)
    test_cases = [
        [{"input": [[1.0, 2.0, 3.0], 0.5], "output": False},
         {"input": [[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3], "output": True}],
        
        [{"input": ["(()()) ((())) () ((())()())"], "output": ["(()())", "((()))", "()", "((())()())"]}],
        
        [{"input": [3.5], "output": 0.5},
         {"input": [1.25], "output": 0.25}]
    ]
    
    print("Testing code execution with realistic responses...")
    print("=" * 50)
    
    result = EvaluationMetrics.code_execution_accuracy(mock_responses, test_cases, timeout=5)
    
    print(f"Overall accuracy: {result.score:.2%}")
    print(f"Total samples: {result.total_samples}")
    print(f"Successful: {result.successful_samples}")
    print(f"Failed: {result.total_samples - result.successful_samples}")
    print()
    
    print("Individual test results:")
    for i, test_result in enumerate(result.details["test_results"]):
        print(f"Test {i+1}:")
        print(f"  Passed: {test_result['passed']}")
        print(f"  Tests run: {test_result['tests_run']}")
        print(f"  Tests passed: {test_result['tests_passed']}")
        if test_result['error']:
            print(f"  Error: {test_result['error']}")
        print()

if __name__ == "__main__":
    test_real_evaluation()