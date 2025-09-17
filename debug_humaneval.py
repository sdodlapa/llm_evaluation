#!/usr/bin/env python3
"""
Debug script to analyze HumanEval code execution failures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import EvaluationMetrics

# Sample prediction from the results
sample_response = """To solve this problem, we can sort the list of numbers and then check the difference between consecutive elements. If any two consecutive elements have a difference less than the threshold, then we return `True`. Otherwise, we return `False`.

Here's the implementation:

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

# Test cases
print(has_close_elements([1.0, 2.0, 3.0], 0.5))  # Output: False
print(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # Output: True
```

This function first sorts the list of numbers to ensure that we only need to check consecutive elements for the closest pair. Then, it iterates through the sorted list and checks if the difference between any two consecutive elements is less than the given threshold. If such a pair is found, it returns `True`; otherwise, it returns `False` after checking all pairs."""

# Sample test case (typical HumanEval format)
sample_test_cases = [
    {"input": [[1.0, 2.0, 3.0], 0.5], "output": False},
    {"input": [[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3], "output": True}
]

def debug_code_execution():
    print("=== Debug HumanEval Code Execution ===")
    print()
    
    # Step 1: Test code extraction
    print("1. Testing code extraction...")
    extracted_code = EvaluationMetrics._extract_code_from_response(sample_response)
    print(f"Extracted code length: {len(extracted_code)} characters")
    print("First 200 characters:")
    print(extracted_code[:200])
    print("...")
    print()
    
    # Step 2: Test code execution
    print("2. Testing code execution...")
    try:
        result = EvaluationMetrics._execute_code_tests(extracted_code, sample_test_cases, timeout=5)
        print(f"Execution result: {result}")
        
        if not result["passed"]:
            print("❌ Code execution failed!")
            print(f"Tests run: {result['tests_run']}")
            print(f"Tests passed: {result['tests_passed']}")
            if result["error"]:
                print(f"Error: {result['error']}")
        else:
            print("✅ Code execution successful!")
            
    except Exception as e:
        print(f"❌ Exception during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 3: Manual test of extracted code
    print("3. Manual test of extracted code...")
    try:
        # Execute the code in a clean environment
        exec_globals = {}
        exec(extracted_code, exec_globals)
        
        if 'has_close_elements' in exec_globals:
            func = exec_globals['has_close_elements']
            print("Function found! Testing manually...")
            
            # Test case 1
            result1 = func([1.0, 2.0, 3.0], 0.5)
            print(f"Test 1: has_close_elements([1.0, 2.0, 3.0], 0.5) = {result1} (expected: False)")
            
            # Test case 2  
            result2 = func([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
            print(f"Test 2: has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) = {result2} (expected: True)")
            
        else:
            print("❌ Function 'has_close_elements' not found in executed code")
            print("Available functions:", [k for k in exec_globals.keys() if callable(exec_globals[k])])
            
    except Exception as e:
        print(f"❌ Manual execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_code_execution()