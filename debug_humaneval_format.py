#!/usr/bin/env python3
"""
Debug script to test actual HumanEval test case format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import EvaluationMetrics

# First HumanEval example with actual test cases
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
```

This function first sorts the list of numbers to ensure that we only need to check consecutive elements for the closest pair."""

# Actual HumanEval test cases from the dataset
actual_test_cases = """

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

def debug_actual_humaneval():
    print("=== Debug Actual HumanEval Test Format ===")
    print()
    
    # Step 1: Extract code
    print("1. Extracting code...")
    extracted_code = EvaluationMetrics._extract_code_from_response(sample_response)
    print(f"Extracted code ({len(extracted_code)} chars):")
    print(extracted_code)
    print()
    
    # Step 2: Test with actual format
    print("2. Testing with actual HumanEval format...")
    
    # Execute the extracted code
    exec_globals = {}
    try:
        exec(extracted_code, exec_globals)
        print("✅ Code executed successfully")
        
        if 'has_close_elements' in exec_globals:
            func = exec_globals['has_close_elements']
            print("✅ Function 'has_close_elements' found")
            
            # Now execute the test cases
            test_globals = {'candidate': func}
            exec(actual_test_cases, test_globals)
            
            # Run the check function
            check_func = test_globals['check']
            check_func(func)
            print("✅ All test cases passed!")
            
        else:
            print("❌ Function not found in extracted code")
            print("Available:", list(exec_globals.keys()))
            
    except AssertionError as e:
        print(f"❌ Test case failed: {e}")
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 3: Test the sorted approach vs expected approach
    print("3. Testing our sorted approach vs expected behavior...")
    
    # Our implementation (sorted)
    def our_has_close_elements(numbers, threshold):
        numbers_copy = numbers.copy()  # Don't modify original
        numbers_copy.sort()
        for i in range(len(numbers_copy) - 1):
            if numbers_copy[i + 1] - numbers_copy[i] < threshold:
                return True
        return False
    
    # Expected implementation (brute force)
    def expected_has_close_elements(numbers, threshold):
        for idx, elem in enumerate(numbers):
            for idx2, elem2 in enumerate(numbers):
                if idx != idx2:
                    distance = abs(elem - elem2)
                    if distance < threshold:
                        return True
        return False
    
    # Test cases
    test_inputs = [
        ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3),
        ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05),
        ([1.0, 2.0, 5.9, 4.0, 5.0], 0.95),
        ([1.0, 2.0, 5.9, 4.0, 5.0], 0.8),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1),
        ([1.1, 2.2, 3.1, 4.1, 5.1], 1.0),
        ([1.1, 2.2, 3.1, 4.1, 5.1], 0.5),
    ]
    
    print("Comparing implementations:")
    for numbers, threshold in test_inputs:
        our_result = our_has_close_elements(numbers, threshold)
        expected_result = expected_has_close_elements(numbers, threshold)
        match = our_result == expected_result
        status = "✅" if match else "❌"
        print(f"{status} {numbers}, {threshold}: Our={our_result}, Expected={expected_result}")

if __name__ == "__main__":
    debug_actual_humaneval()