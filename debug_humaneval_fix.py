#!/usr/bin/env python3
"""
Fix for HumanEval test case execution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import EvaluationMetrics

def _execute_humaneval_tests(code: str, test_code: str, timeout: int = 5) -> dict:
    """Execute HumanEval test cases in the check(candidate) format"""
    import ast
    
    # Extract function from code
    try:
        tree = ast.parse(code)
        function_name = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break
        
        if not function_name:
            return {"passed": False, "error": "No function found", "tests_run": 0, "tests_passed": 0}
        
    except SyntaxError as e:
        return {"passed": False, "error": f"Syntax error: {e}", "tests_run": 0, "tests_passed": 0}
    
    # Create execution environment
    exec_globals = {}
    
    try:
        # Execute the solution code
        exec(code, exec_globals)
        
        if function_name not in exec_globals:
            return {"passed": False, "error": f"Function {function_name} not found after execution", 
                   "tests_run": 0, "tests_passed": 0}
        
        candidate_func = exec_globals[function_name]
        
        # Execute the test code with the candidate function
        test_globals = {"candidate": candidate_func}
        exec(test_code, test_globals)
        
        # Find and execute the check function
        if "check" in test_globals:
            check_func = test_globals["check"]
            check_func(candidate_func)  # This will raise AssertionError if any test fails
            
            # If we get here, all tests passed
            # Count the number of assert statements to estimate tests run
            test_lines = test_code.split('\n')
            assert_count = sum(1 for line in test_lines if 'assert' in line.strip())
            
            return {
                "passed": True,
                "tests_run": assert_count,
                "tests_passed": assert_count,
                "error": None
            }
        else:
            return {"passed": False, "error": "No check function found in test code", 
                   "tests_run": 0, "tests_passed": 0}
        
    except AssertionError:
        # Some test failed
        test_lines = test_code.split('\n')
        assert_count = sum(1 for line in test_lines if 'assert' in line.strip())
        
        return {
            "passed": False,
            "tests_run": assert_count,
            "tests_passed": 0,  # We don't know how many passed before the failure
            "error": "Test assertion failed"
        }
        
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "tests_run": 0,
            "tests_passed": 0
        }

def test_fixed_humaneval():
    """Test the fixed HumanEval execution"""
    
    # Sample response (same as before)
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
    
    print("=== Testing Fixed HumanEval Execution ===")
    print()
    
    # Extract code using our function
    extracted_code = EvaluationMetrics._extract_code_from_response(sample_response)
    print(f"Extracted code ({len(extracted_code)} chars)")
    print()
    
    # Test with fixed execution
    result = _execute_humaneval_tests(extracted_code, test_code)
    print("Execution result:")
    print(f"  Passed: {result['passed']}")
    print(f"  Tests run: {result['tests_run']}")
    print(f"  Tests passed: {result['tests_passed']}")
    if result['error']:
        print(f"  Error: {result['error']}")
    
    if result['passed']:
        print("\n✅ SUCCESS: Fixed execution works!")
    else:
        print(f"\n❌ FAILED: {result['error']}")

if __name__ == "__main__":
    test_fixed_humaneval()