#!/usr/bin/env python3
"""Test the code extraction functionality"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))

from metrics import EvaluationMetrics

def test_code_extraction():
    """Test the _extract_code_from_response function"""
    
    # Test case 1: Markdown formatted response (most common)
    markdown_response = """Here's the solution:

```python
def add_two_numbers(a, b):
    return a + b
```

This function takes two parameters and returns their sum."""
    
    extracted = EvaluationMetrics._extract_code_from_response(markdown_response)
    print("Test 1 - Markdown format:")
    print(f"Extracted: {repr(extracted)}")
    print()
    
    # Test case 2: Response without markdown formatting
    plain_response = """def multiply(x, y):
    result = x * y
    return result"""
    
    extracted2 = EvaluationMetrics._extract_code_from_response(plain_response)
    print("Test 2 - Plain code:")
    print(f"Extracted: {repr(extracted2)}")
    print()
    
    # Test case 3: Response with explanation mixed in
    mixed_response = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

This function calculates the nth Fibonacci number recursively."""
    
    extracted3 = EvaluationMetrics._extract_code_from_response(mixed_response)
    print("Test 3 - Mixed content:")
    print(f"Extracted: {repr(extracted3)}")
    print()
    
    # Test syntax validation
    print("Testing syntax validation:")
    try:
        import ast
        ast.parse(extracted)
        print("✓ Test 1 - Valid syntax")
    except SyntaxError as e:
        print(f"✗ Test 1 - Syntax error: {e}")
    
    try:
        ast.parse(extracted2)
        print("✓ Test 2 - Valid syntax")
    except SyntaxError as e:
        print(f"✗ Test 2 - Syntax error: {e}")
    
    try:
        ast.parse(extracted3)
        print("✓ Test 3 - Valid syntax")
    except SyntaxError as e:
        print(f"✗ Test 3 - Syntax error: {e}")

if __name__ == "__main__":
    test_code_extraction()