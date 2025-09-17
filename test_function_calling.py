#!/usr/bin/env python3
"""Test function calling extraction and accuracy"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))

from metrics import EvaluationMetrics

def test_function_calling():
    """Test function calling extraction and evaluation"""
    
    # Test different response formats that models might generate
    test_responses = [
        # JSON format (OpenAI style)
        '''I need to calculate the tip. Let me use the calculate function:

```json
{
  "function_call": {
    "name": "calculate", 
    "arguments": {"expression": "67.50 * 0.18"}
  }
}
```''',
        
        # XML format
        '''<function_call>
<name>get_weather</name>
<arguments>{"location": "Tokyo"}</arguments>
</function_call>''',
        
        # Natural language with structured info
        '''I'll send an email using the send_email function with the following parameters:
- to: team@company.com
- subject: meeting''',
        
        # Direct function call syntax
        '''calculate(expression="67.50 * 0.18")''',
        
        # Mixed format
        '''To solve this, I need to call:
Function: calculate
Arguments: {"expression": "67.50 * 0.18"}'''
    ]
    
    print("Testing function call extraction...")
    print("=" * 50)
    
    for i, response in enumerate(test_responses):
        print(f"Test {i+1}:")
        print(f"Response: {response[:100]}...")
        print()
        
        extracted = EvaluationMetrics._extract_function_calls(response)
        print(f"Extracted calls: {extracted}")
        print("-" * 30)
        print()
    
    # Test the accuracy evaluation
    predictions = [test_responses[0], test_responses[1]]
    expected_calls = [
        [{"name": "calculate", "arguments": {"expression": "67.50 * 0.18"}}],
        [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
    ]
    
    print("Testing function calling accuracy...")
    try:
        result = EvaluationMetrics.function_calling_accuracy(predictions, expected_calls)
        print(f"Accuracy: {result.score:.2%}")
        print(f"Details: {result.details}")
    except Exception as e:
        print(f"Error in accuracy calculation: {e}")

if __name__ == "__main__":
    test_function_calling()