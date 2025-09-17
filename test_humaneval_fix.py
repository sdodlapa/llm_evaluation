#!/usr/bin/env python3
"""Run a quick HumanEval test to verify the fix works in practice"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_humaneval_fix():
    """Run a small HumanEval evaluation to test the fix"""
    
    print("=== Quick HumanEval Fix Test ===")
    print()
    
    # Load a few HumanEval samples
    with open('evaluation_data/coding/humaneval.json', 'r') as f:
        humaneval_data = json.load(f)
    
    # Take first 3 samples
    samples = humaneval_data['samples'][:3]
    
    print(f"Testing {len(samples)} HumanEval samples:")
    for i, sample in enumerate(samples):
        print(f"  {i+1}. {sample['id']}")
    print()
    
    # Initialize model (we'll use a mock for now)
    try:
        from models.qwen_implementation import QwenImplementation
        from configs.model_configs import get_qwen_configs
        
        config = get_qwen_configs()["qwen3_8b"]
        model = QwenImplementation(config)
        
        print("Generating predictions...")
        predictions = []
        
        for i, sample in enumerate(samples):
            try:
                pred = model.generate_response(sample['prompt'])
                predictions.append(pred)
                print(f"  Sample {i+1}: Generated ({len(pred)} chars)")
            except Exception as e:
                print(f"  Sample {i+1}: ERROR - {e}")
                predictions.append("")
                
        print(f"\nGenerated {len([p for p in predictions if p])} successful predictions")
        
        # Now evaluate with the fixed pipeline
        from evaluation.metrics import evaluate_dataset_predictions
        
        results = evaluate_dataset_predictions("coding", predictions, samples)
        
        print("\nEvaluation Results:")
        for metric_name, result in results.items():
            print(f"  {metric_name}: {result.score:.1%} ({result.successful_samples}/{result.total_samples})")
            
            if metric_name == "code_execution":
                test_results = result.details.get("test_results", [])
                for i, test_result in enumerate(test_results):
                    status = "✅ PASS" if test_result['passed'] else "❌ FAIL"
                    print(f"    Sample {i+1}: {status} ({test_result['tests_passed']}/{test_result['tests_run']} tests)")
                    if test_result.get('error'):
                        print(f"      Error: {test_result['error']}")
        
        # Check if we have good results
        if "code_execution" in results and results["code_execution"].score > 0:
            print(f"\n✅ SUCCESS! Fix is working - {results['code_execution'].score:.1%} accuracy")
        else:
            print(f"\n❌ Issue persists - still showing 0% accuracy")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nTest completed!")

if __name__ == "__main__":
    test_humaneval_fix()