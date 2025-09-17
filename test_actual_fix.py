#!/usr/bin/env python3
"""Quick test with actual evaluation results to verify the fix"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.run_evaluation import LLMEvaluationRunner
from evaluation.metrics import evaluate_dataset_predictions

def test_actual_evaluation():
    """Test with actual saved results to verify the fix"""
    
    print("=== Testing with Actual Evaluation Results ===")
    print()
    
    # Load the latest results
    results_file = "results/performance/qwen3_8b_balanced_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check if we have HumanEval results
    if "HumanEval" not in results["dataset_results"]:
        print("❌ No HumanEval results found")
        return
    
    humaneval_results = results["dataset_results"]["HumanEval"]
    print(f"Loaded HumanEval results for {len(humaneval_results.get('predictions', []))} samples")
    
    # Load the HumanEval dataset
    with open('evaluation_data/coding/humaneval.json', 'r') as f:
        humaneval_data = json.load(f)
    
    samples = humaneval_data['samples']
    predictions = humaneval_results.get('predictions', [])
    
    if len(predictions) != len(samples):
        # Take the minimum length
        min_len = min(len(predictions), len(samples))
        samples = samples[:min_len]
        predictions = predictions[:min_len]
        print(f"Adjusted to {min_len} samples to match prediction count")
    
    print(f"Re-evaluating {len(samples)} samples with fixed pipeline...")
    
    # Re-evaluate with the fixed pipeline
    try:
        new_results = evaluate_dataset_predictions("coding", predictions, samples)
        
        print("\nOriginal vs Fixed Results:")
        original_score = humaneval_results.get('metrics', {}).get('code_execution', {}).get('score', 0)
        
        if "code_execution" in new_results:
            new_score = new_results["code_execution"].score
            print(f"  Original code execution: {original_score:.1%}")
            print(f"  Fixed code execution:    {new_score:.1%}")
            print(f"  Improvement:            +{(new_score - original_score) * 100:.1f} percentage points")
            
            if new_score > original_score:
                print(f"\n✅ SUCCESS! The fix dramatically improved HumanEval accuracy!")
                print(f"   From {original_score:.1%} to {new_score:.1%}")
                
                # Show some sample results
                test_results = new_results["code_execution"].details.get("test_results", [])
                passed_count = sum(1 for r in test_results if r.get('passed'))
                print(f"   Sample breakdown: {passed_count}/{len(test_results)} passed in test subset")
            else:
                print(f"\n⚠️  No improvement detected")
        else:
            print("❌ No code execution results in new evaluation")
            
    except Exception as e:
        print(f"❌ Error during re-evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actual_evaluation()