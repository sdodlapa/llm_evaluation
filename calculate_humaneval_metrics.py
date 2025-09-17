#!/usr/bin/env python3
"""
Calculate HumanEval metrics from saved predictions
This script processes the HumanEval predictions from our Qwen-3 14B evaluation
and calculates pass@1 and pass@k scores using code execution.
"""

import json
import os
import sys
import logging
from typing import Dict, List, Any
from evaluation.metrics import EvaluationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_humaneval_dataset():
    """Load the HumanEval dataset to get test cases"""
    dataset_path = "evaluation_data/coding/humaneval.json"
    try:
        with open(dataset_path, 'r') as f:
            dataset_data = json.load(f)
        
        # Extract samples from the dataset structure
        samples = dataset_data.get('samples', [])
        logger.info(f"Loaded HumanEval dataset with {len(samples)} samples")
        return samples
    except Exception as e:
        logger.error(f"Failed to load HumanEval dataset: {e}")
        return []

def load_predictions(predictions_file):
    """Load model predictions from JSON file"""
    try:
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        model_name = predictions_data.get('model_name', 'Unknown')
        preset = predictions_data.get('preset', 'Unknown')
        predictions = predictions_data.get('predictions', [])
        
        logger.info(f"Loaded predictions for {model_name} ({preset}): {len(predictions)} samples")
        return model_name, preset, predictions
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return None, None, []

def extract_test_cases_from_humaneval(dataset):
    """Extract test cases from HumanEval dataset in the correct format"""
    test_cases = []
    
    for sample in dataset:
        # HumanEval test cases are in the 'test_cases' field as executable code
        test_code = sample.get('test_cases', '')
        if test_code:
            test_cases.append(test_code)
        else:
            test_cases.append('')
    
    return test_cases

def calculate_humaneval_pass_at_k(predictions, test_cases, k_values=[1, 10]):
    """Calculate pass@k metrics for HumanEval"""
    metrics = EvaluationMetrics()
    results = {}
    
    logger.info(f"Calculating HumanEval metrics for {len(predictions)} samples...")
    
    # For each prediction, we need to extract the code and run it against test cases
    model_responses = []
    valid_test_cases = []
    
    for i, pred in enumerate(predictions):
        if i < len(test_cases) and test_cases[i]:
            model_responses.append(pred['prediction'])
            valid_test_cases.append(test_cases[i])
    
    logger.info(f"Processing {len(model_responses)} predictions with valid test cases")
    
    # Calculate code execution accuracy (this is essentially pass@1)
    execution_result = metrics.code_execution_accuracy(
        predictions=model_responses,
        test_cases=valid_test_cases,
        dataset_name="HumanEval"
    )
    
    results['pass@1'] = execution_result.score
    results['total_samples'] = execution_result.total_samples
    results['successful_samples'] = execution_result.successful_samples
    results['details'] = execution_result.details
    
    # For pass@k where k > 1, we would need multiple generations per problem
    # Since we only have one prediction per problem, pass@k = pass@1 for k > 1
    for k in k_values:
        if k > 1:
            results[f'pass@{k}'] = execution_result.score  # Same as pass@1
        else:
            results[f'pass@{k}'] = execution_result.score
    
    return results

def print_detailed_results(results, model_name, preset):
    """Print detailed results"""
    print("\n" + "="*80)
    print(f"🎯 HUMANEVAL EVALUATION RESULTS")
    print(f"📦 Model: {model_name}")
    print(f"⚙️  Preset: {preset}")
    print("="*80)
    
    print(f"\n📊 CORE METRICS:")
    print(f"   ✅ Pass@1 Score: {results['pass@1']:.4f} ({results['pass@1']*100:.2f}%)")
    print(f"   📈 Successful Solutions: {results['successful_samples']}/{results['total_samples']}")
    print(f"   🎯 Success Rate: {results['successful_samples']/results['total_samples']*100:.1f}%")
    
    if 'pass@10' in results:
        print(f"   ⭐ Pass@10 Score: {results['pass@10']:.4f} ({results['pass@10']*100:.2f}%)")
    
    print(f"\n🔍 ANALYSIS:")
    if results['pass@1'] >= 0.3:
        print(f"   🚀 EXCELLENT: Score above 30% indicates strong code generation")
    elif results['pass@1'] >= 0.2:
        print(f"   ✅ GOOD: Score above 20% indicates competent code generation")
    elif results['pass@1'] >= 0.1:
        print(f"   📊 MODERATE: Score above 10% indicates basic code generation")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Score below 10% indicates code generation issues")
    
    # Show some sample results
    if 'details' in results and 'test_results' in results['details']:
        sample_results = results['details']['test_results']
        passed_count = sum(1 for r in sample_results if r.get('passed', False))
        print(f"\n📝 SAMPLE ANALYSIS (first {len(sample_results)} samples):")
        print(f"   ✅ Passed: {passed_count}/{len(sample_results)}")
        print(f"   ❌ Failed: {len(sample_results) - passed_count}/{len(sample_results)}")
        
        # Show error types
        error_types = {}
        for r in sample_results:
            if not r.get('passed', False) and 'error' in r and r['error']:
                error_type = r['error'].split(':')[0] if ':' in r['error'] else r['error']
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            print(f"\n🐛 ERROR ANALYSIS:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {error_type}: {count} occurrences")
    
    print("="*80)

def main():
    """Main execution function"""
    print("🚀 Starting HumanEval Metrics Calculation...")
    
    # Find prediction file
    predictions_file = "test_results/predictions/Qwen-3 14B Instruct_balanced_humaneval_predictions.json"
    
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found: {predictions_file}")
        return
    
    # Load dataset and predictions
    dataset = load_humaneval_dataset()
    if not dataset:
        logger.error("Failed to load HumanEval dataset")
        return
    
    model_name, preset, predictions = load_predictions(predictions_file)
    if not predictions:
        logger.error("Failed to load predictions")
        return
    
    # Extract test cases
    test_cases = extract_test_cases_from_humaneval(dataset)
    logger.info(f"Extracted {len(test_cases)} test cases from dataset")
    
    # Calculate metrics
    results = calculate_humaneval_pass_at_k(predictions, test_cases)
    
    # Print results
    print_detailed_results(results, model_name, preset)
    
    # Save results to file
    output_file = f"test_results/metrics/humaneval_metrics_{model_name.replace(' ', '_')}_{preset}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'model_name': model_name,
            'preset': preset,
            'dataset': 'HumanEval',
            'metrics': results,
            'timestamp': str(__import__('datetime').datetime.now())
        }, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        if results and results['pass@1'] > 0:
            print(f"\n🎉 SUCCESS: HumanEval metrics calculated successfully!")
            print(f"📊 Final Pass@1 Score: {results['pass@1']:.4f} ({results['pass@1']*100:.2f}%)")
        else:
            print(f"\n⚠️ WARNING: Metrics calculation completed but scores may be low")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)