#!/usr/bin/env python3
"""
Calculate MBPP and GSM8K metrics from saved predictions
This script processes predictions from multiple datasets and calculates appropriate metrics.
"""

import json
import os
import sys
import re
import logging
from typing import Dict, List, Any
from evaluation.metrics import EvaluationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(dataset_path):
    """Load any dataset and return samples"""
    try:
        with open(dataset_path, 'r') as f:
            dataset_data = json.load(f)
        
        samples = dataset_data.get('samples', [])
        dataset_name = dataset_data.get('name', 'Unknown')
        logger.info(f"Loaded {dataset_name} dataset with {len(samples)} samples")
        return dataset_name, samples
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        return None, []

def load_predictions(predictions_file):
    """Load model predictions from JSON file"""
    try:
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        model_name = predictions_data.get('model_name', 'Unknown')
        preset = predictions_data.get('preset', 'Unknown')
        dataset_name = predictions_data.get('dataset_name', 'Unknown')
        predictions = predictions_data.get('predictions', [])
        
        logger.info(f"Loaded {dataset_name} predictions for {model_name} ({preset}): {len(predictions)} samples")
        return model_name, preset, dataset_name, predictions
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return None, None, None, []

def calculate_mbpp_metrics(predictions, dataset_samples):
    """Calculate MBPP metrics using code similarity and execution where possible"""
    metrics = EvaluationMetrics()
    results = {}
    
    logger.info(f"Calculating MBPP metrics for {len(predictions)} samples...")
    
    # Extract model responses and expected outputs
    model_responses = [pred['prediction'] for pred in predictions]
    expected_outputs = [sample.get('expected_output', '') for sample in dataset_samples[:len(predictions)]]
    
    # Calculate exact match accuracy (for code structure similarity)
    if expected_outputs:
        exact_match_result = metrics.exact_match(model_responses, expected_outputs)
        results['exact_match'] = exact_match_result.score
        results['exact_match_details'] = exact_match_result.details
    
    # Try to calculate code execution accuracy if test cases exist
    test_cases = [sample.get('test_cases', []) for sample in dataset_samples[:len(predictions)]]
    if any(test_cases):
        try:
            execution_result = metrics.code_execution_accuracy(model_responses, test_cases, "MBPP")
            results['code_execution'] = execution_result.score
            results['execution_details'] = execution_result.details
        except Exception as e:
            logger.warning(f"Could not calculate code execution metrics: {e}")
            results['code_execution'] = 0.0
    
    # Calculate function extraction success rate
    function_extraction_rate = calculate_function_extraction_rate(model_responses)
    results['function_extraction_rate'] = function_extraction_rate
    
    results['total_samples'] = len(predictions)
    results['successful_samples'] = int(results.get('exact_match', 0) * len(predictions))
    
    return results

def calculate_gsm8k_metrics(predictions, dataset_samples):
    """Calculate GSM8K math reasoning metrics"""
    metrics = EvaluationMetrics()
    results = {}
    
    logger.info(f"Calculating GSM8K metrics for {len(predictions)} samples...")
    
    # Extract model responses and expected answers
    model_responses = [pred['prediction'] for pred in predictions]
    expected_answers = [sample.get('answer', '') for sample in dataset_samples[:len(predictions)]]
    
    if not expected_answers:
        logger.warning("No expected answers found in GSM8K dataset")
        return {'error': 'No expected answers found'}
    
    # Extract numerical answers from model responses
    extracted_answers = [extract_numerical_answer(response) for response in model_responses]
    normalized_expected = [extract_numerical_answer(answer) for answer in expected_answers]
    
    # Calculate exact match on numerical answers
    correct_answers = 0
    valid_extractions = 0
    
    for i, (extracted, expected) in enumerate(zip(extracted_answers, normalized_expected)):
        if extracted is not None:
            valid_extractions += 1
            if extracted == expected:
                correct_answers += 1
    
    accuracy = correct_answers / len(predictions) if predictions else 0.0
    extraction_rate = valid_extractions / len(predictions) if predictions else 0.0
    
    results['accuracy'] = accuracy
    results['extraction_rate'] = extraction_rate
    results['correct_answers'] = correct_answers
    results['valid_extractions'] = valid_extractions
    results['total_samples'] = len(predictions)
    results['successful_samples'] = correct_answers
    
    # Show some examples
    results['examples'] = []
    for i in range(min(5, len(predictions))):
        results['examples'].append({
            'extracted': extracted_answers[i],
            'expected': normalized_expected[i],
            'correct': extracted_answers[i] == normalized_expected[i] if extracted_answers[i] is not None else False
        })
    
    return results

def extract_numerical_answer(text):
    """Extract numerical answer from text"""
    # Common patterns for final answers
    patterns = [
        r'(?:the answer is|answer:|final answer:|result:|solution:)\s*(\d+(?:\.\d+)?)',
        r'(?:therefore|thus|so),?\s*(\d+(?:\.\d+)?)',
        r'(?:equals?|=)\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:dollars?|cents?|years?|days?|hours?|minutes?|\$)?\.?\s*$',
        r'\$?\s*(\d+(?:\.\d+)?)',  # Dollar amounts
    ]
    
    text = text.lower().strip()
    
    # Try patterns in order of preference
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                # Take the last match (most likely to be final answer)
                answer = float(matches[-1])
                return answer
            except ValueError:
                continue
    
    # Fallback: extract all numbers and take the last one
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def calculate_function_extraction_rate(responses):
    """Calculate how many responses contain extractable Python functions"""
    function_count = 0
    
    for response in responses:
        if 'def ' in response and '(' in response and ')' in response:
            function_count += 1
    
    return function_count / len(responses) if responses else 0.0

def print_detailed_results(results, model_name, preset, dataset_name):
    """Print detailed results for any dataset"""
    print("\n" + "="*80)
    print(f"🎯 {dataset_name.upper()} EVALUATION RESULTS")
    print(f"📦 Model: {model_name}")
    print(f"⚙️  Preset: {preset}")
    print("="*80)
    
    if dataset_name.upper() == 'MBPP':
        print_mbpp_results(results)
    elif dataset_name.upper() == 'GSM8K':
        print_gsm8k_results(results)
    else:
        print_generic_results(results)
    
    print("="*80)

def print_mbpp_results(results):
    """Print MBPP-specific results"""
    print(f"\n📊 CORE METRICS:")
    if 'exact_match' in results:
        print(f"   🎯 Code Similarity: {results['exact_match']:.4f} ({results['exact_match']*100:.2f}%)")
    if 'code_execution' in results:
        print(f"   ✅ Code Execution: {results['code_execution']:.4f} ({results['code_execution']*100:.2f}%)")
    if 'function_extraction_rate' in results:
        print(f"   🔧 Function Extraction: {results['function_extraction_rate']:.4f} ({results['function_extraction_rate']*100:.2f}%)")
    
    print(f"   📈 Total Samples: {results.get('total_samples', 0)}")
    
    print(f"\n🔍 ANALYSIS:")
    main_score = results.get('code_execution', results.get('exact_match', 0))
    if main_score >= 0.5:
        print(f"   🚀 EXCELLENT: Strong Python code generation capability")
    elif main_score >= 0.3:
        print(f"   ✅ GOOD: Competent code generation with room for improvement")
    elif main_score >= 0.1:
        print(f"   📊 MODERATE: Basic code generation, needs optimization")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Code generation requires significant work")

def print_gsm8k_results(results):
    """Print GSM8K-specific results"""
    print(f"\n📊 CORE METRICS:")
    print(f"   🎯 Mathematical Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   📊 Answer Extraction: {results['extraction_rate']:.4f} ({results['extraction_rate']*100:.2f}%)")
    print(f"   ✅ Correct Answers: {results['correct_answers']}/{results['total_samples']}")
    print(f"   🔢 Valid Extractions: {results['valid_extractions']}/{results['total_samples']}")
    
    print(f"\n🔍 ANALYSIS:")
    if results['accuracy'] >= 0.6:
        print(f"   🚀 EXCELLENT: Strong mathematical reasoning capability")
    elif results['accuracy'] >= 0.4:
        print(f"   ✅ GOOD: Competent mathematical reasoning")
    elif results['accuracy'] >= 0.2:
        print(f"   📊 MODERATE: Basic mathematical reasoning")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Mathematical reasoning requires work")
    
    if 'examples' in results:
        print(f"\n📝 SAMPLE ANSWERS:")
        for i, example in enumerate(results['examples']):
            status = "✅" if example['correct'] else "❌"
            print(f"   {status} Sample {i+1}: Got {example['extracted']}, Expected {example['expected']}")

def print_generic_results(results):
    """Print generic results for other datasets"""
    print(f"\n📊 METRICS:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            if 'rate' in key or 'accuracy' in key or 'score' in key:
                print(f"   • {key}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"   • {key}: {value}")

def main():
    """Main execution function"""
    print("🚀 Starting Multi-Dataset Metrics Calculation...")
    
    # Define datasets to process
    datasets_to_process = [
        {
            'name': 'MBPP',
            'predictions_file': 'test_results/predictions/Qwen-3 14B Instruct_balanced_mbpp_predictions.json',
            'dataset_file': 'evaluation_data/coding/mbpp.json'
        },
        {
            'name': 'GSM8K',
            'predictions_file': 'test_results/predictions/Qwen-3 14B Instruct_balanced_gsm8k_predictions.json',
            'dataset_file': 'evaluation_data/reasoning/gsm8k.json'
        }
    ]
    
    all_results = {}
    
    for dataset_info in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_info['name']}...")
        print(f"{'='*60}")
        
        # Check if files exist
        if not os.path.exists(dataset_info['predictions_file']):
            logger.warning(f"Predictions file not found: {dataset_info['predictions_file']}")
            continue
        
        if not os.path.exists(dataset_info['dataset_file']):
            logger.warning(f"Dataset file not found: {dataset_info['dataset_file']}")
            continue
        
        # Load data
        dataset_name, dataset_samples = load_dataset(dataset_info['dataset_file'])
        model_name, preset, pred_dataset_name, predictions = load_predictions(dataset_info['predictions_file'])
        
        if not predictions or not dataset_samples:
            logger.warning(f"Failed to load data for {dataset_info['name']}")
            continue
        
        # Calculate metrics
        if dataset_info['name'] == 'MBPP':
            results = calculate_mbpp_metrics(predictions, dataset_samples)
        elif dataset_info['name'] == 'GSM8K':
            results = calculate_gsm8k_metrics(predictions, dataset_samples)
        else:
            logger.warning(f"Unknown dataset type: {dataset_info['name']}")
            continue
        
        # Print results
        print_detailed_results(results, model_name, preset, dataset_info['name'])
        
        # Save results
        output_file = f"test_results/metrics/{dataset_info['name'].lower()}_metrics_{model_name.replace(' ', '_')}_{preset}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'preset': preset,
                'dataset': dataset_info['name'],
                'metrics': results,
                'timestamp': str(__import__('datetime').datetime.now())
            }, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        all_results[dataset_info['name']] = results
    
    # Print summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"🎉 OVERALL SUMMARY")
        print(f"{'='*80}")
        
        for dataset_name, results in all_results.items():
            main_metric = 'accuracy' if 'accuracy' in results else 'exact_match'
            if main_metric in results:
                score = results[main_metric]
                print(f"   {dataset_name}: {score:.3f} ({score*100:.1f}%)")
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print(f"\n🎉 SUCCESS: Multi-dataset metrics calculated successfully!")
            for dataset, metrics in results.items():
                main_score = metrics.get('accuracy', metrics.get('exact_match', 0))
                print(f"📊 {dataset}: {main_score:.3f} ({main_score*100:.1f}%)")
        else:
            print(f"\n⚠️ WARNING: No results calculated")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)