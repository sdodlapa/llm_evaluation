#!/usr/bin/env python3
"""
Calculate metrics for remaining datasets (ARC, HellaSwag, MT-Bench)
This script processes predictions and calculates available metrics.
"""

import json
import os
import sys
import re
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset_and_predictions(dataset_path, predictions_path):
    """Load dataset and predictions"""
    try:
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset_data = json.load(f)
        dataset_samples = dataset_data.get('samples', [])
        dataset_name = dataset_data.get('name', 'Unknown')
        
        # Load predictions
        with open(predictions_path, 'r') as f:
            predictions_data = json.load(f)
        
        model_name = predictions_data.get('model_name', 'Unknown')
        preset = predictions_data.get('preset', 'Unknown')
        predictions = predictions_data.get('predictions', [])
        
        logger.info(f"Loaded {dataset_name}: {len(dataset_samples)} samples, {len(predictions)} predictions")
        return dataset_name, dataset_samples, model_name, preset, predictions
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None, [], None, None, []

def calculate_completion_rate(predictions):
    """Calculate how many predictions contain meaningful responses"""
    meaningful_responses = 0
    
    for pred in predictions:
        response = pred.get('prediction', '').strip()
        
        # Check if response is meaningful (not just error messages)
        if (response and 
            len(response) > 20 and  # At least some content
            not response.startswith("It looks like you didn't provide") and
            not response.startswith("It seems like you forgot") and
            "Could you please provide more details" not in response):
            meaningful_responses += 1
    
    return meaningful_responses / len(predictions) if predictions else 0.0

def calculate_multiple_choice_accuracy(predictions, dataset_samples):
    """Calculate multiple choice accuracy where possible"""
    if not dataset_samples:
        return 0.0, 0, 0
    
    correct = 0
    attempted = 0
    
    for i, pred in enumerate(predictions):
        if i >= len(dataset_samples):
            break
            
        sample = dataset_samples[i]
        response = pred.get('prediction', '').strip()
        
        # Skip if no meaningful response
        if not response or len(response) < 10:
            continue
            
        attempted += 1
        
        # Extract expected answer
        expected_answer = sample.get('answer', '')
        if not expected_answer:
            continue
        
        # Extract choice from response
        extracted_choice = extract_multiple_choice_answer(response)
        
        if extracted_choice and extracted_choice.upper() == expected_answer.upper():
            correct += 1
    
    accuracy = correct / attempted if attempted > 0 else 0.0
    return accuracy, correct, attempted

def extract_multiple_choice_answer(text):
    """Extract multiple choice answer from text"""
    text = text.strip().upper()
    
    # Look for patterns like "Answer: C", "The answer is B", "(A)", etc.
    patterns = [
        r'(?:ANSWER|CHOICE)(?:\s*IS)?:?\s*([A-D])',
        r'^\s*([A-D])\s*[.:)]',
        r'\(([A-D])\)',
        r'\b([A-D])\b(?:\s*[.:)]|\s*$)',
        r'OPTION\s*([A-D])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no pattern matches, check if it's just a single letter
    if len(text) == 1 and text in 'ABCD':
        return text
    
    # Look for the first letter A, B, C, or D
    for char in text:
        if char in 'ABCD':
            return char
    
    return None

def analyze_response_quality(predictions):
    """Analyze the quality of responses"""
    analysis = {
        'total_responses': len(predictions),
        'meaningful_responses': 0,
        'empty_responses': 0,
        'error_responses': 0,
        'avg_length': 0,
        'contains_reasoning': 0
    }
    
    total_length = 0
    
    for pred in predictions:
        response = pred.get('prediction', '').strip()
        
        if not response:
            analysis['empty_responses'] += 1
            continue
        
        total_length += len(response)
        
        # Check for error responses
        if (response.startswith("It looks like you didn't provide") or
            response.startswith("It seems like you forgot") or
            "Could you please provide more details" in response):
            analysis['error_responses'] += 1
            continue
        
        analysis['meaningful_responses'] += 1
        
        # Check for reasoning
        if any(keyword in response.lower() for keyword in 
               ['because', 'therefore', 'since', 'due to', 'reason', 'explain']):
            analysis['contains_reasoning'] += 1
    
    analysis['avg_length'] = total_length / len(predictions) if predictions else 0
    
    return analysis

def print_dataset_results(dataset_name, model_name, preset, analysis, accuracy_data=None):
    """Print detailed results for a dataset"""
    print("\n" + "="*80)
    print(f"🎯 {dataset_name.upper()} EVALUATION RESULTS")
    print(f"📦 Model: {model_name}")
    print(f"⚙️  Preset: {preset}")
    print("="*80)
    
    print(f"\n📊 RESPONSE ANALYSIS:")
    print(f"   📝 Total Responses: {analysis['total_responses']}")
    print(f"   ✅ Meaningful Responses: {analysis['meaningful_responses']} ({analysis['meaningful_responses']/analysis['total_responses']*100:.1f}%)")
    print(f"   ❌ Error Responses: {analysis['error_responses']} ({analysis['error_responses']/analysis['total_responses']*100:.1f}%)")
    print(f"   📏 Average Length: {analysis['avg_length']:.1f} characters")
    print(f"   🧠 Contains Reasoning: {analysis['contains_reasoning']} ({analysis['contains_reasoning']/analysis['total_responses']*100:.1f}%)")
    
    if accuracy_data:
        accuracy, correct, attempted = accuracy_data
        print(f"\n🎯 ACCURACY METRICS:")
        print(f"   ✅ Correct Answers: {correct}/{attempted}")
        print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📈 Completion Rate: {attempted}/{analysis['total_responses']} ({attempted/analysis['total_responses']*100:.1f}%)")
    
    print(f"\n🔍 ANALYSIS:")
    meaningful_rate = analysis['meaningful_responses'] / analysis['total_responses']
    if meaningful_rate >= 0.8:
        print(f"   🚀 EXCELLENT: High response quality and completion rate")
    elif meaningful_rate >= 0.6:
        print(f"   ✅ GOOD: Most responses are meaningful")
    elif meaningful_rate >= 0.4:
        print(f"   📊 MODERATE: Some evaluation issues, but partial success")
    else:
        print(f"   ⚠️  NEEDS INVESTIGATION: Low completion rate suggests evaluation issues")
    
    print("="*80)

def main():
    """Main execution function"""
    print("🚀 Starting Remaining Dataset Metrics Calculation...")
    
    # Define remaining datasets to process
    remaining_datasets = [
        {
            'name': 'ARC-Challenge',
            'predictions_file': 'test_results/predictions/Qwen-3 14B Instruct_balanced_arc_challenge_predictions.json',
            'dataset_file': 'evaluation_data/reasoning/arc_challenge.json'
        },
        {
            'name': 'HellaSwag',
            'predictions_file': 'test_results/predictions/Qwen-3 14B Instruct_balanced_hellaswag_predictions.json',
            'dataset_file': 'evaluation_data/reasoning/hellaswag.json'
        },
        {
            'name': 'MT-Bench',
            'predictions_file': 'test_results/predictions/Qwen-3 14B Instruct_balanced_mt_bench_predictions.json',
            'dataset_file': None  # MT-Bench might not have a dataset file
        }
    ]
    
    all_results = {}
    
    for dataset_info in remaining_datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_info['name']}...")
        print(f"{'='*60}")
        
        # Check if prediction file exists
        if not os.path.exists(dataset_info['predictions_file']):
            logger.warning(f"Predictions file not found: {dataset_info['predictions_file']}")
            continue
        
        # Load data
        if dataset_info['dataset_file'] and os.path.exists(dataset_info['dataset_file']):
            dataset_name, dataset_samples, model_name, preset, predictions = load_dataset_and_predictions(
                dataset_info['dataset_file'], dataset_info['predictions_file'])
        else:
            # Load just predictions
            with open(dataset_info['predictions_file'], 'r') as f:
                predictions_data = json.load(f)
            
            dataset_name = dataset_info['name']
            dataset_samples = []
            model_name = predictions_data.get('model_name', 'Unknown')
            preset = predictions_data.get('preset', 'Unknown')
            predictions = predictions_data.get('predictions', [])
            
            logger.info(f"Loaded {dataset_name} predictions only: {len(predictions)} samples")
        
        if not predictions:
            logger.warning(f"No predictions found for {dataset_info['name']}")
            continue
        
        # Analyze responses
        analysis = analyze_response_quality(predictions)
        
        # Calculate accuracy if we have dataset samples
        accuracy_data = None
        if dataset_samples:
            accuracy_data = calculate_multiple_choice_accuracy(predictions, dataset_samples)
        
        # Print results
        print_dataset_results(dataset_name, model_name, preset, analysis, accuracy_data)
        
        # Save results
        results = {
            'response_analysis': analysis,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'preset': preset
        }
        
        if accuracy_data:
            accuracy, correct, attempted = accuracy_data
            results['accuracy'] = accuracy
            results['correct_answers'] = correct
            results['attempted_answers'] = attempted
        
        output_file = f"test_results/metrics/{dataset_name.lower().replace('-', '_')}_metrics_{model_name.replace(' ', '_')}_{preset}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'preset': preset,
                'dataset': dataset_name,
                'metrics': results,
                'timestamp': str(__import__('datetime').datetime.now())
            }, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        all_results[dataset_name] = results
    
    # Print summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"🎉 REMAINING DATASETS SUMMARY")
        print(f"{'='*80}")
        
        for dataset_name, results in all_results.items():
            analysis = results['response_analysis']
            meaningful_rate = analysis['meaningful_responses'] / analysis['total_responses']
            print(f"   {dataset_name}: {meaningful_rate:.3f} meaningful response rate ({meaningful_rate*100:.1f}%)")
            
            if 'accuracy' in results:
                print(f"      └─ Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print(f"\n🎉 SUCCESS: Remaining dataset metrics calculated!")
            for dataset, metrics in results.items():
                meaningful_rate = metrics['response_analysis']['meaningful_responses'] / metrics['response_analysis']['total_responses']
                print(f"📊 {dataset}: {meaningful_rate:.3f} response quality")
        else:
            print(f"\n⚠️ WARNING: No results calculated")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)