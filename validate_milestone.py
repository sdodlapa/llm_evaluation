#!/usr/bin/env python3
"""
Milestone Validation Script for Large-Scale Evaluation
Usage: python validate_milestone.py <milestone_number>
"""

import sys
import os
import json
from datetime import datetime
import subprocess

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_check(item, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {item}")
    if details:
        print(f"   └─ {details}")

def validate_milestone_1():
    """Pipeline Validation & Infrastructure Testing"""
    print_header("MILESTONE 1: Pipeline Validation")
    
    success = True
    
    # Check core dependencies
    try:
        import torch
        import vllm
        import datasets
        print_check("Core dependencies (torch, vllm, datasets)", True)
    except ImportError as e:
        print_check("Core dependencies", False, f"Missing: {e}")
        success = False
    
    # Check framework imports
    try:
        from evaluation.run_evaluation import main
        from models.registry import list_available_models
        models = list_available_models()
        print_check("Framework imports", True, f"Found {len(models)} models")
    except Exception as e:
        print_check("Framework imports", False, str(e))
        success = False
    
    # Check essential files
    essential_files = [
        'configs/model_configs.py',
        'evaluation/run_evaluation.py', 
        'evaluation/dataset_manager.py',
        'models/base_model.py'
    ]
    
    for file in essential_files:
        exists = os.path.exists(file)
        print_check(f"Essential file: {file}", exists)
        if not exists:
            success = False
    
    # Check GPU availability (if possible)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_available = 'H100' in result.stdout or 'GPU' in result.stdout
        print_check("GPU availability", gpu_available, "H100 or GPU detected" if gpu_available else "No GPU detected")
    except:
        print_check("GPU availability", False, "nvidia-smi not accessible")
    
    return success

def validate_milestone_2():
    """Model Coverage Testing"""
    print_header("MILESTONE 2: Model Coverage Testing")
    
    success = True
    
    # Check if model coverage was run
    log_exists = any(f.startswith('model_coverage') for f in os.listdir('logs/') if os.path.exists('logs/'))
    print_check("Model coverage log exists", log_exists)
    
    # Check results directory
    results_dir = 'results/evaluations/model_coverage/'
    results_exist = os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0
    print_check("Model coverage results directory", results_exist)
    
    if results_exist:
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        print_check("Model coverage JSON results", len(json_files) > 0, f"Found {len(json_files)} files")
    else:
        success = False
    
    # Check for summary file
    summary_file = os.path.join(results_dir, 'latest_summary.json')
    if os.path.exists(summary_file):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            models_tested = len(data)
            print_check("Model coverage summary", True, f"{models_tested} models tested")
        except:
            print_check("Model coverage summary", False, "Invalid JSON format")
            success = False
    else:
        print_check("Model coverage summary", False, "Summary file missing")
        success = False
    
    return success

def validate_milestone_3():
    """Large-Scale Dataset Evaluation"""
    print_header("MILESTONE 3: Large-Scale Dataset Evaluation")
    
    success = True
    
    # Check evaluation logs
    log_exists = any('evaluation' in f for f in os.listdir('logs/') if os.path.exists('logs/'))
    print_check("Evaluation logs exist", log_exists)
    
    # Check multi-model results
    multi_model_dir = 'results/evaluations/multi_model_comprehensive/'
    multi_model_exists = os.path.exists(multi_model_dir)
    print_check("Multi-model results directory", multi_model_exists)
    
    if multi_model_exists:
        result_files = [f for f in os.listdir(multi_model_dir) if f.endswith('.json')]
        print_check("Multi-model result files", len(result_files) > 0, f"Found {len(result_files)} files")
        
        # Expect at least 6 models × multiple datasets
        expected_min_files = 5  # Conservative estimate
        sufficient_results = len(result_files) >= expected_min_files
        print_check("Sufficient evaluation results", sufficient_results, f"Expected ≥{expected_min_files}, got {len(result_files)}")
        if not sufficient_results:
            success = False
    else:
        success = False
    
    # Check detailed analysis results
    detailed_dirs = ['results/evaluations/qwen3_8b_detailed/', 'results/evaluations/qwen3_14b_detailed/']
    for detailed_dir in detailed_dirs:
        model_name = detailed_dir.split('/')[-2]
        detailed_exists = os.path.exists(detailed_dir)
        print_check(f"Detailed analysis: {model_name}", detailed_exists)
        if not detailed_exists:
            success = False
    
    return success

def validate_milestone_4():
    """Detailed Scaling Analysis"""
    print_header("MILESTONE 4: Scaling Analysis")
    
    success = True
    
    # Check scaling analysis log
    scaling_log = any('scaling' in f for f in os.listdir('logs/') if os.path.exists('logs/'))
    print_check("Scaling analysis log", scaling_log)
    
    # Check comparison results
    comparison_dir = 'results/comparisons/'
    comparison_exists = os.path.exists(comparison_dir)
    print_check("Comparisons directory", comparison_exists)
    
    if comparison_exists:
        # Check for key comparison files
        key_files = [
            '8b_vs_14b_comparison.json',
            'performance_matrix.json',
            'scaling_insights.json'
        ]
        
        for file in key_files:
            file_path = os.path.join(comparison_dir, file)
            file_exists = os.path.exists(file_path)
            print_check(f"Comparison file: {file}", file_exists)
            if not file_exists:
                success = False
                
        # Validate scaling insights if it exists
        insights_file = os.path.join(comparison_dir, 'scaling_insights.json')
        if os.path.exists(insights_file):
            try:
                with open(insights_file) as f:
                    insights = json.load(f)
                has_scaling_data = 'parameter_scaling' in insights
                print_check("Scaling insights content", has_scaling_data)
                if not has_scaling_data:
                    success = False
            except:
                print_check("Scaling insights content", False, "Invalid JSON")
                success = False
    else:
        success = False
    
    return success

def validate_milestone_5():
    """Documentation & Results Integration"""
    print_header("MILESTONE 5: Documentation & Integration")
    
    success = True
    
    # Check comprehensive summary
    summary_file = 'results/evaluations/comprehensive_summary.json'
    summary_exists = os.path.exists(summary_file)
    print_check("Comprehensive summary", summary_exists)
    
    if summary_exists:
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            has_required_fields = all(field in summary for field in ['evaluation_date', 'total_models', 'total_evaluations'])
            print_check("Summary content validation", has_required_fields)
            if not has_required_fields:
                success = False
        except:
            print_check("Summary content validation", False, "Invalid JSON")
            success = False
    else:
        success = False
    
    # Check final report
    final_report = 'results/LARGE_SCALE_EVALUATION_FINAL_REPORT.md'
    report_exists = os.path.exists(final_report)
    print_check("Final evaluation report", report_exists)
    if not report_exists:
        success = False
    
    # Check tracker update content
    tracker_update = 'tracker_update_content.md'
    tracker_content_exists = os.path.exists(tracker_update)
    print_check("Tracker update content", tracker_content_exists)
    
    # Check results structure
    required_dirs = [
        'results/evaluations/',
        'results/comparisons/',
        'logs/'
    ]
    
    for dir_path in required_dirs:
        dir_exists = os.path.exists(dir_path)
        print_check(f"Results directory: {dir_path}", dir_exists)
        if not dir_exists:
            success = False
    
    # Count total result files
    total_files = 0
    for root, dirs, files in os.walk('results/'):
        total_files += len([f for f in files if f.endswith('.json') or f.endswith('.md')])
    
    sufficient_files = total_files >= 10  # Conservative minimum
    print_check("Sufficient result files", sufficient_files, f"Found {total_files} result files")
    if not sufficient_files:
        success = False
    
    return success

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_milestone.py <milestone_number>")
        print("Milestone numbers: 1-5")
        sys.exit(1)
    
    try:
        milestone = int(sys.argv[1])
    except ValueError:
        print("Error: Milestone number must be an integer (1-5)")
        sys.exit(1)
    
    if milestone not in range(1, 6):
        print("Error: Milestone number must be between 1 and 5")
        sys.exit(1)
    
    print(f"🚀 MILESTONE {milestone} VALIDATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validation_functions = {
        1: validate_milestone_1,
        2: validate_milestone_2,
        3: validate_milestone_3,
        4: validate_milestone_4,
        5: validate_milestone_5
    }
    
    success = validation_functions[milestone]()
    
    print(f"\n{'='*60}")
    if success:
        print(f"🎉 MILESTONE {milestone} VALIDATION: ✅ PASSED")
        print("Ready to proceed to next milestone")
        sys.exit(0)
    else:
        print(f"🚨 MILESTONE {milestone} VALIDATION: ❌ FAILED")
        print("Review issues above before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()