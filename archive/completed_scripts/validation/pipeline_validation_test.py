#!/usr/bin/env python3
"""
Comprehensive Pipeline Validation Test
Tests all models on their optimal datasets with 5 samples each to verify modular architecture
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add project paths
sys.path.append('evaluation')
sys.path.append('models') 
sys.path.append('configs')

from evaluation.run_evaluation import LLMEvaluationRunner
from configs.model_configs import get_all_model_configs
from evaluation.dataset_manager import EnhancedDatasetManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineValidationTest:
    """Comprehensive pipeline validation testing"""
    
    def __init__(self, output_dir: str = "test_results/pipeline_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get available models and datasets
        self.all_models = get_all_model_configs()
        self.dataset_manager = EnhancedDatasetManager()
        self.implemented_datasets = self.dataset_manager.get_implemented_datasets()
        
        # Define optimal model-dataset pairings based on specialization
        self.test_plan = self._create_test_plan()
        
    def _create_test_plan(self) -> List[Tuple[str, str, str]]:
        """Create optimal model-dataset test combinations"""
        
        # Model specialization mapping to datasets
        test_combinations = []
        
        # Core Qwen models with coding datasets
        qwen_coding_models = ['qwen3_8b', 'qwen3_14b', 'qwen_8b', 'qwen_14b', 'qwen25_7b', 'qwen25_14b']
        coding_datasets = ['humaneval', 'mbpp', 'bigcodebench']
        
        for model in qwen_coding_models:
            if model in self.all_models:
                for dataset in coding_datasets:
                    if dataset in self.implemented_datasets:
                        test_combinations.append((model, dataset, 'balanced'))
        
        # Specialized coding models
        specialized_coding = [
            ('qwen3_coder_30b', 'humaneval', 'balanced'),
            ('qwen3_coder_30b', 'bigcodebench', 'balanced'),
            ('deepseek_coder_16b', 'humaneval', 'balanced'),
            ('deepseek_coder_16b', 'mbpp', 'balanced'),
        ]
        
        # Math specialization
        math_combinations = [
            ('qwen25_math_7b', 'gsm8k', 'balanced'),
            ('qwen25_math_14b', 'gsm8k', 'balanced'),
        ]
        
        # Biomedical specialization  
        biomedical_combinations = [
            ('biomistral_7b', 'bioasq', 'balanced'),
            ('biomistral_7b', 'pubmedqa', 'balanced'),
            ('biogpt_large', 'pubmedqa', 'balanced'),
            ('clinical_t5_large', 'bioasq', 'balanced'),
        ]
        
        # Efficiency models
        efficiency_combinations = [
            ('qwen25_3b', 'humaneval', 'memory_optimized'),
            ('qwen25_0_5b', 'humaneval', 'memory_optimized'),
            ('phi35_mini_efficiency', 'humaneval', 'memory_optimized'),
        ]
        
        # Reasoning models
        reasoning_combinations = [
            ('qwen3_8b', 'gsm8k', 'balanced'),
            ('qwen3_14b', 'gsm8k', 'balanced'),
            ('granite_3_1_8b', 'gsm8k', 'balanced'),
            ('llama31_8b', 'gsm8k', 'balanced'),
        ]
        
        # Combine all test plans
        all_combinations = (
            test_combinations + 
            specialized_coding + 
            math_combinations + 
            biomedical_combinations + 
            efficiency_combinations + 
            reasoning_combinations
        )
        
        # Filter to only include available models and datasets
        valid_combinations = []
        for model, dataset, preset in all_combinations:
            if model in self.all_models and dataset in self.implemented_datasets:
                valid_combinations.append((model, dataset, preset))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_combinations = []
        for combo in valid_combinations:
            if combo not in seen:
                seen.add(combo)
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def run_validation_test(self) -> Dict[str, Any]:
        """Run comprehensive pipeline validation"""
        
        start_time = time.time()
        
        print(f"🚀 COMPREHENSIVE PIPELINE VALIDATION TEST")
        print(f"=" * 60)
        print(f"📊 Total Test Combinations: {len(self.test_plan)}")
        print(f"🔬 Samples per test: 5")
        print(f"⏱️  Estimated time: {len(self.test_plan) * 2} minutes")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        results = {
            'test_start_time': datetime.now().isoformat(),
            'test_plan': self.test_plan,
            'total_combinations': len(self.test_plan),
            'sample_size': 5,
            'results': {},
            'summary': {
                'successful': 0,
                'failed': 0,
                'errors': []
            }
        }
        
        # Run each test combination
        for i, (model_name, dataset_name, preset) in enumerate(self.test_plan, 1):
            test_id = f"{model_name}_{dataset_name}_{preset}"
            
            print(f"[{i:2d}/{len(self.test_plan)}] Testing {model_name} on {dataset_name} ({preset})")
            
            try:
                # Run individual evaluation
                result = self._run_single_test(model_name, dataset_name, preset)
                results['results'][test_id] = result
                results['summary']['successful'] += 1
                
                print(f"        ✅ SUCCESS - {result.get('samples_processed', 0)} samples")
                
            except Exception as e:
                error_msg = str(e)
                print(f"        ❌ FAILED - {error_msg}")
                
                results['results'][test_id] = {
                    'status': 'failed',
                    'error': error_msg,
                    'model': model_name,
                    'dataset': dataset_name,
                    'preset': preset
                }
                results['summary']['failed'] += 1
                results['summary']['errors'].append(f"{test_id}: {error_msg}")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        results['test_end_time'] = datetime.now().isoformat()
        results['total_duration_seconds'] = total_time
        results['average_time_per_test'] = total_time / len(self.test_plan)
        
        # Save results
        results_file = self.output_dir / f"pipeline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _run_single_test(self, model_name: str, dataset_name: str, preset: str) -> Dict[str, Any]:
        """Run a single model-dataset evaluation"""
        
        try:
            # Create evaluation runner
            runner = LLMEvaluationRunner(
                output_dir=str(self.output_dir / "individual_results"),
                cache_dir=None
            )
            
            # Run evaluation with 5 samples
            result = runner.run_individual_evaluation(
                model_name=model_name,
                model_config=self.all_models[model_name],
                preset=preset,
                save_predictions=False,
                prediction_count=0,
                dataset_filter=[dataset_name],
                sample_limit=5
            )
            
            return {
                'status': 'success',
                'model': model_name,
                'dataset': dataset_name,
                'preset': preset,
                'samples_processed': 5,
                'evaluation_result': result
            }
            
        except Exception as e:
            logger.error(f"Test failed for {model_name} on {dataset_name}: {e}")
            raise
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""
        
        print(f"\\n" + "=" * 60)
        print(f"📊 PIPELINE VALIDATION TEST SUMMARY")
        print(f"=" * 60)
        
        # Overall statistics
        total = results['total_combinations']
        successful = results['summary']['successful']  
        failed = results['summary']['failed']
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {total}")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   ⏱️  Total Time: {results['total_duration_seconds']:.1f}s")
        print(f"   📊 Avg Time/Test: {results['average_time_per_test']:.1f}s")
        
        # Category analysis
        print(f"\\n🔍 TEST CATEGORY ANALYSIS:")
        categories = {}
        
        for test_id, result in results['results'].items():
            if result['status'] == 'success':
                model = result['model']
                
                # Categorize models
                if 'qwen' in model:
                    category = 'Qwen Family'
                elif 'bio' in model or 'clinical' in model:
                    category = 'Biomedical'
                elif 'coder' in model or 'deepseek' in model:
                    category = 'Coding Specialist'
                elif 'math' in model:
                    category = 'Math Specialist'
                else:
                    category = 'Other'
                
                if category not in categories:
                    categories[category] = {'success': 0, 'total': 0}
                categories[category]['success'] += 1
            
            # Count total for category
            model = result['model']
            if 'qwen' in model:
                category = 'Qwen Family'
            elif 'bio' in model or 'clinical' in model:
                category = 'Biomedical'
            elif 'coder' in model or 'deepseek' in model:
                category = 'Coding Specialist'
            elif 'math' in model:
                category = 'Math Specialist'
            else:
                category = 'Other'
            
            if category not in categories:
                categories[category] = {'success': 0, 'total': 0}
            categories[category]['total'] += 1
        
        for category, stats in categories.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   • {category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Error analysis
        if results['summary']['errors']:
            print(f"\\n❌ ERROR ANALYSIS:")
            for i, error in enumerate(results['summary']['errors'][:5], 1):  # Show first 5 errors
                print(f"   {i}. {error}")
            
            if len(results['summary']['errors']) > 5:
                print(f"   ... and {len(results['summary']['errors']) - 5} more errors")
        
        # Recommendations
        print(f"\\n🎯 RECOMMENDATIONS:")
        if success_rate >= 95:
            print("   ✅ EXCELLENT - Pipeline is production ready!")
        elif success_rate >= 80:
            print("   ⚠️  GOOD - Minor issues to investigate")
        elif success_rate >= 60:
            print("   🔧 NEEDS WORK - Significant issues detected")
        else:
            print("   🚨 CRITICAL - Major pipeline problems")
        
        print(f"\\n📁 Detailed results saved to: {self.output_dir}")


def main():
    """Main execution function"""
    
    # Initialize and run validation test
    validator = PipelineValidationTest()
    results = validator.run_validation_test()
    
    # Return appropriate exit code
    success_rate = (results['summary']['successful'] / results['total_combinations'] * 100) if results['total_combinations'] > 0 else 0
    
    if success_rate >= 80:
        print(f"\\n🎉 VALIDATION PASSED - Pipeline is working correctly!")
        sys.exit(0)
    else:
        print(f"\\n⚠️  VALIDATION CONCERNS - Please review failed tests")
        sys.exit(1)


if __name__ == "__main__":
    main()