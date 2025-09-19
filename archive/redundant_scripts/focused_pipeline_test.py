#!/usr/bin/env python3
"""
Focused Pipeline Test - All Models on Their Optimal Datasets
============================================================

Runs each model on its specialized category datasets with 5 samples.
This provides comprehensive validation across all model specializations
while keeping runtime manageable.

Model-Dataset Mapping:
- Qwen Family → Coding (HumanEval, MBPP, BigCodeBench)
- Coding Specialists → Coding datasets  
- Math Specialists → Mathematical reasoning (GSM8K)
- Biomedical → Biomedical datasets (BioASQ, PubMedQA)
- Efficiency Models → General coding tasks
- Other → Best fit datasets
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.run_evaluation import run_single_evaluation
from evaluation.dataset_manager import get_dataset_manager
from configs.model_configs import get_all_model_configs
from evaluation.json_serializer import safe_json_dump

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FocusedPipelineTest:
    """Focused pipeline test for model-dataset specialization validation"""
    
    def __init__(self, samples_per_test: int = 5):
        self.samples_per_test = samples_per_test
        self.output_dir = Path("test_results/focused_pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset manager
        self.dataset_manager = get_dataset_manager()
        
        # Get all model configurations
        self.all_configs = get_all_model_configs()
        
        # Define model-dataset specialization mapping
        self.specialization_mapping = self._create_specialization_mapping()
        
    def _create_specialization_mapping(self) -> Dict[str, List[str]]:
        """Create mapping of models to their optimal datasets"""
        
        # Define dataset categories
        coding_datasets = ["humaneval", "mbpp", "bigcodebench"]
        math_datasets = ["gsm8k"]
        biomedical_datasets = ["bioasq", "pubmedqa"]
        
        mapping = {}
        
        for model_name, config in self.all_configs.items():
            # Determine optimal datasets based on model characteristics
            if "qwen" in model_name.lower():
                # Qwen models are strong at coding
                mapping[model_name] = coding_datasets
                
            elif "coder" in model_name.lower() or "deepseek" in model_name.lower():
                # Coding specialists
                mapping[model_name] = coding_datasets
                
            elif "math" in model_name.lower() or "wizard" in model_name.lower():
                # Math specialists  
                mapping[model_name] = math_datasets
                
            elif ("bio" in model_name.lower() or "clinical" in model_name.lower() or 
                  "medical" in model_name.lower()):
                # Biomedical specialists
                mapping[model_name] = biomedical_datasets
                
            elif ("granite" in model_name.lower() or "llama" in model_name.lower() or
                  "mistral" in model_name.lower()):
                # General models - test on math for reasoning
                mapping[model_name] = math_datasets
                
            elif ("0_5b" in model_name or "3b" in model_name or 
                  "efficiency" in model_name.lower()):
                # Efficiency models - lighter coding tasks
                mapping[model_name] = ["humaneval"]  # Single dataset for efficiency
                
            else:
                # Default to coding for unknown models
                mapping[model_name] = ["humaneval"]
                
        return mapping
    
    def generate_test_plan(self) -> List[Tuple[str, str, str]]:
        """Generate list of (model_name, dataset, preset) test combinations"""
        test_plan = []
        
        for model_name, datasets in self.specialization_mapping.items():
            if model_name in self.all_configs:
                config = self.all_configs[model_name]
                
                # Use model's default preset or balanced
                preset = getattr(config, 'preset', 'balanced')
                
                for dataset in datasets:
                    # Verify dataset exists
                    if self.dataset_manager.is_dataset_available(dataset):
                        test_plan.append((model_name, dataset, preset))
                    else:
                        logger.warning(f"Dataset {dataset} not available, skipping")
        
        return test_plan
    
    def run_single_test(self, model_name: str, dataset: str, preset: str) -> Dict:
        """Run a single model-dataset test"""
        logger.info(f"Testing {model_name} on {dataset} ({preset})")
        
        start_time = time.time()
        success = False
        error_message = None
        results = None
        
        try:
            # Run evaluation
            results = run_single_evaluation(
                model_name=model_name,
                dataset_name=dataset,
                num_samples=self.samples_per_test,
                preset=preset,
                output_dir=str(self.output_dir),
                save_results=True
            )
            
            success = results is not None
            if not success:
                error_message = "Evaluation returned None"
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Test failed for {model_name} on {dataset}: {e}")
        
        duration = time.time() - start_time
        
        return {
            "model": model_name,
            "dataset": dataset, 
            "preset": preset,
            "success": success,
            "duration": duration,
            "error": error_message,
            "results": results
        }
    
    def categorize_models(self, test_plan: List[Tuple[str, str, str]]) -> Dict[str, int]:
        """Categorize models by type for reporting"""
        categories = {
            "Qwen Family": 0,
            "Coding Specialist": 0,
            "Math Specialist": 0,
            "Biomedical": 0,
            "Efficiency": 0,
            "General": 0
        }
        
        model_names = set(test[0] for test in test_plan)
        
        for model_name in model_names:
            if "qwen" in model_name.lower():
                categories["Qwen Family"] += 1
            elif "coder" in model_name.lower() or "deepseek" in model_name.lower():
                categories["Coding Specialist"] += 1
            elif "math" in model_name.lower() or "wizard" in model_name.lower():
                categories["Math Specialist"] += 1
            elif ("bio" in model_name.lower() or "clinical" in model_name.lower() or 
                  "medical" in model_name.lower()):
                categories["Biomedical"] += 1
            elif ("0_5b" in model_name or "3b" in model_name):
                categories["Efficiency"] += 1
            else:
                categories["General"] += 1
                
        return categories
    
    def run_focused_validation(self):
        """Run the focused pipeline validation test"""
        logger.info("🚀 FOCUSED PIPELINE VALIDATION TEST")
        logger.info("=" * 60)
        
        # Generate test plan
        test_plan = self.generate_test_plan()
        
        # Get unique models count
        unique_models = len(set(test[0] for test in test_plan))
        categories = self.categorize_models(test_plan)
        
        logger.info(f"📊 Total Test Combinations: {len(test_plan)}")
        logger.info(f"🤖 Unique Models: {unique_models}")
        logger.info(f"🔬 Samples per test: {self.samples_per_test}")
        logger.info(f"⏱️  Estimated time: {len(test_plan) * 2} minutes")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info("")
        
        # Show model categories
        logger.info("🎯 MODEL CATEGORIES:")
        for category, count in categories.items():
            if count > 0:
                logger.info(f"   • {category}: {count} models")
        logger.info("")
        
        # Run tests
        results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, (model_name, dataset, preset) in enumerate(test_plan, 1):
            logger.info(f"[{i:2d}/{len(test_plan)}] Testing {model_name} on {dataset} ({preset})")
            
            result = self.run_single_test(model_name, dataset, preset)
            results.append(result)
            
            if result["success"]:
                successful_tests += 1
                logger.info(f"        ✅ SUCCESS - {self.samples_per_test} samples")
            else:
                failed_tests += 1
                logger.info(f"        ❌ FAILED - {result['error']}")
            
        # Calculate summary statistics
        total_time = sum(r["duration"] for r in results)
        avg_time_per_test = total_time / len(results) if results else 0
        success_rate = (successful_tests / len(results)) * 100 if results else 0
        
        # Category-wise success analysis
        category_results = {}
        for category in categories.keys():
            if categories[category] > 0:
                category_successes = 0
                category_total = 0
                
                for result in results:
                    model_name = result["model"]
                    if ((category == "Qwen Family" and "qwen" in model_name.lower()) or
                        (category == "Coding Specialist" and ("coder" in model_name.lower() or "deepseek" in model_name.lower())) or
                        (category == "Math Specialist" and ("math" in model_name.lower() or "wizard" in model_name.lower())) or
                        (category == "Biomedical" and ("bio" in model_name.lower() or "clinical" in model_name.lower() or "medical" in model_name.lower())) or
                        (category == "Efficiency" and ("0_5b" in model_name or "3b" in model_name)) or
                        (category == "General" and not any(x in model_name.lower() for x in ["qwen", "coder", "deepseek", "math", "wizard", "bio", "clinical", "medical"]) and not any(x in model_name for x in ["0_5b", "3b"]))):
                        
                        category_total += 1
                        if result["success"]:
                            category_successes += 1
                
                if category_total > 0:
                    category_results[category] = {
                        "successful": category_successes,
                        "total": category_total,
                        "rate": (category_successes / category_total) * 100
                    }
        
        # Print comprehensive summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 FOCUSED PIPELINE VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info("🎯 OVERALL RESULTS:")
        logger.info(f"   Total Tests: {len(results)}")
        logger.info(f"   ✅ Successful: {successful_tests}")
        logger.info(f"   ❌ Failed: {failed_tests}")
        logger.info(f"   📈 Success Rate: {success_rate:.1f}%")
        logger.info(f"   ⏱️  Total Time: {total_time:.1f}s")
        logger.info(f"   📊 Avg Time/Test: {avg_time_per_test:.1f}s")
        logger.info("")
        
        logger.info("🔍 CATEGORY ANALYSIS:")
        for category, stats in category_results.items():
            logger.info(f"   • {category}: {stats['successful']}/{stats['total']} ({stats['rate']:.1f}%)")
        logger.info("")
        
        # Save detailed results
        results_file = self.output_dir / "focused_pipeline_results.json"
        detailed_results = {
            "summary": {
                "total_tests": len(results),
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_time": total_time,
                "avg_time_per_test": avg_time_per_test,
                "samples_per_test": self.samples_per_test
            },
            "category_results": category_results,
            "test_results": results,
            "test_plan": test_plan
        }
        
        if not safe_json_dump(detailed_results, results_file, indent=2):
            logger.error(f"Failed to save results: {results_file}")
            # Fallback to basic json
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
        
        # Final recommendations
        if success_rate >= 90:
            logger.info("🎯 RECOMMENDATIONS:")
            logger.info("   ✅ EXCELLENT - Pipeline is production ready!")
        elif success_rate >= 75:
            logger.info("🎯 RECOMMENDATIONS:")
            logger.info("   ⚠️  GOOD - Minor issues to investigate")
        else:
            logger.info("🎯 RECOMMENDATIONS:")
            logger.info("   ❌ NEEDS ATTENTION - Multiple failures detected")
        
        logger.info("")
        logger.info(f"📁 Detailed results saved to: {results_file}")
        logger.info("")
        
        if success_rate >= 90:
            logger.info("🎉 FOCUSED VALIDATION PASSED - Specialized pipeline working correctly!")
        else:
            logger.info("⚠️ FOCUSED VALIDATION COMPLETED - Review failures for optimization")
        
        return detailed_results


def main():
    """Main execution function"""
    try:
        # Create and run focused test
        tester = FocusedPipelineTest(samples_per_test=5)
        results = tester.run_focused_validation()
        
        # Exit with appropriate code
        success_rate = results["summary"]["success_rate"]
        sys.exit(0 if success_rate >= 90 else 1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Focused pipeline test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()