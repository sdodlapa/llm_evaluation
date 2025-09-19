#!/usr/bin/env python3
"""
Consolidated Category Validation Tests
====================================

Comprehensive testing suite for validating model categories, mappings, and
evaluation pipeline functionality. Consolidates functionality from multiple
individual test files.

Usage:
    python tests/category_validation_tests.py --test-category biomedical
    python tests/category_validation_tests.py --test-mapping
    python tests/category_validation_tests.py --test-models
    python tests/category_validation_tests.py --all
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lazy imports for faster startup
def lazy_import_framework():
    """Import framework components only when needed"""
    from configs.model_configs import MODEL_CONFIGS, get_model_config
    from evaluation.mappings.model_categories import CATEGORY_REGISTRY, get_category_for_model
    from category_evaluation import lazy_import_category_system
    return MODEL_CONFIGS, get_model_config, CATEGORY_REGISTRY, get_category_for_model, lazy_import_category_system

class CategoryValidationTests:
    """Consolidated test suite for category validation"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def test_biomedical_category(self, quick_test: bool = True) -> Dict[str, Any]:
        """Test biomedical category models and datasets"""
        print("\n🧬 Testing Biomedical Category...")
        
        try:
            MODEL_CONFIGS, get_model_config, CATEGORY_REGISTRY, get_category_for_model, _ = lazy_import_framework()
            
            # Test biomedical models
            biomedical_models = CATEGORY_REGISTRY.get("BIOMEDICAL_SPECIALISTS", {}).get("models", [])
            results = {
                "category": "BIOMEDICAL_SPECIALISTS",
                "models_found": len(biomedical_models),
                "models": biomedical_models,
                "model_configs": {},
                "validation_status": "success"
            }
            
            # Validate each model configuration
            for model_name in biomedical_models:
                try:
                    config = get_model_config(model_name)
                    results["model_configs"][model_name] = {
                        "exists": True,
                        "model_path": config.model_path if config else None,
                        "quantization": config.quantization if config else None
                    }
                except Exception as e:
                    results["model_configs"][model_name] = {
                        "exists": False,
                        "error": str(e)
                    }
            
            if quick_test:
                print(f"✅ Found {len(biomedical_models)} biomedical models")
                print(f"✅ Validated {len([m for m in results['model_configs'].values() if m.get('exists')])} configurations")
            else:
                # Run actual evaluation test (would be implemented for full testing)
                print("ℹ️  Full evaluation test not implemented in quick mode")
            
            self.results["tests"]["biomedical_category"] = results
            return results
            
        except Exception as e:
            error_result = {"error": str(e), "validation_status": "failed"}
            self.results["tests"]["biomedical_category"] = error_result
            print(f"❌ Biomedical category test failed: {e}")
            return error_result
    
    def test_category_mapping(self) -> Dict[str, Any]:
        """Test category mapping system"""
        print("\n🗺️  Testing Category Mapping System...")
        
        try:
            MODEL_CONFIGS, get_model_config, CATEGORY_REGISTRY, get_category_for_model, _ = lazy_import_framework()
            
            results = {
                "total_categories": len(CATEGORY_REGISTRY),
                "categories": list(CATEGORY_REGISTRY.keys()),
                "total_models": len(MODEL_CONFIGS),
                "mapped_models": 0,
                "unmapped_models": [],
                "category_distribution": {},
                "validation_status": "success"
            }
            
            # Test mapping for all models
            for model_name in MODEL_CONFIGS.keys():
                category = get_category_for_model(model_name)
                if category:
                    results["mapped_models"] += 1
                    if category not in results["category_distribution"]:
                        results["category_distribution"][category] = 0
                    results["category_distribution"][category] += 1
                else:
                    results["unmapped_models"].append(model_name)
            
            print(f"✅ Found {results['total_categories']} categories")
            print(f"✅ Mapped {results['mapped_models']}/{results['total_models']} models")
            if results["unmapped_models"]:
                print(f"⚠️  Unmapped models: {results['unmapped_models']}")
            
            self.results["tests"]["category_mapping"] = results
            return results
            
        except Exception as e:
            error_result = {"error": str(e), "validation_status": "failed"}
            self.results["tests"]["category_mapping"] = error_result
            print(f"❌ Category mapping test failed: {e}")
            return error_result
    
    def test_model_configurations(self) -> Dict[str, Any]:
        """Test model configuration validity"""
        print("\n🤖 Testing Model Configurations...")
        
        try:
            MODEL_CONFIGS, get_model_config, _, _, _ = lazy_import_framework()
            
            results = {
                "total_models": len(MODEL_CONFIGS),
                "valid_configs": 0,
                "invalid_configs": 0,
                "config_details": {},
                "validation_status": "success"
            }
            
            for model_name, config in MODEL_CONFIGS.items():
                try:
                    # Basic validation
                    has_path = bool(config.model_path)
                    has_valid_quantization = config.quantization in [None, "awq", "gptq", "fp16", "bf16"]
                    
                    if has_path and has_valid_quantization:
                        results["valid_configs"] += 1
                        status = "valid"
                    else:
                        results["invalid_configs"] += 1
                        status = "invalid"
                    
                    results["config_details"][model_name] = {
                        "status": status,
                        "has_path": has_path,
                        "quantization": config.quantization,
                        "model_path": config.model_path
                    }
                    
                except Exception as e:
                    results["invalid_configs"] += 1
                    results["config_details"][model_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            print(f"✅ Valid configurations: {results['valid_configs']}")
            print(f"❌ Invalid configurations: {results['invalid_configs']}")
            
            self.results["tests"]["model_configurations"] = results
            return results
            
        except Exception as e:
            error_result = {"error": str(e), "validation_status": "failed"}
            self.results["tests"]["model_configurations"] = error_result
            print(f"❌ Model configuration test failed: {e}")
            return error_result
    
    def test_scientific_models(self) -> Dict[str, Any]:
        """Test scientific models configuration for Phase 2"""
        print("\n🔬 Testing Scientific Models (Phase 2 Preparation)...")
        
        try:
            MODEL_CONFIGS, get_model_config, _, _, _ = lazy_import_framework()
            
            # Check for scientific models
            scientific_models = [
                name for name in MODEL_CONFIGS.keys() 
                if any(keyword in name.lower() for keyword in ['sci', 'bio', 'medical', 'genomic'])
            ]
            
            results = {
                "scientific_models_found": len(scientific_models),
                "models": scientific_models,
                "phase_2_readiness": len(scientific_models) > 0,
                "validation_status": "success"
            }
            
            print(f"✅ Found {len(scientific_models)} scientific models")
            print(f"✅ Phase 2 readiness: {'Ready' if results['phase_2_readiness'] else 'Not Ready'}")
            
            self.results["tests"]["scientific_models"] = results
            return results
            
        except Exception as e:
            error_result = {"error": str(e), "validation_status": "failed"}
            self.results["tests"]["scientific_models"] = error_result
            print(f"❌ Scientific models test failed: {e}")
            return error_result
    
    def run_all_tests(self, quick_test: bool = True) -> Dict[str, Any]:
        """Run all validation tests"""
        print("🧪 Running All Category Validation Tests...")
        print("=" * 50)
        
        # Run all test methods
        self.test_biomedical_category(quick_test)
        self.test_category_mapping()
        self.test_model_configurations()
        self.test_scientific_models()
        
        # Generate summary
        total_tests = len(self.results["tests"])
        successful_tests = len([t for t in self.results["tests"].values() if t.get("validation_status") == "success"])
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "success" if successful_tests == total_tests else "partial_failure"
        }
        
        print("\n📊 Test Summary:")
        print(f"✅ Successful: {successful_tests}/{total_tests}")
        print(f"❌ Failed: {total_tests - successful_tests}/{total_tests}")
        print(f"📈 Success Rate: {self.results['summary']['success_rate']:.1%}")
        
        return self.results
    
    def save_results(self, output_file: str = None) -> str:
        """Save test results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_results/category_validation_{timestamp}.json"
        
        output_path = PROJECT_ROOT / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        return str(output_path)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Category Validation Tests")
    parser.add_argument("--test-category", choices=["biomedical"], help="Test specific category")
    parser.add_argument("--test-mapping", action="store_true", help="Test category mapping")
    parser.add_argument("--test-models", action="store_true", help="Test model configurations")
    parser.add_argument("--test-scientific", action="store_true", help="Test scientific models")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--full", action="store_true", help="Run full tests (not quick)")
    parser.add_argument("--save-results", help="Save results to file")
    
    args = parser.parse_args()
    
    tester = CategoryValidationTests()
    
    if args.all:
        results = tester.run_all_tests(quick_test=not args.full)
    elif args.test_category == "biomedical":
        results = tester.test_biomedical_category(quick_test=not args.full)
    elif args.test_mapping:
        results = tester.test_category_mapping()
    elif args.test_models:
        results = tester.test_model_configurations()
    elif args.test_scientific:
        results = tester.test_scientific_models()
    else:
        print("No test specified. Use --help for options.")
        return
    
    if args.save_results:
        tester.save_results(args.save_results)

if __name__ == "__main__":
    main()