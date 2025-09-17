# LLM Evaluation Pipeline Restructuring Plan
## Simplified Modular Refactoring for Multi-Model Support

**Author**: sdodlapa  
**Date**: September 17, 2025  
**Objective**: Transform monolithic evaluation pipeline into clean, maintainable architecture supporting all planned model families  
**Strategy**: Minimal, focused refactoring with immediate value and low complexity

---

## 🎯 Current State Analysis

### Existing Architecture Issues
- **Monolithic Structure**: `run_evaluation.py` (1,135 lines) handles everything
- **Hardcoded Model Creation**: Only supports 2 Qwen models with if/else logic
- **Mixed Responsibilities**: Performance, datasets, reporting all in one class
- **Extension Difficulty**: Adding new models requires code changes

### Target Architecture Goals (SIMPLIFIED)
- **Component Separation**: Extract 3 focused modules from monolith
- **Simple Model Registry**: Dictionary-based model loader mapping
- **Easy Extension**: Adding models requires only registry entry
- **Maintainable Size**: Each module 150-250 lines max

---

## 🏗️ Target Architecture Overview (SIMPLIFIED)

### Final Directory Structure
```
llm_evaluation/
├── configs/
│   ├── model_configs.py            # Enhanced with registry (existing)
│   └── registry.py                 # Simple model loader mapping (74 lines)
├── models/
│   ├── base_model.py               # Abstract base class (existing - 382 lines)
│   ├── registry.py                 # Simple model discovery (new - ~100 lines)
│   ├── qwen_implementation.py      # Qwen implementation (existing - 394 lines)
│   └── llama_implementation.py     # Llama implementation (new - ~400 lines)
├── evaluation/
│   ├── run_evaluation.py           # Streamlined main runner (~250 lines)
│   ├── performance.py              # Performance benchmarking (~150 lines)
│   ├── datasets.py                 # Dataset evaluation (~200 lines)
│   ├── reporting.py                # Results management (~150 lines)
│   ├── dataset_manager.py          # Dataset management (existing - 595 lines)
│   ├── metrics.py                  # Evaluation metrics (existing - 876 lines)
│   └── config_manager.py           # Evaluation config (existing - 95 lines)
├── scripts/                        # Utility scripts (existing)
├── docs/                          # Documentation (existing)
├── test_results/                  # Output directory (existing)
└── RESTRUCTURING_PLAN.md          # This document
```

**Total New Files**: 4 (performance.py, datasets.py, reporting.py, models/registry.py)  
**Files Modified**: 2 (run_evaluation.py simplified, configs/model_configs.py enhanced)  
**Complexity**: LOW - Simple extraction and dictionary-based registry

---

## 📋 Implementation Plan (SIMPLIFIED - 1 Week)

### **Phase 1: Extract Core Components (Days 1-3)**
**Goal**: Break down the 1,135-line monolith into focused modules

#### **Step 1.1: Extract Performance Benchmarking**
**Estimated Time**: 4 hours  
**Risk Level**: LOW

**Files to Create:**
- `evaluation/performance.py` (~150 lines)

**Files to Modify:**
- `evaluation/run_evaluation.py` (remove performance methods, add import)

**Methods to Move:**
- `_run_performance_benchmark()`
- Performance-related helper methods

**Implementation:**
```python
# evaluation/performance.py
class PerformanceBenchmark:
    def run_benchmark(self, model: BaseModelImplementation) -> Optional[ModelPerformanceMetrics]:
        # Move existing _run_performance_benchmark logic here
        pass
```

**Validation Command:**
```bash
python evaluation/run_evaluation.py --models qwen3_8b --preset balanced --datasets HumanEval --sample-limit 5
```

#### **Step 1.2: Extract Dataset Evaluation**
**Estimated Time**: 6 hours  
**Risk Level**: MEDIUM

**Files to Create:**
- `evaluation/datasets.py` (~200 lines)

**Files to Modify:**
- `evaluation/run_evaluation.py` (remove dataset methods, add import)

**Methods to Move:**
- `_run_dataset_evaluation()`
- `_evaluate_on_single_dataset()`
- `_create_prompt_from_sample()`
- `_calculate_summary_scores()`

**Implementation:**
```python
# evaluation/datasets.py
class DatasetEvaluator:
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
    
    def evaluate_datasets(self, model, preset="balanced", **kwargs):
        # Move existing dataset evaluation logic here
        pass
```

#### **Step 1.3: Extract Results Management**
**Estimated Time**: 4 hours  
**Risk Level**: LOW

**Files to Create:**
- `evaluation/reporting.py` (~150 lines)

**Files to Modify:**
- `evaluation/run_evaluation.py` (remove result methods, add import)

**Methods to Move:**
- `_save_individual_results()`
- `_create_comparison_report()`
- `_create_summary_report()`
- Report generation logic

**Implementation:**
```python
# evaluation/reporting.py
class ResultsManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
    def save_results(self, model_name, results, preset="balanced"):
        # Move existing result saving logic here
        pass
```

### **Phase 2: Simple Model Registry (Day 4)**
**Goal**: Make adding new models easy through simple registry

#### **Step 2.1: Create Model Registry**
**Estimated Time**: 6 hours  
**Risk Level**: LOW

**Files to Create:**
- `models/registry.py` (~100 lines)

**Files to Modify:**
- `evaluation/run_evaluation.py` (use registry for model creation)

**Implementation:**
```python
# models/registry.py
from models.qwen_implementation import create_qwen3_8b, create_qwen3_14b

MODEL_REGISTRY = {
    "qwen3_8b": {
        "family": "qwen",
        "loader": create_qwen3_8b,
        "config_key": "qwen3_8b"
    },
    "qwen3_14b": {
        "family": "qwen", 
        "loader": create_qwen3_14b,
        "config_key": "qwen3_14b"
    }
    # Easy to add new models here!
}

def get_model_loader(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]["loader"]

def get_available_models():
    return list(MODEL_REGISTRY.keys())
```

### **Phase 3: Streamline Main Runner (Day 5)**
**Goal**: Clean up run_evaluation.py to use new components

#### **Step 3.1: Refactor Main Runner**
**Estimated Time**: 8 hours  
**Risk Level**: MEDIUM

**Files to Modify:**
- `evaluation/run_evaluation.py` (major simplification)

**Target Size**: 250 lines (down from 1,135)

**New Structure:**
```python
# evaluation/run_evaluation.py (simplified)
from .performance import PerformanceBenchmark
from .datasets import DatasetEvaluator
from .reporting import ResultsManager
from models.registry import get_model_loader

class LLMEvaluationRunner:
    def __init__(self, output_dir="results", cache_dir=None, data_cache_dir=None):
        self.performance = PerformanceBenchmark()
        self.datasets = DatasetEvaluator(EvaluationDatasetManager(data_cache_dir))
        self.reporting = ResultsManager(output_dir)
        
    def run_individual_evaluation(self, model_name, model_config, preset="balanced", **kwargs):
        # Create model using registry
        model_loader = get_model_loader(model_name)
        model = model_loader(preset=preset, cache_dir=self.cache_dir)
        
        # Use extracted components
        perf_results = self.performance.run_benchmark(model)
        dataset_results = self.datasets.evaluate_datasets(model, preset, **kwargs)
        
        # Save results
        combined_results = {**perf_results, **dataset_results}
        self.reporting.save_results(model_name, combined_results, preset)
        
        return combined_results
```

---

## 🧪 Testing Strategy

### Validation Commands

#### **After Each Step:**
```bash
# Basic functionality test
python evaluation/run_evaluation.py --models qwen3_8b --preset balanced --datasets HumanEval --save-predictions --sample-limit 5

# Multi-model test
python evaluation/run_evaluation.py --models qwen3_8b,qwen3_14b --preset balanced --datasets HumanEval,MBPP --sample-limit 3
```

#### **Final Validation:**
```bash
# Test with new model (after registry implementation)
python evaluation/run_evaluation.py --models qwen3_8b,llama3_8b --preset balanced --datasets HumanEval --sample-limit 3

# Performance check
time python evaluation/run_evaluation.py --models qwen3_8b --preset balanced --datasets HumanEval --sample-limit 10
```

---

## 📊 Benefits Summary

### **Before vs After:**

| Aspect | Current | After Refactoring |
|--------|---------|-------------------|
| **Main File Size** | 1,135 lines | ~250 lines |
| **Modularity** | Monolithic | 4 focused components |
| **Adding Models** | Code changes required | Registry entry only |
| **Testing** | Hard to unit test | Easy component testing |
| **Maintenance** | Single large file | Small, focused modules |
| **Time to Implement** | N/A | 1 week |
| **Risk** | N/A | LOW |

---

## 🎯 Getting Started

### **Day 1: Start Implementation**

Ready to begin? Let's start with **Step 1.1: Extract Performance Benchmarking**

```bash
# Create feature branch
git checkout -b simplify-refactoring

# First step: Extract performance.py
# (Implementation provided below)
```