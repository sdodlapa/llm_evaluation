# LLM Evaluation Pipeline: Cleanup & Consolidation Plan

**Date**: September 17, 2025  
**Goal**: Create clean, efficient, modular evaluation pipeline  
**Strategy**: Quick testing → Systematic expansion → Clean architecture  

## 🎯 **PHASE 1: IMMEDIATE QUICK TEST (Priority 1)**

### **Step 1.1: Cancel Current Wasteful Evaluation**
```bash
# Kill redundant comprehensive evaluation
pkill -f "run_full_comprehensive_evaluation.py"
```

### **Step 1.2: Create Ultra-Fast Test Script**
**Target**: 2 models × 12 datasets × 20 samples = 24 evaluations (~2 hours)

**New Script**: `run_quick_all_datasets.py`
- ✅ Only `balanced` preset (proven optimal)
- ✅ 20 samples per dataset (ultra-fast)
- ✅ All 12 datasets (6 implemented + 6 placeholders)
- ✅ Only Qwen models (qwen3_8b, qwen3_14b)
- ✅ Simple, clean, no redundancy

### **Step 1.3: Handle Missing Datasets**
**Strategy**: Create minimal placeholder data for 6 unimplemented datasets
- Generate simple test samples for quick validation
- Focus on pipeline testing, not dataset perfection

---

## 🧹 **PHASE 2: CODE CONSOLIDATION (Priority 2)**

### **Step 2.1: Script Cleanup**
**Current Mess**:
```
❌ run_comprehensive_evaluation.py      (redundant)
❌ run_full_comprehensive_evaluation.py (redundant) 
❌ run_optimal_evaluation.py            (redundant)
❌ compare_8b_vs_14b.py                 (specific use case)
```

**Clean Target**:
```
✅ run_evaluation.py                    (single entry point)
✅ config/evaluation_config.py          (centralized config)
✅ utils/evaluation_utils.py            (shared utilities)
```

### **Step 2.2: Configuration Consolidation**
**Current Issues**:
- 8 models in config, only 2 implemented
- Sample counts scattered across files
- Preset selection logic duplicated

**Clean Target**:
```python
# config/evaluation_config.py
EVALUATION_CONFIG = {
    "implemented_models": ["qwen3_8b", "qwen3_14b"],
    "planned_models": ["deepseek_coder_16b", "llama31_8b", ...],
    "optimal_presets": {"qwen3_8b": "balanced", "qwen3_14b": "balanced"},
    "all_datasets": [
        # Implemented
        "humaneval", "mbpp", "gsm8k", "arc_challenge", "mt_bench", "hellaswag",
        # Unimplemented  
        "math", "bfcl", "toolllama", "mmlu", "ifeval", "winogrande"
    ],
    "sample_counts": {
        "quick_test": 20,
        "standard_test": 100,
        "full_test": 200
    }
}
```

### **Step 2.3: Module Responsibilities**
**Clean Architecture**:
```
evaluation/
├── dataset_manager.py      (handles all 12 datasets)
├── model_manager.py        (handles model loading/unloading)
├── evaluator.py           (core evaluation logic)
├── performance_monitor.py  (monitoring & metrics)
└── results_manager.py     (save/load/analyze results)

config/
├── model_configs.py       (model definitions)
└── evaluation_config.py   (evaluation settings)

scripts/
├── run_evaluation.py      (main entry point)
├── analyze_results.py     (result analysis)
└── compare_models.py      (model comparison)
```

---

## 📊 **PHASE 3: SYSTEMATIC EXPANSION (Priority 3)**

### **Step 3.1: Dataset Implementation Priority**
1. **High Priority**: `math`, `mmlu` (academic benchmarks)
2. **Medium Priority**: `bfcl`, `toolllama` (function calling)
3. **Low Priority**: `ifeval`, `winogrande` (specialized tasks)

### **Step 3.2: Model Implementation Order**
1. **Phase 3a**: Complete Qwen models (qwen3_8b, qwen3_14b)
2. **Phase 3b**: Add coding specialist (deepseek_coder_16b)
3. **Phase 3c**: Add general models (llama31_8b, mistral_7b)
4. **Phase 3d**: Add research models (olmo2_13b, yi_9b, phi35_mini)

### **Step 3.3: Preset Optimization Strategy**
**Current Approach**: Test all 3 presets per model
**New Approach**: Smart preset selection
- Start with `balanced` for all models
- Only test other presets if balanced shows issues
- Use performance data to guide preset choices

---

## 🛠 **PHASE 4: TECHNICAL DEBT CLEANUP (Priority 4)**

### **Step 4.1: Code Quality Issues**
**Fix**:
- Remove duplicate evaluation logic
- Standardize error handling
- Improve logging consistency
- Add proper type hints
- Create comprehensive tests

### **Step 4.2: Performance Optimizations**
**Optimize**:
- Model loading/unloading efficiency
- Memory management between evaluations
- Parallel dataset processing (where safe)
- Result caching and incremental evaluation

### **Step 4.3: Documentation & Usability**
**Create**:
- Clear README with usage examples
- API documentation
- Configuration guide
- Troubleshooting guide
- Performance tuning guide

---

## 📋 **IMPLEMENTATION TIMELINE**

### **Week 1: Quick Test & Immediate Cleanup**
- [ ] **Day 1**: Cancel current job, create quick test script
- [ ] **Day 2**: Run 24-combination quick test (2 models × 12 datasets × 20 samples)
- [ ] **Day 3**: Analyze results, identify issues
- [ ] **Day 4-5**: Consolidate scripts, clean configuration

### **Week 2: Architecture Cleanup**
- [ ] **Day 1-2**: Restructure modules, remove redundancy
- [ ] **Day 3-4**: Implement missing datasets (basic versions)
- [ ] **Day 5**: Test consolidated pipeline

### **Week 3: Systematic Expansion**
- [ ] **Day 1-2**: Add remaining models one by one
- [ ] **Day 3-4**: Optimize performance and memory usage
- [ ] **Day 5**: Full pipeline validation

### **Week 4: Polish & Documentation**
- [ ] **Day 1-2**: Code quality improvements
- [ ] **Day 3-4**: Documentation and guides
- [ ] **Day 5**: Final testing and validation

---

## 🎯 **SUCCESS METRICS**

### **Efficiency Gains**:
- ✅ **Evaluation time**: 24 combinations in ~2 hours (vs current 36 in ~5 hours)
- ✅ **Code complexity**: 1 main script (vs 4+ current scripts)
- ✅ **Configuration**: Centralized config (vs scattered settings)

### **Quality Improvements**:
- ✅ **Modularity**: Clear separation of concerns
- ✅ **Maintainability**: Single source of truth for each component
- ✅ **Extensibility**: Easy to add new models/datasets
- ✅ **Reliability**: Consistent error handling and logging

### **Coverage Goals**:
- ✅ **Models**: 2 working → 8 planned models
- ✅ **Datasets**: 6 working → 12 total datasets  
- ✅ **Presets**: Smart selection vs brute force testing

---

## 🚀 **IMMEDIATE NEXT STEPS**

1. **Cancel current job**: `pkill -f "run_full_comprehensive_evaluation.py"`
2. **Create quick test script**: 20 samples × 12 datasets × 2 models
3. **Run quick test**: Validate pipeline with minimal resources
4. **Begin consolidation**: Remove redundant scripts
5. **Plan systematic expansion**: Add models/datasets incrementally

**Estimated Time to Complete Phase 1**: 2-3 hours  
**Estimated GPU Time for Quick Test**: 2 hours  
**Total Efficiency Gain**: 70%+ time savings