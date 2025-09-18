# Documentation Index

**LLM Evaluation Framework - Consolidated Documentation**  
**Last Updated**: September 17, 2025  
**Status**: Ready for Large-Scale Evaluation

---

## 📚 **Core Documentation**

### **Primary Tracker**
- **[QWEN_EVALUATION_TRACKER.md](./QWEN_EVALUATION_TRACKER.md)** - **MAIN DOCUMENT**
  - Live evaluation results and progress tracking
  - Performance metrics and benchmarks
  - Model comparison analysis
  - **Updated with AWQ-Marlin breakthrough results**

### **Technical Architecture**
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System design and modular structure
- **[DATASETS.md](./DATASETS.md)** - Dataset specifications and management
- **[README.md](./README.md)** - Quick start and project overview

### **Breakthrough Documentation**
- **[docs/AWQ_PERFORMANCE_BREAKTHROUGH.md](./docs/AWQ_PERFORMANCE_BREAKTHROUGH.md)** - AWQ-Marlin optimization discovery
- **[docs/LLM_INDIVIDUAL_MODEL_IMPLEMENTATION_PLAN.md](./docs/LLM_INDIVIDUAL_MODEL_IMPLEMENTATION_PLAN.md)** - Implementation strategy

---

## 🗂️ **Current Status Summary**

### **✅ Ready for Production**
- ✅ **Qwen-3 8B**: Fully evaluated (performance + balanced presets)
- ✅ **Qwen-3 14B**: AWQ-Marlin configuration optimized, ready for evaluation
- ✅ **Pipeline Fixed**: HumanEval/MBPP code execution format resolved
- ✅ **Performance Optimized**: 926% speedup achieved via kernel optimization

### **🔧 Recent Optimizations**
1. **AWQ-Marlin Kernel**: 13.68 → 126.70 tok/s (926% improvement)
2. **Memory Efficiency**: 66% VRAM reduction with quantization
3. **Pipeline Fixes**: Code execution format issues resolved
4. **Documentation**: Consolidated and redundancies removed

### **📊 Key Metrics**
- **Qwen-3 8B Performance**: 119+ tok/s, 14.25GB VRAM
- **Qwen-3 8B Balanced**: 119+ tok/s, 5.87GB VRAM (58% memory savings)
- **Qwen-3 14B AWQ-Marlin**: 126+ tok/s, 9.38GB VRAM (66% memory savings)

### **🎯 Ready for Next Phase**
- [ ] **Large Dataset Runs**: Full evaluation with optimized configurations
- [ ] **Multi-Model Testing**: Expand beyond Qwen family
- [ ] **Production Deployment**: Agent system integration

---

## 🗃️ **Archived Documentation**

**Location**: `./archive/`

### **Superseded Files**
- `PROGRESS_REPORT.md` - Outdated status reports
- `AWQ_INVESTIGATION.md` - Superseded by breakthrough documentation
- `docs/README.md` - Minimal content, redundant

### **Duplicate Test Results**
- `test_final/` - Duplicate preset comparison reports
- `test_results/` - Duplicate preset comparison reports  
- `preset_comparison/` - Duplicate preset comparison reports

---

## 🚀 **Next Actions**

### **Immediate (Ready Now)**
1. **Run Qwen-3 14B evaluation** with AWQ-Marlin configuration
2. **Re-evaluate HumanEval/MBPP** with fixed pipeline (expected significant improvement)
3. **Scale up dataset sizes** for comprehensive benchmarking

### **Short Term**
1. **Complete Qwen family evaluation** (all presets, all datasets)
2. **Add additional model families** (DeepSeek, Llama, etc.)
3. **Production readiness validation**

### **Performance Targets**
- **Memory Efficiency**: <12GB per 14B model with quantization
- **Throughput**: >120 tok/s for agent workflows
- **Accuracy**: >50% on reasoning tasks, >70% on code generation

---

## 📋 **File Organization**

### **Kept (Essential)**
```
├── QWEN_EVALUATION_TRACKER.md     # Main tracker (ACTIVE)
├── ARCHITECTURE.md                 # System design
├── DATASETS.md                     # Dataset specs  
├── README.md                       # Quick start
├── docs/
│   ├── AWQ_PERFORMANCE_BREAKTHROUGH.md  # Major optimization
│   └── LLM_INDIVIDUAL_MODEL_IMPLEMENTATION_PLAN.md
└── DOCUMENTATION_INDEX.md          # This file
```

### **Archived (Redundant)**
```
└── archive/
    ├── PROGRESS_REPORT.md          # Outdated
    ├── AWQ_INVESTIGATION.md        # Superseded
    ├── docs_readme.md              # Redundant
    ├── test_final/                 # Duplicate reports
    ├── test_results/               # Duplicate reports
    └── preset_comparison/          # Duplicate reports
```

---

*This index serves as the single source of truth for navigating the LLM evaluation framework documentation. All redundancies have been resolved and the system is ready for large-scale evaluation runs.*