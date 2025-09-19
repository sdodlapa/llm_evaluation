# LLM Evaluation Framework - CLI Quick Reference Guide

## 🚀 Quick Start Commands

### Most Common Operations

```bash
# List all available categories and their status
python category_evaluation.py --list-categories

# List all models organized by category  
python category_evaluation.py --list-models

# Evaluate all coding specialists (recommended starting point)
python category_evaluation.py --category coding_specialists --samples 5

# Test specific model on its category datasets
python category_evaluation.py --model qwen3_8b --samples 5

# Test specific model-dataset combination
python category_evaluation.py --model qwen3_8b --dataset humaneval --samples 5

# Preview what would be evaluated (dry run)
python category_evaluation.py --category coding_specialists --samples 5 --dry-run
```

---

## 📋 Command Reference

### Information Commands (No Execution)

| Command | Purpose | Example Output |
|---------|---------|----------------|
| `--list-categories` | Show all categories with readiness status | ✅ coding_specialists: READY<br>❌ mathematical_reasoning: NOT READY |
| `--list-models` | Show models grouped by category | CODING_SPECIALISTS:<br>- qwen3_8b<br>- qwen3_14b<br>- qwen3_coder |
| `--category-info <name>` | Detailed category information | Models, datasets, configuration details |
| `--validate` | Check all category readiness | Summary of missing datasets, model availability |

### Evaluation Commands

| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| `--category <name>` | Evaluate all models in category | Systematic category testing |
| `--model <name>` | Evaluate model on category datasets | Single model focus |
| `--model <name> --dataset <name>` | Specific model-dataset pair | Targeted testing |

### Modifiers & Options

| Argument | Default | Options | Purpose |
|----------|---------|---------|---------|
| `--samples <n>` | 5 | 1-1000 | Samples per dataset |
| `--preset <name>` | balanced | balanced, performance, memory_optimized | Performance profile |
| `--include-optional` | False | - | Include optional datasets |
| `--models <list>` | All | Space-separated | Filter models in category |
| `--exclude-models <list>` | None | Space-separated | Exclude specific models |
| `--dry-run` | False | - | Preview without execution |
| `--output-dir <path>` | category_evaluation_results | Any path | Output directory |

---

## 🎯 Common Use Cases

### 1. **Initial System Validation**
```bash
# Check what's available
python category_evaluation.py --list-categories
python category_evaluation.py --validate

# Quick test of working category
python category_evaluation.py --category coding_specialists --samples 3 --dry-run
python category_evaluation.py --category coding_specialists --samples 3
```

### 2. **Model Performance Testing**
```bash
# Test single model thoroughly
python category_evaluation.py --model qwen3_8b --samples 10

# Compare models in category
python category_evaluation.py --category coding_specialists --samples 5

# Test with different presets
python category_evaluation.py --model qwen3_8b --samples 5 --preset performance
python category_evaluation.py --model qwen3_8b --samples 5 --preset memory_optimized
```

### 3. **Dataset-Specific Testing**
```bash
# Test all category models on specific dataset
python category_evaluation.py --category coding_specialists --dataset humaneval --samples 5

# Test specific model-dataset combination
python category_evaluation.py --model qwen3_8b --dataset humaneval --samples 10
```

### 4. **Subset Evaluation**
```bash
# Include only specific models in category
python category_evaluation.py --category coding_specialists --models qwen3_8b qwen3_14b --samples 5

# Exclude problematic models
python category_evaluation.py --category coding_specialists --exclude-models problematic_model --samples 5
```

### 5. **Large-Scale Evaluation**
```bash
# Include optional datasets for comprehensive testing
python category_evaluation.py --category coding_specialists --include-optional --samples 5

# High sample count for thorough evaluation
python category_evaluation.py --category coding_specialists --samples 20
```

---

## ⚙️ Configuration Presets

### **balanced** (Default)
- **Purpose:** Reliable performance with reasonable resource usage
- **Memory:** 85% GPU utilization
- **Batch Size:** 64 sequences
- **Use When:** Standard evaluation, production testing

### **performance**  
- **Purpose:** Maximum throughput and speed
- **Memory:** 95% GPU utilization  
- **Batch Size:** 256 sequences
- **Use When:** Benchmarking, time-critical evaluation

### **memory_optimized**
- **Purpose:** Minimal memory usage
- **Memory:** 75% GPU utilization
- **Batch Size:** 32 sequences  
- **Use When:** Large models, limited VRAM, stability focus

---

## 📁 Output Structure

```
category_evaluation_results/
├── evaluation_session_YYYYMMDD_HHMMSS.json    # Session log
├── detailed_results_YYYYMMDD_HHMMSS.json      # Detailed results
└── summary_YYYYMMDD_HHMMSS.txt                # Human-readable summary
```

### Session Log Contents
```json
{
  "session_id": "evaluation_session_20250918_140530",
  "start_time": "2025-09-18T14:05:30",
  "end_time": "2025-09-18T14:15:45", 
  "parameters": {
    "category": "coding_specialists",
    "samples": 5,
    "preset": "balanced"
  },
  "results": [
    {
      "model": "qwen3_8b",
      "dataset": "humaneval", 
      "success": true,
      "result": { /* detailed metrics */ }
    }
  ]
}
```

---

## 🛠️ Troubleshooting

### **Common Error: "Category not found"**
```bash
# Check available categories
python category_evaluation.py --list-categories

# Verify spelling (case-sensitive)
python category_evaluation.py --category coding_specialists  # ✅ Correct
python category_evaluation.py --category Coding_Specialists  # ❌ Wrong case
```

### **Common Error: "AttributeError: 'dict' object has no attribute 'models'"**
```bash
# This indicates a data structure issue (should be fixed now)
# If you see this, check that model_categories.py was updated correctly
```

### **Common Error: "No models remaining after filtering"**
```bash
# Check if model names are correct
python category_evaluation.py --list-models

# Verify model is in the specified category
python category_evaluation.py --category-info coding_specialists
```

### **Common Error: "ModuleNotFoundError"**
```bash
# Ensure running from project root
cd /home/sdodl001_odu_edu/llm_evaluation

# Check if virtual environment is activated
crun -p ~/envs/llm_env python category_evaluation.py --list-categories
```

### **Performance Issues**
```bash
# Use memory_optimized preset for large models
python category_evaluation.py --model large_model --preset memory_optimized

# Reduce sample count for initial testing
python category_evaluation.py --category coding_specialists --samples 2

# Use dry-run to verify configuration before execution
python category_evaluation.py --category coding_specialists --dry-run
```

---

## 📈 Best Practices

### **1. Start Small**
```bash
# Always start with dry-run
python category_evaluation.py --category coding_specialists --samples 3 --dry-run

# Use small sample counts initially
python category_evaluation.py --category coding_specialists --samples 3
```

### **2. Incremental Testing**
```bash
# Test single model first
python category_evaluation.py --model qwen3_8b --samples 5

# Then expand to category
python category_evaluation.py --category coding_specialists --samples 5

# Finally scale up
python category_evaluation.py --category coding_specialists --samples 20 --include-optional
```

### **3. Monitor Resources**
```bash
# Check GPU memory usage
nvidia-smi

# Use appropriate preset for your system
python category_evaluation.py --model large_model --preset memory_optimized
```

### **4. Save Important Results**
```bash
# Use custom output directory for important runs
python category_evaluation.py --category coding_specialists --samples 20 --output-dir important_results

# Keep session logs for analysis
ls category_evaluation_results/evaluation_session_*.json
```

---

## 🔍 Quick Diagnostics

### **System Health Check**
```bash
# 1. Check categories
python category_evaluation.py --list-categories

# 2. Validate readiness  
python category_evaluation.py --validate

# 3. Check models
python category_evaluation.py --list-models

# 4. Test working category
python category_evaluation.py --category coding_specialists --samples 1 --dry-run
```

### **Debug Specific Issues**
```bash
# Check category details
python category_evaluation.py --category-info coding_specialists

# Test single model-dataset combination
python category_evaluation.py --model qwen3_8b --dataset humaneval --samples 1

# Check output directory permissions
ls -la category_evaluation_results/
```

---

## 📚 Related Commands

### **Other Evaluation Scripts**
```bash
# Simple model evaluation (legacy)
python simple_model_evaluation.py

# Show available datasets  
python show_datasets.py

# Show available models
python show_models.py

# Dataset management
python manage_datasets.py --help
```

### **Configuration Validation**
```bash
# Validate model configurations
python -c "from configs.model_configs import MODEL_CONFIGS; print(len(MODEL_CONFIGS))"

# Check dataset availability
python -c "from evaluation.dataset_manager import EnhancedDatasetManager; dm = EnhancedDatasetManager(); print(dm.get_all_datasets())"
```

---

**Quick Reference Version:** 1.0  
**Last Updated:** September 18, 2025  
**For detailed documentation, see:** `CLI_ARCHITECTURE_DOCUMENTATION.md`