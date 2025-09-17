````markdown
# LLM Evaluation Framework

A comprehensive evaluation framework for Large Language Models focusing on models under 16B parameters, **optimized for H100 GPU with AWQ-Marlin quantization**.

## 🚀 Quick Start

```bash
# Set up environment
./scripts/setup_environment.sh

# Install packages
./scripts/install_packages.sh
```

## 📊 Current Status

**✅ Production Ready** - Major performance breakthrough achieved!

### **Latest Results**
- **Qwen-3 8B**: 119+ tok/s, comprehensive evaluation complete
- **Qwen-3 14B**: 126+ tok/s with AWQ-Marlin (926% improvement), ready for evaluation
- **Memory Efficiency**: 58-66% VRAM savings with optimized quantization

## 📚 Documentation

**Primary Tracker**: [QWEN_EVALUATION_TRACKER.md](./QWEN_EVALUATION_TRACKER.md) - Live results and progress  
**Documentation Index**: [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md) - Complete navigation guide  
**Architecture**: [ARCHITECTURE.md](./ARCHITECTURE.md) - System design  
**Datasets**: [DATASETS.md](./DATASETS.md) - Benchmark specifications

### **Major Breakthrough**
See [AWQ_PERFORMANCE_BREAKTHROUGH.md](./docs/AWQ_PERFORMANCE_BREAKTHROUGH.md) for details on the 926% performance improvement achieved via AWQ-Marlin kernel optimization.

## 🏗️ Project Structure

- `configs/` - Model configuration files with preset optimization
- `docs/` - Technical documentation and breakthrough reports
- `evaluation/` - Modular evaluation pipeline (4 core modules)
- `models/` - Model implementation classes with registry pattern
- `scripts/` - Setup and installation automation
- `archive/` - Archived redundant files (cleaned up)

## 🎯 GPU Requirements

- **Primary**: H100 GPU (80GB VRAM) - optimal performance
- **Quantization**: AWQ-Marlin kernel for best speed/memory trade-off
- **Memory Usage**: 9-14GB per model with quantization

## 📋 Ready for Production

The framework has been **fully validated** and is ready for:
- ✅ Large-scale dataset evaluation runs
- ✅ Multi-model comparative analysis  
- ✅ Production agent system integration
- ✅ Quantized model deployment

*Last Updated: September 17, 2025 - Post-consolidation and AWQ-Marlin optimization*
````