# Configuration Directory Structure
# ==============================

This directory contains all configuration files for the LLM evaluation framework.

## Current Configuration Files

### **Core Configuration**
- **`model_configs.py`** - Primary model configuration registry (43 models)
- **`model_registry.py`** - Enhanced model discovery and loading system
- **`preset_configs.py`** - Evaluation preset configurations (performance, balanced, memory)

### **Specialized Configurations**
- **`biomedical_model_dataset_mappings.py`** - Biomedical category specific mappings
- **`scientific_configs.py`** - Scientific research models configuration (Phase 2)
- **`h100_optimization.py`** - H100 GPU specific optimizations

### **Validation & Testing**
- **`config_validation.py`** - Configuration validation utilities

## Configuration Hierarchy

```
configs/
├── model_configs.py              # Primary model registry
├── model_registry.py             # Enhanced model system
├── preset_configs.py             # Evaluation presets
├── biomedical_model_dataset_mappings.py  # Biomedical specialization
├── scientific_configs.py         # Scientific models (Phase 2)
├── h100_optimization.py          # GPU optimizations
└── config_validation.py          # Validation utilities
```

## Usage Examples

```python
# Load primary model configurations
from configs.model_configs import MODEL_CONFIGS, get_model_config

# Use enhanced model registry
from configs.model_registry import ModelRegistry
registry = ModelRegistry()

# Apply evaluation presets
from configs.preset_configs import PRESET_CONFIGS

# Biomedical specific mappings
from configs.biomedical_model_dataset_mappings import BIOMEDICAL_MAPPINGS
```

## Configuration Standards

- **Model Configs**: Use `ModelConfig` dataclass with standardized fields
- **Naming Convention**: `snake_case` for model names, `UPPER_CASE` for constants
- **Documentation**: Each config file includes usage examples and field descriptions
- **Validation**: All configurations include validation utilities

## Archive Information

Historical configuration files are archived in:
- `archive/old_modules/model_configs_original.py` - Original model configuration backup