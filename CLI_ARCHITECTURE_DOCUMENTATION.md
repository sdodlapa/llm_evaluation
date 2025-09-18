# LLM Evaluation Framework - CLI Architecture Documentation

## Executive Summary

This document provides comprehensive documentation of the CLI architecture for the LLM Evaluation Framework. It maps all entry points, execution paths, data structure patterns, and identifies architectural inconsistencies to enable systematic development and troubleshooting.

**Key Findings:**
- **3 Primary CLI Entry Points** with distinct purposes and execution flows
- **Mixed Data Structure Pattern** causing inconsistencies between dataclass and dictionary access
- **Layered Architecture** with clear separation of concerns but some coupling issues
- **Category-Based Evaluation System** as the main abstraction for systematic testing

---

## Table of Contents

1. [CLI Entry Points](#cli-entry-points)
2. [Execution Flow Analysis](#execution-flow-analysis)
3. [Data Structure Patterns](#data-structure-patterns)
4. [Architectural Inconsistencies](#architectural-inconsistencies)
5. [Command Reference](#command-reference)
6. [Development Guidelines](#development-guidelines)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## CLI Entry Points

### 1. **category_evaluation.py** - Primary Category-Based Evaluation Interface

**Purpose:** Systematic evaluation of model categories against appropriate datasets
**Status:** Production-ready, actively used
**Architecture:** Category-focused with flexible model/dataset selection

**Key Command Patterns:**
```bash
# Category-based evaluation (primary use case)
python category_evaluation.py --category coding_specialists --samples 5 --preset balanced

# Model-specific evaluation on category datasets
python category_evaluation.py --model qwen3_8b --samples 10 --preset performance

# Specific model-dataset combination
python category_evaluation.py --model qwen3_8b --dataset humaneval --samples 5

# Information commands
python category_evaluation.py --list-categories
python category_evaluation.py --list-models
python category_evaluation.py --category-info coding_specialists
python category_evaluation.py --validate
```

**Execution Flow:**
1. `CategoryEvaluationCLI.__init__()` → Initialize CategoryMappingManager & EvaluationOrchestrator
2. `parse_args()` → Handle argument parsing with mutual exclusion groups
3. **Information Path:** `list_categories()`, `list_models()`, `show_category_info()`, `validate_categories()`
4. **Evaluation Path:** `generate_evaluation_tasks()` → `run_evaluations()`
5. **Data Flow:** Args → Tasks → Orchestrator → Engine → Results

### 2. **evaluation/cli_interface.py** - Legacy General Evaluation Interface

**Purpose:** General-purpose LLM evaluation with model/dataset flexibility
**Status:** Legacy system, less actively maintained
**Architecture:** Model-focused with preset comparison capabilities

**Key Command Patterns:**
```bash
# Single model evaluation
python -m evaluation.cli_interface --model qwen3_8b --preset balanced --dataset humaneval --samples 10

# Multi-model comparison
python -m evaluation.cli_interface --model qwen3_8b,qwen3_14b --dataset humaneval,mbpp

# Preset comparison
python -m evaluation.cli_interface --model qwen3_8b --compare-presets

# Quick test
python -m evaluation.cli_interface --model qwen3_8b --quick-test
```

### 3. **simple_model_evaluation.py** - Simplified Testing Interface

**Purpose:** Quick validation of models using predefined specialization mappings
**Status:** Development/testing utility
**Architecture:** Hardcoded model-dataset mappings for rapid testing

---

## Execution Flow Analysis

### Category Evaluation Execution Flow

```
CLI Arguments
     ↓
CategoryEvaluationCLI.parse_args()
     ↓
Information Commands ←→ Evaluation Commands
     ↓                        ↓
List/Validate/Info    generate_evaluation_tasks()
     ↓                        ↓
Direct Output        CategoryMappingManager.generate_evaluation_tasks()
                             ↓
                     EvaluationTask objects
                             ↓
                     run_evaluations()
                             ↓
                     EvaluationOrchestrator.run_comprehensive_evaluation()
                             ↓
                     EvaluationEngine.evaluate_model_on_dataset()
                             ↓
                     Results saved to category_evaluation_results/
```

### Data Flow Architecture

```
Model Configs (ModelConfig dataclass)
     ↓
Category Registry (ModelCategory dataclass)
     ↓
CategoryMappingManager (Task generation)
     ↓
EvaluationTask objects (Bridge layer)
     ↓
EvaluationOrchestrator (High-level coordination)
     ↓
EvaluationEngine (Core evaluation logic)
     ↓
Results (Dictionary structures)
```

---

## Data Structure Patterns

### **Pattern 1: Dataclass-Based Configuration** ✅ **RECOMMENDED**

**Usage:** Model configurations, category definitions, performance metrics
**Implementation:** `@dataclass` with typed fields and methods

```python
@dataclass
class ModelConfig:
    model_name: str
    huggingface_id: str
    license: str
    size_gb: float
    # ... additional fields with defaults
    
    def get_vllm_config(self) -> Dict[str, Any]:
        # Method-based access
```

**Benefits:**
- Type safety with IDE support
- Immutable with controlled access
- Built-in serialization support
- Self-documenting structure

### **Pattern 2: Dictionary-Based Results** ✅ **APPROPRIATE FOR RESULTS**

**Usage:** Evaluation results, dynamic configurations, API responses
**Implementation:** `Dict[str, Any]` with nested structures

```python
{
    "evaluation_metadata": {...},
    "results": {...},
    "summary": {...},
    "errors": []
}
```

**Benefits:**
- Flexible for varying result structures
- Easy JSON serialization
- Dynamic field addition
- Compatible with external APIs

### **Pattern 3: Mixed Access Pattern** ❌ **PROBLEMATIC**

**Current Issue:** Some code tries to access dataclass objects as dictionaries

```python
# WRONG - causes AttributeError
category = CATEGORY_REGISTRY["coding_specialists"]  # Returns ModelCategory dataclass
models = category['models']  # ❌ Tries dictionary access on dataclass

# CORRECT - proper dataclass access
category = CATEGORY_REGISTRY["coding_specialists"]  # Returns ModelCategory dataclass  
models = category.models  # ✅ Proper attribute access
```

---

## Architectural Inconsistencies

### **Issue 1: Mixed Data Structure Access Patterns**

**Problem:** Code alternates between dataclass and dictionary access for the same objects
**Root Cause:** Evolution from dictionary-based to dataclass-based category system
**Impact:** Runtime AttributeError exceptions

**Examples Found:**
```python
# In category_mappings.py (FIXED in recent updates)
missing_primary = [d for d in category['primary_datasets'] if d not in self.available_datasets]
# Should be:
missing_primary = [d for d in category.primary_datasets if d not in self.available_datasets]
```

**Resolution Strategy:**
1. Standardize on dataclass access for category objects
2. Use dictionaries only for dynamic/variable structures
3. Create clear type hints to prevent confusion

### **Issue 2: Inconsistent Import Patterns**

**Problem:** Mixed absolute and relative imports throughout codebase
**Impact:** Import errors when running modules in different contexts

**Examples:**
```python
# Some modules use relative imports
from .model_categories import ModelCategory

# Others use absolute imports  
from evaluation.mappings.model_categories import ModelCategory
```

**Resolution Strategy:**
1. Standardize on absolute imports for CLI entry points
2. Use relative imports only within package modules
3. Add proper `__init__.py` files for all packages

### **Issue 3: Circular Dependencies**

**Problem:** Some modules have circular import dependencies
**Impact:** Import errors and initialization issues

**Resolution Strategy:**
1. Move shared types to separate `types.py` module
2. Use dependency injection for complex dependencies
3. Lazy imports where appropriate

---

## Command Reference

### category_evaluation.py Commands

#### Information Commands
| Command | Purpose | Output |
|---------|---------|--------|
| `--list-categories` | Show all available categories with readiness status | Category list with model counts |
| `--list-models` | Show all models grouped by category | Model list by category |
| `--category-info <name>` | Detailed category information | Category details with datasets |
| `--validate` | Validate all category readiness | Category validation report |

#### Evaluation Commands
| Command Pattern | Purpose | Example |
|----------------|---------|---------|
| `--category <name>` | Evaluate category | `--category coding_specialists` |
| `--model <name>` | Evaluate specific model | `--model qwen3_8b` |
| `--dataset <name>` | Specific dataset (with model/category) | `--dataset humaneval` |

#### Modifiers
| Argument | Default | Purpose |
|----------|---------|---------|
| `--samples <n>` | 5 | Samples per dataset |
| `--preset <name>` | balanced | Model configuration preset |
| `--include-optional` | False | Include optional datasets |
| `--models <list>` | All | Filter models in category |
| `--exclude-models <list>` | None | Exclude specific models |
| `--dry-run` | False | Preview without execution |
| `--output-dir <path>` | category_evaluation_results | Output directory |

### Preset Options
- **balanced** (default): Optimized for reliability and performance balance
- **performance**: Maximum throughput, higher memory usage
- **memory_optimized**: Lower memory usage, slower performance

---

## Development Guidelines

### **1. Data Structure Guidelines**

**✅ DO:**
- Use `@dataclass` for fixed-structure configuration objects
- Use `Dict[str, Any]` for dynamic results and API responses
- Include type hints for all function parameters and returns
- Use attribute access (`.field`) for dataclass objects

**❌ DON'T:**
- Mix dictionary and dataclass access patterns
- Use untyped dictionaries for structured data
- Access dataclass fields using dictionary syntax

### **2. CLI Design Guidelines**

**✅ DO:**
- Use mutually exclusive argument groups for different modes
- Provide information commands that don't require execution
- Include dry-run options for complex operations
- Use clear, descriptive argument names

**❌ DON'T:**
- Make complex argument combinations that are hard to validate
- Use cryptic abbreviations for arguments
- Skip validation of argument combinations

### **3. Error Handling Guidelines**

**✅ DO:**
- Validate arguments early in the execution flow
- Provide clear, actionable error messages
- Continue on non-critical errors with logging
- Save partial results before failures

**❌ DON'T:**
- Let exceptions propagate without context
- Fail silently on validation errors
- Lose work due to late-stage failures

---

## Troubleshooting Guide

### **Common Error: AttributeError on category access**

**Error Message:**
```
AttributeError: 'dict' object has no attribute 'models'
AttributeError: 'ModelCategory' object has no attribute 'getitem'
```

**Cause:** Mixed access patterns between dictionary and dataclass
**Solution:** Check how category objects are being accessed

**Debugging Steps:**
1. Verify `CATEGORY_REGISTRY` returns `ModelCategory` dataclass instances
2. Replace dictionary access (`category['field']`) with attribute access (`category.field`)
3. Check imports to ensure correct module versions

### **Common Error: Import errors in different execution contexts**

**Error Message:**
```
ModuleNotFoundError: No module named 'evaluation.mappings'
```

**Cause:** Inconsistent import patterns
**Solution:** Use absolute imports from project root

**Debugging Steps:**
1. Check if running from project root directory
2. Verify `sys.path` includes project root
3. Use absolute imports for CLI scripts

### **Common Error: Category not found**

**Error Message:**
```
Category 'coding_specialists' not found
```

**Cause:** Category registry not properly initialized
**Solution:** Check category registration

**Debugging Steps:**
1. Verify `CATEGORY_REGISTRY` is populated
2. Check category name spelling (case-sensitive)
3. Ensure category module is imported correctly

---

## Architecture Decision Records

### **ADR-001: Category-Based Evaluation System**

**Status:** Adopted
**Decision:** Use category-based model organization instead of individual model management
**Rationale:** 
- Systematic evaluation of related models
- Clear specialization boundaries
- Scalable to new model types

### **ADR-002: Dataclass for Configuration, Dict for Results**

**Status:** Adopted
**Decision:** Use dataclasses for configuration objects, dictionaries for results
**Rationale:**
- Type safety for configurations
- Flexibility for dynamic results
- Clear separation of concerns

### **ADR-003: CLI-First Architecture**

**Status:** Adopted  
**Decision:** Design evaluation system with CLI as primary interface
**Rationale:**
- Scriptable for automation
- Clear interface contracts
- Easy debugging and testing

---

## Future Improvements

### **1. Type Safety Enhancements**
- Add comprehensive type hints throughout codebase
- Use `TypedDict` for structured result dictionaries
- Implement runtime type checking for critical paths

### **2. Configuration Management**
- Centralized configuration system
- Environment-based configuration overrides  
- Configuration validation at startup

### **3. Plugin Architecture**
- Pluggable dataset processors
- Configurable evaluation metrics
- Extensible model implementations

### **4. CLI Improvements**
- Interactive mode for complex evaluations
- Configuration file support
- Progress bars and real-time status updates

---

**Document Version:** 1.0  
**Last Updated:** September 18, 2025  
**Author:** LLM Evaluation Framework Team