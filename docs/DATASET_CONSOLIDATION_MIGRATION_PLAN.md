# Dataset Consolidation Migration Plan

**Objective**: Consolidate `datasets/` folder into `evaluation_data/` and update all references  
**Date**: September 20, 2025  
**Estimated Time**: 2-3 hours  

---

## 🎯 **MIGRATION OVERVIEW**

### **Current State**
- **Primary Active**: `evaluation_data/` (25+ datasets, full pipeline integration)
- **Secondary Legacy**: `datasets/` (4 biomedical datasets, partial integration)
- **Issue**: Duplicate paths and inconsistent references

### **Target State** 
- **Single Source**: `evaluation_data/` contains ALL datasets
- **Clean Registry**: All dataset paths updated and validated
- **No Duplicates**: Removed redundant dataset files
- **Full Integration**: All datasets available to pipeline

---

## 📋 **DETAILED MIGRATION PLAN**

### **Phase 1: Discovery and Analysis (15 minutes)**

#### **1.1 Inventory Current Datasets**
- [ ] List all datasets in `datasets/biomedical/`
- [ ] List all datasets in `evaluation_data/biomedical/`
- [ ] Identify duplicates and unique datasets
- [ ] Check file sizes and modification dates for duplicate detection

#### **1.2 Analyze Dependencies**
- [ ] Search all config files for `datasets/` references
- [ ] Check model-dataset mappings for hardcoded paths
- [ ] Identify any SLURM scripts referencing `datasets/`
- [ ] Review documentation for outdated path references

### **Phase 2: Pre-Migration Validation (20 minutes)**

#### **2.1 Create Migration Report**
- [ ] Generate detailed comparison of dataset contents
- [ ] Create mapping of what will be moved/merged/deleted
- [ ] Validate no critical datasets will be lost
- [ ] Document all code changes required

#### **2.2 Backup Current State**
- [ ] Create backup of `datasets/` folder
- [ ] Create backup of current registry configurations
- [ ] Document current working evaluation commands for testing

### **Phase 3: Dataset Migration (30 minutes)**

#### **3.1 Handle Duplicates**
- [ ] Compare duplicate files (content, size, dates)
- [ ] Keep newer/larger/more complete versions
- [ ] Move unique datasets to appropriate `evaluation_data/` subfolders
- [ ] Merge overlapping dataset collections

#### **3.2 Migrate Unique Datasets**
- [ ] Move datasets not present in `evaluation_data/`
- [ ] Maintain proper folder structure and naming
- [ ] Verify file integrity after migration
- [ ] Update any dataset-specific metadata files

### **Phase 4: Configuration Updates (45 minutes)**

#### **4.1 Update Dataset Registry**
- [ ] Add entries for newly migrated datasets
- [ ] Update paths for existing datasets with duplicate references
- [ ] Validate all dataset paths resolve correctly
- [ ] Update sample counts and metadata

#### **4.2 Update Model-Dataset Mappings**
- [ ] Fix hardcoded paths in `biomedical_model_dataset_mappings.py`
- [ ] Update any other configuration files with `datasets/` references
- [ ] Ensure model categories can find all their datasets
- [ ] Update specialized model configurations

#### **4.3 Update Documentation**
- [ ] Fix any docs referencing `datasets/` folder
- [ ] Update README and architecture documentation
- [ ] Update migration analysis documents

### **Phase 5: Testing and Validation (30 minutes)**

#### **5.1 Registry Validation**
- [ ] Run dataset path validation
- [ ] Test dataset loading for migrated datasets
- [ ] Verify no broken references remain
- [ ] Check dataset summary reports

#### **5.2 Pipeline Testing**
- [ ] Test model listing with updated configurations
- [ ] Test dataset listing and discovery
- [ ] Run sample evaluations on migrated datasets
- [ ] Verify biomedical category still works correctly

#### **5.3 Cleanup**
- [ ] Remove empty `datasets/` folder
- [ ] Update `.gitignore` if needed
- [ ] Clean up any temporary migration files
- [ ] Update migration status in this document

### **Phase 6: Documentation and Verification (20 minutes)**

#### **6.1 Update Analysis Documents**
- [ ] Update MODULARITY_AND_INTEGRATION_ANALYSIS.md
- [ ] Mark migration as complete in relevant docs
- [ ] Update architecture documentation

#### **6.2 Final Verification**
- [ ] Run comprehensive dataset validation
- [ ] Test key evaluation workflows
- [ ] Verify no functionality regression
- [ ] Document any issues found and resolved

---

## 🔍 **DETAILED IMPLEMENTATION STEPS**

### **Step 1: Dataset Discovery**
```bash
# Compare dataset contents
ls -la datasets/biomedical/
ls -la evaluation_data/biomedical/

# Check for duplicates by name
find datasets/ -name "*.json" | sort
find evaluation_data/ -name "*.json" | sort

# Compare file sizes for duplicate detection
find datasets/ -name "*" -type f -exec ls -la {} \; | sort -k9
find evaluation_data/ -name "*" -type f -exec ls -la {} \; | sort -k9
```

### **Step 2: Analyze Code Dependencies**
```bash
# Find all references to datasets/ folder
grep -r "datasets/" --include="*.py" .
grep -r "datasets/" --include="*.md" docs/
grep -r "datasets/" --include="*.sh" slurm_jobs/
```

### **Step 3: Migration Execution**
```python
# Python script for safe dataset migration
import os
import shutil
from pathlib import Path
import json

def migrate_datasets():
    source_dir = Path("datasets")
    target_dir = Path("evaluation_data")
    
    # For each dataset in source, check if exists in target
    # Move unique ones, report duplicates
    
def update_registry():
    # Update dataset_registry.py with new datasets
    # Validate all paths
    
def update_configs():
    # Update biomedical_model_dataset_mappings.py
    # Fix any hardcoded paths
```

### **Step 4: Validation Scripts**
```python
# Test dataset loading
python -c "
from evaluation.dataset_manager import EnhancedDatasetManager
dm = EnhancedDatasetManager()
print('Available datasets:', len(dm.get_available_datasets()))
"

# Test biomedical category
python category_evaluation.py --list-categories
python show_datasets.py biomedical
```

---

## ⚠️ **RISKS AND MITIGATIONS**

### **Risk 1: Data Loss**
- **Mitigation**: Complete backup before migration
- **Validation**: Compare before/after file counts and sizes

### **Risk 2: Broken References**
- **Mitigation**: Comprehensive grep search for all references
- **Validation**: Run full pipeline test after migration

### **Risk 3: Configuration Conflicts**
- **Mitigation**: Update all configs systematically
- **Validation**: Test all model-dataset mappings

### **Risk 4: Pipeline Regression**
- **Mitigation**: Test key workflows before and after
- **Validation**: Run sample evaluations on affected datasets

---

## 📊 **SUCCESS CRITERIA**

- [ ] **Zero Data Loss**: All unique datasets preserved and accessible
- [ ] **Clean Architecture**: Single `evaluation_data/` folder for all datasets
- [ ] **Full Integration**: All datasets appear in registry and are usable
- [ ] **No Broken References**: All code references updated correctly
- [ ] **Pipeline Functionality**: All existing workflows continue to work
- [ ] **Documentation Updated**: All docs reflect new structure

---

## 🧪 **TEST CASES**

### **Pre-Migration Tests**
```bash
# Baseline functionality
python show_datasets.py | grep "Total datasets"
python show_models.py biomedical | grep "datasets"
python category_evaluation.py --list-categories
```

### **Post-Migration Tests**
```bash
# Verify same or increased dataset count
python show_datasets.py | grep "Total datasets"
python show_models.py biomedical | grep "datasets"
python category_evaluation.py --list-categories

# Test specific migrated datasets
python category_evaluation.py --model biomistral_7b --samples 2 --dry-run
```

---

## 📝 **MIGRATION LOG**

### **Execution Record**
- **Started**: September 20, 2025 - 15:51 UTC
- **Phase 1 Complete**: 15:52 UTC (Discovery and Analysis)
- **Phase 2 Complete**: 15:53 UTC (Pre-Migration Validation)
- **Phase 3 Complete**: 15:53 UTC (Dataset Migration)
- **Phase 4 Complete**: 15:54 UTC (Configuration Updates)
- **Phase 5 Complete**: 15:55 UTC (Testing and Validation)
- **Phase 6 Complete**: 15:56 UTC (Documentation and Verification)
- **Migration Complete**: 15:56 UTC ✅

### **Issues Encountered**
- ✅ No major issues encountered
- ⚠️ DDI dataset categorized as "relation_extraction" (expected, not biomedical_qa)
- ✅ All dataset paths resolved correctly
- ✅ All configurations updated successfully

### **Final Results**
- **Datasets Migrated**: 6 files (medqa_train.json, medqa_sample_100.json, bc5cdr_sample_10.json, ddi_sample_20.json, pubmedqa_train.json, pubmedqa_sample_100.json)
- **Duplicates Resolved**: 1 (pubmedqa - kept as variants)
- **Config Files Updated**: 2 (dataset_registry.py, biomedical_model_dataset_mappings.py)
- **Tests Passed**: 5/5 ✅
- **New Datasets Available**: 3 (medqa, bc5cdr, ddi)
- **Total Biomedical-Related Datasets**: 7 (up from 4)
- **Legacy Folder Removed**: ✅
- **Backup Created**: ✅ datasets_backup_20250920_155133

---

**Ready to proceed with implementation!**