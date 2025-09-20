# Dataset Migration Analysis Report

**Generated**: September 20, 2025  
**Analysis Status**: Complete  

---

## 📊 **DISCOVERY RESULTS**

### **Current Dataset Distribution**

#### **datasets/ folder (Legacy)**
- **Total Files**: 6 JSON files
- **Location**: `datasets/biomedical/`
- **Datasets Found**:
  1. `bc5cdr/bc5cdr_sample_10.json` (2.5KB - sample)
  2. `ddi/ddi_sample_20.json` (5.5KB - sample)  
  3. `medqa/medqa_sample_100.json` (98KB - sample)
  4. `medqa/medqa_train.json` (10.1MB - full dataset)
  5. `pubmedqa/pubmedqa_sample_100.json` (211KB - sample)
  6. `pubmedqa/pubmedqa_train.json` (2.1MB - full dataset)

#### **evaluation_data/ folder (Active)**
- **Total Files**: 55 JSON files across all categories
- **Biomedical Files**: 3 JSON files
- **Location**: `evaluation_data/biomedical/`
- **Datasets Found**:
  1. `bioasq.json` (1.1MB)
  2. `mediqa.json` (361KB)
  3. `pubmedqa.json` (1.8MB)

### **Code Dependencies Analysis**

#### **Critical Dependencies Found**
- **File**: `configs/biomedical_model_dataset_mappings.py`
- **References**: 5 hardcoded paths to `datasets/biomedical/`
- **Impact**: Biomedical model configurations will break without migration

#### **URLs References (No Impact)**
- **File**: `scripts/dataset_downloader.py` 
- **Type**: HuggingFace dataset URLs (not file paths)
- **Impact**: None - these are remote URLs, not local paths

---

## 🔄 **MIGRATION STRATEGY**

### **Duplicate Analysis**

#### **pubmedqa Conflict**
- **Legacy**: `datasets/biomedical/pubmedqa/pubmedqa_train.json` (2.1MB)
- **Active**: `evaluation_data/biomedical/pubmedqa.json` (1.8MB)
- **Resolution**: Keep both, rename legacy to avoid conflict
- **Action**: Move as `pubmedqa_full.json` (larger/complete version)

#### **New Datasets to Migrate**
1. **medqa** - Not in evaluation_data, full migration needed
2. **bc5cdr** - Not in evaluation_data, full migration needed  
3. **ddi** - Not in evaluation_data, full migration needed

### **Migration Actions Required**

#### **Dataset Moves**
```bash
# Create new folders in evaluation_data
mkdir -p evaluation_data/biomedical/medqa
mkdir -p evaluation_data/biomedical/bc5cdr  
mkdir -p evaluation_data/biomedical/ddi

# Move unique datasets
cp datasets/biomedical/medqa/* evaluation_data/biomedical/medqa/
cp datasets/biomedical/bc5cdr/* evaluation_data/biomedical/bc5cdr/
cp datasets/biomedical/ddi/* evaluation_data/biomedical/ddi/

# Handle pubmedqa conflict - move as additional version
cp datasets/biomedical/pubmedqa/pubmedqa_train.json evaluation_data/biomedical/pubmedqa_full.json
cp datasets/biomedical/pubmedqa/pubmedqa_sample_100.json evaluation_data/biomedical/pubmedqa_sample.json
```

#### **Registry Updates Needed**
- Add entries for: `medqa`, `bc5cdr`, `ddi`
- Update paths in `biomedical_model_dataset_mappings.py`
- Ensure dataset path manager can find all datasets

#### **Configuration Updates**
- Update 5 path references in `biomedical_model_dataset_mappings.py`
- Change from `datasets/biomedical/` to `biomedical/` (relative paths)

---

## ✅ **VALIDATION PLAN**

### **Pre-Migration Baseline**
```bash
python show_datasets.py | grep biomedical
python -c "from configs.biomedical_model_dataset_mappings import BIOMEDICAL_DATASETS; print(len(BIOMEDICAL_DATASETS))"
```

### **Post-Migration Validation**
```bash
python show_datasets.py | grep biomedical  
python -c "from evaluation.dataset_manager import EnhancedDatasetManager; dm = EnhancedDatasetManager(); print('Biomedical datasets:', len([d for d in dm.get_available_datasets() if 'biomedical' in dm.get_dataset_info(d).task_type]))"
```

---

## 🎯 **EXPECTED OUTCOMES**

### **Benefits**
- **+3 New Datasets**: medqa, bc5cdr, ddi available in pipeline
- **+2 Dataset Variants**: pubmedqa_full, pubmedqa_sample  
- **Clean Architecture**: Single dataset location
- **Full Integration**: All biomedical datasets in registry

### **Metrics**
- **Before**: 3 biomedical datasets in pipeline
- **After**: 6+ biomedical datasets in pipeline  
- **Files Migrated**: 6 files (~12.5MB data)
- **Duplicates Resolved**: 1 (pubmedqa)

---

**Ready for Implementation!**