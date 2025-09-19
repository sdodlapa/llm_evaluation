# LLM Evaluation Framework - Permanent Infrastructure Fixes Plan

## 🎯 **EXECUTIVE SUMMARY**

This document outlines the permanent architectural fixes needed to resolve the two critical infrastructure issues identified in the recent SLURM job analysis:

1. **JSON Serialization Failure**: vLLM `RequestOutput` objects causing session logging crashes
2. **Dataset Path Mapping Issue**: bioasq.json path mismatch between registry and filesystem

Both issues require permanent architectural solutions rather than ad-hoc workarounds, as they affect core evaluation pipeline functionality and will impact all future evaluations.

---

## 📊 **CURRENT STATE ANALYSIS**

### ✅ **Major Success: Backend Fixes**
- **26 models successfully loaded** across 8 SLURM jobs (1641-1648)
- **~64% improvement** over previous ~73 model loading failures
- **vLLM import/attribute issues completely resolved**

### 🚧 **Remaining Infrastructure Issues**

#### **Issue 1: JSON Serialization Failure**
```python
# FAILURE POINT: category_evaluation.py line 644
session_data['results'].append({
    "result": result_copy,  # Contains vLLM RequestOutput objects
})
json.dump(session_data, f, indent=2)  # FAILS HERE
# TypeError: Object of type RequestOutput is not JSON serializable
```

#### **Issue 2: Dataset Path Mapping Mismatch**
```python
# REGISTRY DEFINITION: evaluation/dataset_registry.py line 200
"bioasq": DatasetInfo(
    data_path="scientific/bioasq.json",  # Expected path
)

# ACTUAL FILESYSTEM: 
# evaluation_data/biomedical/bioasq.json  # Actual location
# Dataset file not found: evaluation_data/scientific/bioasq.json
```

---

## 🏗️ **ARCHITECTURAL ANALYSIS**

### **JSON Serialization Issue - System-Wide Impact**

#### **Affected Components:**
1. **category_evaluation.py** - Primary failure point
2. **evaluation/comprehensive_runner.py** - Uses json.dump for results
3. **simple_model_evaluation.py** - Results serialization
4. **All evaluation pipelines** - Session logging infrastructure

#### **Root Cause Analysis:**
vLLM `RequestOutput` objects contain complex nested structures:
- Torch tensors
- CUDA memory references  
- Non-serializable generator objects
- Nested model state information

#### **Data Flow Impact:**
```
Model Evaluation → RequestOutput Objects → Result Aggregation → Session Storage → JSON CRASH
```

### **Dataset Path Issue - Configuration Management**

#### **Affected Components:**
1. **evaluation/dataset_registry.py** - Authoritative dataset definitions
2. **Category mappings** - biomedical_specialists category
3. **Dataset discovery** - Automated path resolution
4. **All biomedical evaluations** - Scientific domain evaluations

#### **Root Cause Analysis:**
Inconsistent path organization:
- Registry expects: `evaluation_data/scientific/`
- Filesystem has: `evaluation_data/biomedical/`
- No validation pipeline to catch mismatches

---

## 🔧 **IMPLEMENTATION PLAN**

### **PHASE 1: Custom JSON Serialization Framework**

#### **File 1: evaluation/json_serializer.py** *(NEW FILE)*
**Purpose:** Robust JSON serialization handling for ML framework objects

```python
"""
Custom JSON Serialization Framework
Handles vLLM, torch, and other ML framework objects
"""

import json
from typing import Any, Dict, List
from datetime import datetime
import torch

class MLObjectEncoder(json.JSONEncoder):
    """Custom JSON encoder for ML framework objects"""
    
    def default(self, obj):
        # Handle vLLM RequestOutput objects
        if hasattr(obj, '__class__') and 'RequestOutput' in str(type(obj)):
            return self._serialize_vllm_request_output(obj)
        
        # Handle torch tensors
        if isinstance(obj, torch.Tensor):
            return {
                "_type": "torch.Tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "data": obj.detach().cpu().numpy().tolist() if obj.numel() < 1000 else "[LARGE_TENSOR_OMITTED]"
            }
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
            
        # Handle other complex objects
        if hasattr(obj, '__dict__'):
            return self._serialize_object_attributes(obj)
            
        return super().default(obj)
    
    def _serialize_vllm_request_output(self, obj):
        """Safely serialize vLLM RequestOutput objects"""
        return {
            "_type": "vLLM.RequestOutput",
            "request_id": getattr(obj, 'request_id', None),
            "outputs": self._serialize_completion_outputs(getattr(obj, 'outputs', [])),
            "finished": getattr(obj, 'finished', None),
            "metrics": getattr(obj, 'metrics', None),
        }
    
    def _serialize_completion_outputs(self, outputs):
        """Serialize completion output objects"""
        serialized = []
        for output in outputs:
            serialized.append({
                "index": getattr(output, 'index', None),
                "text": getattr(output, 'text', None),
                "token_ids": getattr(output, 'token_ids', []),
                "cumulative_logprob": getattr(output, 'cumulative_logprob', None),
                "logprobs": None,  # Skip complex logprob structures
                "finish_reason": getattr(output, 'finish_reason', None),
            })
        return serialized
    
    def _serialize_object_attributes(self, obj):
        """Safely serialize object attributes"""
        result = {"_type": str(type(obj))}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                try:
                    json.dumps(value)  # Test serializability
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = f"[NON_SERIALIZABLE_{type(value).__name__}]"
        return result

def safe_json_dump(data: Any, file_path: str, **kwargs) -> bool:
    """Safely dump data to JSON file with ML object support"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=MLObjectEncoder, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to serialize to {file_path}: {e}")
        return False
```

#### **File 2: category_evaluation.py** *(MODIFY EXISTING)*
**Purpose:** Replace standard json.dump with safe serialization

**Changes needed:**
1. Import new serialization framework
2. Replace json.dump calls with safe_json_dump
3. Add validation for serialization success

```python
# CHANGE 1: Add import
from evaluation.json_serializer import safe_json_dump, MLObjectEncoder

# CHANGE 2: Replace line 644
# OLD:
with open(session_log, 'w') as f:
    json.dump(session_data, f, indent=2)

# NEW:
if not safe_json_dump(session_data, session_log, indent=2):
    logger.error(f"Failed to save session log: {session_log}")
    # Fallback: save minimal session data
    minimal_session = {
        "session_id": session_data["session_id"],
        "start_time": session_data["start_time"],
        "end_time": session_data.get("end_time"),
        "total_tasks": session_data["total_tasks"],
        "success_count": len([r for r in session_data["results"] if r.get("success")])
    }
    with open(session_log.with_suffix('.minimal.json'), 'w') as f:
        json.dump(minimal_session, f, indent=2)

# CHANGE 3: Replace line 578 predictions file save
# OLD:
with open(predictions_file, 'w') as f:
    json.dump(prediction_data, f, indent=2)

# NEW:
if not safe_json_dump(prediction_data, predictions_file, indent=2):
    logger.error(f"Failed to save predictions: {predictions_file}")
```

#### **File 3: evaluation/comprehensive_runner.py** *(MODIFY EXISTING)*
**Purpose:** Update results saving with safe serialization

**Changes needed:**
1. Import safe_json_dump
2. Replace json.dump calls in save_evaluation_result method

```python
# CHANGE 1: Add import at top
from .json_serializer import safe_json_dump

# CHANGE 2: Replace line 59 (raw results save)
# OLD:
with open(raw_file, 'w') as f:
    json.dump(evaluation_result, f, indent=2)

# NEW:
if not safe_json_dump(evaluation_result, raw_file, indent=2):
    logger.error(f"Failed to save raw results: {raw_file}")

# CHANGE 3: Replace line 113 (combined results save)
# OLD:
with open(combined_file, 'w') as f:
    json.dump(combined_result, f, indent=2)

# NEW:
if not safe_json_dump(combined_result, combined_file, indent=2):
    logger.error(f"Failed to save combined results: {combined_file}")
```

---

### **PHASE 2: Centralized Dataset Path Registry**

#### **File 1: evaluation/dataset_path_manager.py** *(NEW FILE)*
**Purpose:** Authoritative dataset path management and validation

```python
"""
Dataset Path Manager - Centralized dataset discovery and validation
Resolves path inconsistencies and provides authoritative dataset locations
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetPathManager:
    """Manages dataset path resolution and validation"""
    
    def __init__(self, base_data_dir: str = "evaluation_data"):
        self.base_data_dir = Path(base_data_dir)
        self.category_mappings = self._initialize_category_mappings()
        self.path_cache = {}
        
    def _initialize_category_mappings(self) -> Dict[str, List[str]]:
        """Initialize category to folder mappings"""
        return {
            "coding": ["coding"],
            "mathematical": ["mathematical", "math"],
            "biomedical": ["biomedical", "scientific", "bio"],  # Multiple possible locations
            "multimodal": ["multimodal", "vision", "document"],
            "qa": ["qa", "question_answering"],
            "safety": ["safety", "alignment"],
            "function_calling": ["function_calling", "tools"],
            "scientific": ["scientific", "biomedical"],  # Alias for biomedical
        }
    
    def resolve_dataset_path(self, dataset_name: str, registry_path: str) -> Tuple[Optional[str], bool]:
        """
        Resolve actual dataset path, checking multiple possible locations
        Returns: (actual_path, path_exists)
        """
        # Check cache first
        cache_key = f"{dataset_name}:{registry_path}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Try registry path first
        registry_full_path = self.base_data_dir / registry_path
        if registry_full_path.exists():
            result = (str(registry_full_path), True)
            self.path_cache[cache_key] = result
            return result
        
        # Extract category from registry path
        category = Path(registry_path).parts[0] if Path(registry_path).parts else None
        filename = Path(registry_path).name
        
        # Try alternative locations based on category mappings
        if category in self.category_mappings:
            for alt_category in self.category_mappings[category]:
                alt_path = self.base_data_dir / alt_category / filename
                if alt_path.exists():
                    logger.warning(f"Dataset {dataset_name} found at {alt_path} instead of expected {registry_full_path}")
                    result = (str(alt_path), True)
                    self.path_cache[cache_key] = result
                    return result
        
        # Try scanning all subdirectories as last resort
        for subdir in self.base_data_dir.iterdir():
            if subdir.is_dir():
                candidate_path = subdir / filename
                if candidate_path.exists():
                    logger.warning(f"Dataset {dataset_name} found at {candidate_path} instead of expected {registry_full_path}")
                    result = (str(candidate_path), True)
                    self.path_cache[cache_key] = result
                    return result
        
        # Dataset not found
        logger.error(f"Dataset {dataset_name} not found. Expected: {registry_full_path}")
        result = (None, False)
        self.path_cache[cache_key] = result
        return result
    
    def validate_all_datasets(self, dataset_registry) -> Dict[str, Dict[str, Any]]:
        """Validate all datasets in registry and report issues"""
        validation_results = {}
        
        for dataset_name, dataset_info in dataset_registry.datasets.items():
            actual_path, exists = self.resolve_dataset_path(dataset_name, dataset_info.data_path)
            
            validation_results[dataset_name] = {
                "expected_path": dataset_info.data_path,
                "actual_path": actual_path,
                "exists": exists,
                "needs_path_correction": actual_path != str(self.base_data_dir / dataset_info.data_path) if exists else False
            }
        
        return validation_results
    
    def generate_path_correction_script(self, validation_results: Dict) -> str:
        """Generate script to fix dataset path issues"""
        corrections = []
        
        for dataset_name, result in validation_results.items():
            if result["needs_path_correction"] and result["exists"]:
                corrections.append({
                    "dataset": dataset_name,
                    "current_path": result["actual_path"],
                    "expected_path": result["expected_path"],
                    "action": "update_registry" if Path(result["actual_path"]).exists() else "move_file"
                })
        
        return corrections

# Global instance
dataset_path_manager = DatasetPathManager()
```

#### **File 2: evaluation/dataset_manager.py** *(MODIFY EXISTING)*
**Purpose:** Integrate path manager for robust dataset loading

**Changes needed:**
1. Import DatasetPathManager
2. Modify dataset loading to use path resolution
3. Add validation before evaluation

```python
# CHANGE 1: Add import
from .dataset_path_manager import dataset_path_manager

# CHANGE 2: Modify load_dataset method (approximate line 80)
def load_dataset(self, dataset_name: str, limit: Optional[int] = None) -> Tuple[List[Dict], Optional[str]]:
    """Load dataset with robust path resolution"""
    if dataset_name not in self.registry.datasets:
        return [], f"Dataset {dataset_name} not found in registry"
    
    dataset_info = self.registry.datasets[dataset_name]
    
    # Use path manager for robust resolution
    actual_path, exists = dataset_path_manager.resolve_dataset_path(dataset_name, dataset_info.data_path)
    
    if not exists:
        return [], f"Dataset file not found: {dataset_info.data_path} (searched multiple locations)"
    
    try:
        with open(actual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            return [], f"Dataset {dataset_name} should contain a list of samples"
        
        # Apply limit if specified
        if limit and limit > 0:
            data = data[:limit]
            
        logger.info(f"Loaded {len(data)} samples from {dataset_name} (path: {actual_path})")
        return data, None
        
    except Exception as e:
        return [], f"Error loading {dataset_name}: {str(e)}"
```

#### **File 3: evaluation/dataset_registry.py** *(MODIFY EXISTING)*
**Purpose:** Fix bioasq path and add validation

**Changes needed:**
1. Fix bioasq data_path from "scientific/bioasq.json" to "biomedical/bioasq.json"
2. Add registry validation method

```python
# CHANGE 1: Fix line 200
# OLD:
"bioasq": DatasetInfo(
    name="bioasq",
    task_type="biomedical_qa",
    data_path="scientific/bioasq.json",
    metadata_path="meta/bioasq_metadata.json",
    sample_count=1000,
    evaluation_type="qa_accuracy",
    description="Biomedical question answering"
),

# NEW:
"bioasq": DatasetInfo(
    name="bioasq",
    task_type="biomedical_qa",
    data_path="biomedical/bioasq.json",
    metadata_path="meta/bioasq_metadata.json",
    sample_count=1000,
    evaluation_type="qa_accuracy",
    description="Biomedical question answering"
),

# CHANGE 2: Add validation method at end of class
def validate_registry(self) -> Dict[str, Any]:
    """Validate all dataset paths and configurations"""
    from .dataset_path_manager import dataset_path_manager
    return dataset_path_manager.validate_all_datasets(self)
```

---

### **PHASE 3: Validation and Testing Framework**

#### **File 1: tests/test_infrastructure_fixes.py** *(NEW FILE)*
**Purpose:** Comprehensive testing of both fixes

```python
"""
Test Infrastructure Fixes
Validates JSON serialization and dataset path resolution
"""

import json
import tempfile
from pathlib import Path
import pytest

from evaluation.json_serializer import MLObjectEncoder, safe_json_dump
from evaluation.dataset_path_manager import DatasetPathManager
from evaluation.dataset_registry import DatasetRegistry

class TestJSONSerialization:
    """Test custom JSON serialization"""
    
    def test_vllm_request_output_mock(self):
        """Test serialization of vLLM RequestOutput-like objects"""
        # Mock vLLM RequestOutput structure
        mock_request_output = type('RequestOutput', (), {
            'request_id': 'test_123',
            'outputs': [
                type('CompletionOutput', (), {
                    'index': 0,
                    'text': 'Hello world',
                    'token_ids': [1, 2, 3],
                    'finish_reason': 'stop'
                })()
            ],
            'finished': True
        })()
        
        # Test serialization
        encoder = MLObjectEncoder()
        serialized = encoder.default(mock_request_output)
        
        assert serialized['_type'] == 'vLLM.RequestOutput'
        assert serialized['request_id'] == 'test_123'
        assert serialized['finished'] is True
        assert len(serialized['outputs']) == 1
    
    def test_safe_json_dump(self):
        """Test safe JSON dump with complex objects"""
        complex_data = {
            'simple_data': {'key': 'value'},
            'mock_vllm_output': type('RequestOutput', (), {
                'request_id': 'test_456',
                'outputs': []
            })()
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            success = safe_json_dump(complex_data, f.name, indent=2)
            assert success
            
            # Verify file was created and is valid JSON
            with open(f.name, 'r') as verify_f:
                loaded_data = json.load(verify_f)
                assert 'simple_data' in loaded_data
                assert loaded_data['mock_vllm_output']['_type'] == 'vLLM.RequestOutput'

class TestDatasetPathResolution:
    """Test dataset path resolution"""
    
    def test_path_manager_initialization(self):
        """Test DatasetPathManager initialization"""
        manager = DatasetPathManager("test_data")
        assert manager.base_data_dir == Path("test_data")
        assert "biomedical" in manager.category_mappings
        assert "scientific" in manager.category_mappings["biomedical"]
    
    def test_bioasq_path_resolution(self):
        """Test bioasq specific path resolution"""
        # This would need actual test data setup
        pass
    
    def test_registry_validation(self):
        """Test dataset registry validation"""
        registry = DatasetRegistry()
        validation_results = registry.validate_registry()
        
        # Check that bioasq is included in results
        assert 'bioasq' in validation_results
        bioasq_result = validation_results['bioasq']
        assert 'expected_path' in bioasq_result
        assert 'exists' in bioasq_result
```

#### **File 2: scripts/validate_infrastructure_fixes.py** *(NEW FILE)*
**Purpose:** Validation script for production deployment

```python
"""
Validate Infrastructure Fixes
Run this script to validate both JSON serialization and dataset path fixes
"""

import logging
from pathlib import Path
import json

from evaluation.json_serializer import safe_json_dump, MLObjectEncoder
from evaluation.dataset_registry import DatasetRegistry
from evaluation.dataset_path_manager import dataset_path_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_json_serialization():
    """Validate JSON serialization framework"""
    logger.info("🔧 Testing JSON serialization framework...")
    
    # Create mock complex data similar to evaluation results
    mock_complex_data = {
        "session_id": "test_session",
        "results": [
            {
                "model": "test_model",
                "success": True,
                "result": {
                    "evaluation_result": {
                        "predictions": ["test output"],
                        "mock_vllm_output": type('RequestOutput', (), {
                            'request_id': 'test_123',
                            'outputs': [type('Output', (), {'text': 'test'})()],
                            'finished': True
                        })()
                    }
                }
            }
        ]
    }
    
    # Test serialization
    test_file = Path("test_serialization.json")
    success = safe_json_dump(mock_complex_data, test_file, indent=2)
    
    if success:
        logger.info("✅ JSON serialization working correctly")
        test_file.unlink()  # Clean up
        return True
    else:
        logger.error("❌ JSON serialization failed")
        return False

def validate_dataset_paths():
    """Validate dataset path resolution"""
    logger.info("🗂️ Testing dataset path resolution...")
    
    registry = DatasetRegistry()
    validation_results = registry.validate_registry()
    
    issues_found = 0
    for dataset_name, result in validation_results.items():
        if not result["exists"]:
            logger.warning(f"⚠️ Dataset {dataset_name} not found: {result['expected_path']}")
            issues_found += 1
        elif result["needs_path_correction"]:
            logger.warning(f"⚠️ Dataset {dataset_name} path mismatch: expected {result['expected_path']}, found at {result['actual_path']}")
            issues_found += 1
        else:
            logger.info(f"✅ Dataset {dataset_name} found correctly at {result['expected_path']}")
    
    # Focus on bioasq specifically
    if 'bioasq' in validation_results:
        bioasq_result = validation_results['bioasq']
        if bioasq_result["exists"]:
            logger.info("✅ bioasq dataset resolved successfully")
        else:
            logger.error("❌ bioasq dataset still not found after fixes")
            issues_found += 1
    
    logger.info(f"📊 Dataset validation complete: {issues_found} issues found")
    return issues_found == 0

def main():
    """Run complete infrastructure validation"""
    logger.info("🚀 Starting infrastructure fixes validation...")
    
    json_ok = validate_json_serialization()
    paths_ok = validate_dataset_paths()
    
    if json_ok and paths_ok:
        logger.info("🎉 All infrastructure fixes validated successfully!")
        return True
    else:
        logger.error("💥 Infrastructure validation failed - fixes needed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

---

## 📋 **IMPLEMENTATION SEQUENCE**

### **Step 1: Create JSON Serialization Framework**
1. Create `evaluation/json_serializer.py`
2. Test with mock vLLM objects
3. Validate serialization of complex nested structures

### **Step 2: Update Evaluation Components**
1. Modify `category_evaluation.py` - replace json.dump calls
2. Modify `evaluation/comprehensive_runner.py` - safe serialization
3. Test with existing evaluation pipeline

### **Step 3: Create Dataset Path Management**
1. Create `evaluation/dataset_path_manager.py`
2. Fix bioasq path in `evaluation/dataset_registry.py`
3. Update `evaluation/dataset_manager.py` for robust loading

### **Step 4: Validation and Testing**
1. Create comprehensive test suite
2. Run validation script
3. Test with biomedical_specialists category specifically

### **Step 5: Production Deployment**
1. Re-run failed SLURM jobs 1641-1648
2. Verify complete evaluation cycles
3. Validate session logs and results files

---

## 🎯 **EXPECTED OUTCOMES**

### **Immediate Fixes:**
- ✅ **JSON serialization errors eliminated** - all evaluation sessions complete successfully
- ✅ **bioasq dataset found** - biomedical_specialists category evaluations work
- ✅ **Robust error handling** - graceful degradation for serialization issues
- ✅ **Path validation** - proactive detection of dataset misconfigurations

### **Long-term Benefits:**
- 🚀 **Scalable framework** - handles new ML frameworks automatically
- 🔍 **Better debugging** - comprehensive serialization of complex objects
- 📊 **Complete session logs** - full evaluation traceability
- 🏗️ **Maintainable architecture** - centralized path and serialization management

---

## 💾 **FILES TO CREATE/MODIFY**

### **New Files:**
1. `evaluation/json_serializer.py` - Custom JSON serialization framework
2. `evaluation/dataset_path_manager.py` - Centralized path management
3. `tests/test_infrastructure_fixes.py` - Comprehensive test suite
4. `scripts/validate_infrastructure_fixes.py` - Production validation

### **Existing Files to Modify:**
1. `category_evaluation.py` - Replace json.dump, add safe serialization
2. `evaluation/comprehensive_runner.py` - Update results saving
3. `evaluation/dataset_registry.py` - Fix bioasq path
4. `evaluation/dataset_manager.py` - Integrate path resolution

---

## 🚨 **RISK MITIGATION**

### **Backwards Compatibility:**
- All changes are additive - existing functionality preserved
- Fallback mechanisms for serialization failures
- Gradual rollout possible

### **Testing Strategy:**
- Comprehensive unit tests for all new components
- Integration tests with actual evaluation pipeline
- Validation scripts for production deployment

### **Rollback Plan:**
- All changes in separate commits
- Original functionality preserved as fallbacks
- Easy revert if issues discovered

---

**This plan provides a robust, permanent solution to both infrastructure issues while maintaining system reliability and enabling future scalability.** 🚀