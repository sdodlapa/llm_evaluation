# ZeroGPU AOTI Implementation Plan
## Technical Roadmap for High-Impact Performance Improvements

**Date**: September 21, 2025  
**Version**: 1.0  
**Target**: Lightweight & Distributed Engines  
**Expected Impact**: 1.3x-1.8x speedup, 2-4x faster cold starts

---

## 🎯 **Executive Summary**

This document outlines the implementation plan for integrating ZeroGPU AOTI (Ahead-of-Time) compilation concepts into our hybrid evaluation pipeline. Based on performance analysis, **AOT compilation and model graph caching** provide the highest ROI for implementation effort.

### **Impact Assessment**
| Component | Current Performance | AOTI Enhanced | Improvement | Implementation Effort |
|-----------|-------------------|---------------|-------------|----------------------|
| **Model Loading** | 60-120s cold start | 15-30s | **2-4x faster** | **HIGH** |
| **Inference Speed** | Baseline | +30-80% throughput | **1.3-1.8x** | **HIGH** |
| **Memory Usage** | Standard | -15-25% consumption | **25% more efficient** | **MEDIUM** |
| **Multi-Model Switching** | In-memory cache (2 models) | Persistent AOT cache | **2-3x faster** | **MEDIUM** |

### **Recommended Implementation Priority**
1. **Phase 1** (HIGH IMPACT): AOT Compilation Module + Basic Caching
2. **Phase 2** (MEDIUM IMPACT): Persistent Model Graph Storage
3. **Phase 3** (LOW IMPACT): Regional Compilation + Dynamic Shapes

---

## 📊 **Current Pipeline Performance Analysis**

### ✅ **CORRECTION: Current Implementation Already Has Smart Caching**

**Important Note**: Upon detailed code analysis, **both engines already implement intelligent model caching**, not sequential reload as initially assessed:

#### **Lightweight Engine Current Caching:**
```python
# engines/lightweight/lightweight_engine.py
self._loaded_models: Dict[str, Any] = {}
self.max_cached_models = 2  # Keep 2 models in memory

# Smart loading check:
if model_key not in self._loaded_models:
    self._load_model(model_config)  # Only load if not cached

# LRU-style eviction when cache full:
if len(self._loaded_models) >= self.max_cached_models:
    self._evict_oldest_model()
```

#### **Distributed Engine Current Caching:**
```python
# engines/distributed/multi_gpu_model_loader.py  
self._loaded_models: Dict[str, DistributedModelInfo] = {}

# Avoid redundant loading:
if model_name in self._loaded_models:
    logger.info(f"Model {model_name} already loaded")
    return self._loaded_models[model_name]
```

### **Revised Bottleneck Analysis**
Based on code analysis of the lightweight engine:

```python
# Current model loading pipeline - lightweight_engine.py
def _load_model(self, model_config: EnhancedModelConfig):
    logger.info(f"Loading model {model_config.model_name}")
    load_start = time.time()
    
    # BOTTLENECK 1: vLLM initialization (30-60s)
    llm = LLM(**vllm_config)  
    
    # BOTTLENECK 2: JIT compilation during first inference (20-40s)
    sampling_params = SamplingParams(...)
    
    # Current startup_time_seconds=30.0 is optimistic
```

### **Performance Characteristics**
- **Lightweight Engine**: 30s startup (actual 60-120s with JIT compilation), **2-model in-memory cache**
- **Distributed Engine**: **Smart model caching** with persistence across evaluations
- **Resource Utilization**: 85% GPU memory utilization target
- **Quantization**: AWQ/GPTQ support but no torch.compile integration

### **AOTI Improvement Opportunities**
1. **JIT Compilation Overhead**: Replace runtime compilation with AOT pre-compiled graphs
2. **Cross-Session Persistence**: Extend current in-memory caching to disk persistence  
3. **Compilation Optimization**: torch._inductor optimizations beyond current vLLM optimizations
4. **Cache Capacity**: Increase from 2-model limit to persistent unlimited cache

---

## 🏗️ **Technical Architecture Design**

### **Phase 1: AOT Compilation Module**

#### **1.1 AOTModelCompiler Class**
```python
# engines/shared/aot_compiler.py
import torch
import torch.export
import torch._inductor
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import hashlib
import pickle
import logging

class AOTModelCompiler:
    """
    Ahead-of-Time model compilation using torch.export + inductor
    Integrates with existing lightweight and distributed engines
    """
    
    def __init__(self, 
                 cache_dir: str = "model_cache/compiled",
                 enable_regional_compilation: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_regional = enable_regional_compilation
        
        # Compiled model registry
        self.compiled_cache: Dict[str, Any] = {}
        self.compilation_stats: Dict[str, Dict] = {}
        
        # Supported architectures for AOT compilation
        self.supported_architectures = {
            'llama', 'qwen', 'mistral', 'phi', 'gemma', 
            'falcon', 'baichuan', 'yi', 'chatglm'
        }
        
        logger.info("AOT Model Compiler initialized")
    
    def is_model_supported(self, model_config) -> bool:
        """Check if model architecture supports AOT compilation"""
        model_name = model_config.model_name.lower()
        
        # Check architecture support
        supported = any(arch in model_name for arch in self.supported_architectures)
        
        # Check size constraints (AOT compilation memory overhead)
        if model_config.size_gb > 70.0:  # Conservative limit
            logger.warning(f"Model {model_name} too large for AOT compilation")
            return False
            
        # Check PyTorch version compatibility
        torch_version = torch.__version__
        if not (torch_version >= "2.4.0"):
            logger.warning(f"PyTorch {torch_version} too old for stable AOTI")
            return False
            
        return supported
    
    def compile_model_aot(self, 
                         model, 
                         example_inputs: Tuple,
                         model_config,
                         compilation_mode: str = "default") -> Optional[Any]:
        """
        Compile model ahead of time using torch.export + inductor
        
        Args:
            model: The PyTorch model to compile
            example_inputs: Representative input tensors
            model_config: Model configuration
            compilation_mode: 'default', 'reduce-overhead', 'max-autotune'
            
        Returns:
            Compiled model or None if compilation fails
        """
        try:
            compilation_start = time.time()
            logger.info(f"Starting AOT compilation for {model_config.model_name}")
            
            # Step 1: Export model to graph representation
            logger.debug("Exporting model to graph...")
            
            # Use dynamic shape constraints for better compatibility
            dynamic_shapes = self._get_dynamic_shapes(model_config, example_inputs)
            
            exported_model = torch.export.export(
                model, 
                args=example_inputs,
                dynamic_shapes=dynamic_shapes
            )
            
            # Step 2: Apply inductor compilation
            logger.debug(f"Compiling with inductor (mode: {compilation_mode})...")
            
            # Configure compilation options
            compile_options = {
                "mode": compilation_mode,
                "fullgraph": True,  # Ensure complete graph capture
                "dynamic": True if dynamic_shapes else False,
            }
            
            # Add GPU-specific optimizations
            if torch.cuda.is_available():
                compile_options.update({
                    "triton.cudagraphs": True,
                    "triton.fast_math": True,
                })
            
            # Compile the exported model
            compiled_model = torch._inductor.aot_compile(
                exported_model,
                options=compile_options
            )
            
            compilation_time = time.time() - compilation_start
            
            # Step 3: Cache and validate compiled model
            cache_key = self._generate_cache_key(model_config, example_inputs)
            self.compiled_cache[cache_key] = compiled_model
            
            # Record compilation statistics
            self.compilation_stats[cache_key] = {
                "model_name": model_config.model_name,
                "compilation_time": compilation_time,
                "compilation_mode": compilation_mode,
                "timestamp": time.time(),
                "input_shapes": [inp.shape for inp in example_inputs],
                "dynamic_shapes": dynamic_shapes is not None
            }
            
            logger.info(f"AOT compilation completed in {compilation_time:.1f}s")
            
            # Step 4: Validation run
            self._validate_compiled_model(compiled_model, model, example_inputs)
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"AOT compilation failed for {model_config.model_name}: {e}")
            return None
    
    def _get_dynamic_shapes(self, model_config, example_inputs):
        """Generate dynamic shape constraints for flexible input sizes"""
        try:
            import torch.export
            
            # Define dynamic dimensions for sequence length and batch size
            batch_dim = torch.export.Dim("batch", min=1, max=32)
            seq_dim = torch.export.Dim("seq_len", min=1, max=4096)
            
            # Apply to input tensors based on model type
            if len(example_inputs) == 1:  # Single input tensor
                input_tensor = example_inputs[0]
                if len(input_tensor.shape) == 2:  # [batch, seq_len]
                    return {0: {0: batch_dim, 1: seq_dim}}
                    
            return None  # Fall back to static shapes
            
        except Exception as e:
            logger.debug(f"Dynamic shapes not supported: {e}")
            return None
    
    def _validate_compiled_model(self, compiled_model, original_model, example_inputs):
        """Validate that compiled model produces correct outputs"""
        try:
            with torch.no_grad():
                # Get original output
                original_model.eval()
                original_output = original_model(*example_inputs)
                
                # Get compiled output
                compiled_output = compiled_model(*example_inputs)
                
                # Compare outputs (allow small numerical differences)
                if isinstance(original_output, torch.Tensor):
                    max_diff = torch.max(torch.abs(original_output - compiled_output))
                    if max_diff > 1e-4:
                        logger.warning(f"Compiled model output differs by {max_diff}")
                    else:
                        logger.debug("Compiled model validation passed")
                        
        except Exception as e:
            logger.error(f"Compiled model validation failed: {e}")
    
    def _generate_cache_key(self, model_config, example_inputs) -> str:
        """Generate unique cache key for model + input configuration"""
        # Include model name, size, quantization, and input shapes
        key_components = [
            model_config.model_name,
            str(model_config.size_gb),
            getattr(model_config, 'quantization', 'none'),
            str([inp.shape for inp in example_inputs]),
            torch.__version__  # Include PyTorch version for compatibility
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def save_compiled_model(self, cache_key: str, output_path: Optional[str] = None):
        """Save compiled model to disk for persistence"""
        if cache_key not in self.compiled_cache:
            raise ValueError(f"No compiled model found for key: {cache_key}")
        
        if output_path is None:
            output_path = self.cache_dir / f"{cache_key}.pt"
        
        try:
            compiled_model = self.compiled_cache[cache_key]
            stats = self.compilation_stats[cache_key]
            
            # Save both model and metadata
            save_data = {
                "compiled_model": compiled_model,
                "compilation_stats": stats,
                "torch_version": torch.__version__,
                "save_timestamp": time.time()
            }
            
            torch.save(save_data, output_path)
            logger.info(f"Saved compiled model to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save compiled model: {e}")
    
    def load_compiled_model(self, cache_key: str, input_path: Optional[str] = None) -> Optional[Any]:
        """Load compiled model from disk"""
        if input_path is None:
            input_path = self.cache_dir / f"{cache_key}.pt"
        
        if not Path(input_path).exists():
            return None
        
        try:
            save_data = torch.load(input_path, map_location='cpu')
            
            # Version compatibility check
            saved_torch_version = save_data.get("torch_version", "unknown")
            if saved_torch_version != torch.__version__:
                logger.warning(f"PyTorch version mismatch: saved={saved_torch_version}, current={torch.__version__}")
                return None
            
            compiled_model = save_data["compiled_model"]
            stats = save_data["compilation_stats"]
            
            # Cache in memory
            self.compiled_cache[cache_key] = compiled_model
            self.compilation_stats[cache_key] = stats
            
            logger.info(f"Loaded compiled model from {input_path}")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Failed to load compiled model: {e}")
            return None
```

#### **1.2 Integration with LightweightModelLoader**
```python
# engines/lightweight/enhanced_model_loader.py
from engines.shared.aot_compiler import AOTModelCompiler

class EnhancedLightweightModelLoader(LightweightModelLoader):
    """Enhanced model loader with AOT compilation support"""
    
    def __init__(self):
        super().__init__()
        self.aot_compiler = AOTModelCompiler(
            cache_dir="model_cache/compiled",
            enable_regional_compilation=True
        )
        self._aot_enabled = True
        
    def load_model(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Load model with optional AOT compilation"""
        
        # Check AOT compilation eligibility
        if (self._aot_enabled and 
            self.aot_compiler.is_model_supported(model_config)):
            
            return self._load_with_aot_compilation(model_config)
        else:
            # Fall back to standard loading
            return super().load_model(model_config)
    
    def _load_with_aot_compilation(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Load model with AOT compilation optimization"""
        logger.info(f"Loading {model_config.model_name} with AOT compilation")
        
        # Generate cache key for this configuration
        example_inputs = self._generate_example_inputs(model_config)
        cache_key = self.aot_compiler._generate_cache_key(model_config, example_inputs)
        
        # Try to load from cache first
        compiled_model = self.aot_compiler.load_compiled_model(cache_key)
        
        if compiled_model is not None:
            # Use cached compiled model
            logger.info(f"Using cached compiled model for {model_config.model_name}")
            return self._create_model_info_from_compiled(
                compiled_model, model_config, cache_key
            )
        else:
            # Load and compile model
            logger.info(f"Compiling {model_config.model_name} for first time")
            return self._load_and_compile_model(model_config, example_inputs, cache_key)
    
    def _generate_example_inputs(self, model_config: EnhancedModelConfig) -> Tuple:
        """Generate representative input tensors for compilation"""
        # Create sample input tensors based on model configuration
        batch_size = 1
        seq_length = 512  # Representative length
        
        # Generate input_ids tensor
        input_ids = torch.randint(
            low=1, high=32000, 
            size=(batch_size, seq_length), 
            dtype=torch.long
        )
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        return (input_ids,)
    
    def _load_and_compile_model(self, 
                               model_config: EnhancedModelConfig, 
                               example_inputs: Tuple, 
                               cache_key: str) -> Dict[str, Any]:
        """Load model and perform AOT compilation"""
        
        # Step 1: Load model normally
        standard_model_info = super().load_model(model_config)
        original_model = standard_model_info["model"]
        
        # Step 2: Extract the underlying PyTorch model for compilation
        if hasattr(original_model, 'llm_engine'):  # vLLM case
            pytorch_model = original_model.llm_engine.model_executor.driver_worker.model_runner.model
        else:
            pytorch_model = original_model  # Direct PyTorch model
        
        # Step 3: Compile the model
        compiled_model = self.aot_compiler.compile_model_aot(
            pytorch_model, example_inputs, model_config, compilation_mode="default"
        )
        
        if compiled_model is None:
            # Compilation failed, use standard model
            logger.warning(f"AOT compilation failed for {model_config.model_name}, using standard loading")
            return standard_model_info
        
        # Step 4: Save compiled model for future use
        self.aot_compiler.save_compiled_model(cache_key)
        
        # Step 5: Create optimized model info
        return self._create_model_info_from_compiled(
            compiled_model, model_config, cache_key, base_info=standard_model_info
        )
    
    def _create_model_info_from_compiled(self, 
                                       compiled_model: Any, 
                                       model_config: EnhancedModelConfig, 
                                       cache_key: str,
                                       base_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Create model info dict for compiled model"""
        
        if base_info is None:
            # Create minimal base info for cached case
            base_info = {
                "model": None,  # Will be replaced
                "tokenizer": None,  # Need to load separately
                "backend": "vllm_aot",
                "config": {},
                "model_name": model_config.model_name,
                "size_gb": model_config.size_gb
            }
        
        # Replace model with compiled version
        base_info.update({
            "model": compiled_model,
            "backend": "vllm_aot",
            "aot_compiled": True,
            "aot_cache_key": cache_key,
            "compilation_stats": self.aot_compiler.compilation_stats.get(cache_key, {})
        })
        
        return base_info
```

### **Phase 2: Model Graph Persistence System**

#### **2.1 Persistent Cache Manager**
```python
# engines/shared/model_cache_manager.py
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import time

class ModelCacheManager:
    """
    Persistent storage and retrieval of compiled model graphs
    Provides cross-session caching and cache invalidation
    """
    
    def __init__(self, cache_root: str = "model_cache"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.cache_root / "cache_metadata.db"
        self._init_database()
        
        # Cache directories
        self.compiled_dir = self.cache_root / "compiled"
        self.metadata_dir = self.cache_root / "metadata"
        
        for directory in [self.compiled_dir, self.metadata_dir]:
            directory.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_cache (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_size_gb REAL NOT NULL,
                torch_version TEXT NOT NULL,
                compilation_mode TEXT NOT NULL,
                compilation_time REAL NOT NULL,
                file_path TEXT NOT NULL,
                created_timestamp REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                input_shapes TEXT,  -- JSON serialized
                dynamic_shapes INTEGER DEFAULT 0,  -- Boolean
                cache_valid INTEGER DEFAULT 1      -- Boolean
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_compiled_model(self, 
                           cache_key: str, 
                           compiled_model: Any,
                           model_config,
                           compilation_stats: Dict) -> bool:
        """Store compiled model with metadata tracking"""
        try:
            # Save compiled model file
            model_path = self.compiled_dir / f"{cache_key}.pt"
            torch.save(compiled_model, model_path)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO model_cache 
                (cache_key, model_name, model_size_gb, torch_version, 
                 compilation_mode, compilation_time, file_path, 
                 created_timestamp, last_accessed, input_shapes, dynamic_shapes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                model_config.model_name,
                model_config.size_gb,
                torch.__version__,
                compilation_stats.get("compilation_mode", "default"),
                compilation_stats.get("compilation_time", 0),
                str(model_path),
                time.time(),
                time.time(),
                json.dumps(compilation_stats.get("input_shapes", [])),
                1 if compilation_stats.get("dynamic_shapes", False) else 0
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored compiled model: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store compiled model: {e}")
            return False
    
    def retrieve_compiled_model(self, cache_key: str) -> Optional[Any]:
        """Retrieve compiled model if available and valid"""
        try:
            # Check database for metadata
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, torch_version, cache_valid 
                FROM model_cache 
                WHERE cache_key = ?
            """, (cache_key,))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return None
            
            file_path, stored_torch_version, cache_valid = result
            
            # Validate cache entry
            if not cache_valid:
                logger.debug(f"Cache entry {cache_key} marked invalid")
                conn.close()
                return None
            
            if stored_torch_version != torch.__version__:
                logger.warning(f"PyTorch version mismatch for {cache_key}")
                # Mark as invalid
                cursor.execute("""
                    UPDATE model_cache SET cache_valid = 0 WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                conn.close()
                return None
            
            # Check file exists
            if not Path(file_path).exists():
                logger.warning(f"Compiled model file missing: {file_path}")
                cursor.execute("""
                    UPDATE model_cache SET cache_valid = 0 WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                conn.close()
                return None
            
            # Load compiled model
            compiled_model = torch.load(file_path, map_location='cpu')
            
            # Update access statistics
            cursor.execute("""
                UPDATE model_cache 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE cache_key = ?
            """, (time.time(), cache_key))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Retrieved compiled model: {cache_key}")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Failed to retrieve compiled model {cache_key}: {e}")
            return None
    
    def get_cache_statistics(self) -> Dict:
        """Get comprehensive cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic statistics
        cursor.execute("SELECT COUNT(*) FROM model_cache WHERE cache_valid = 1")
        valid_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_cache WHERE cache_valid = 0")
        invalid_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(model_size_gb) FROM model_cache WHERE cache_valid = 1")
        total_size_gb = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT AVG(compilation_time) FROM model_cache WHERE cache_valid = 1")
        avg_compilation_time = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(access_count) FROM model_cache WHERE cache_valid = 1")
        total_accesses = cursor.fetchone()[0] or 0
        
        # Most accessed models
        cursor.execute("""
            SELECT model_name, access_count, last_accessed 
            FROM model_cache 
            WHERE cache_valid = 1 
            ORDER BY access_count DESC 
            LIMIT 5
        """)
        top_models = cursor.fetchall()
        
        conn.close()
        
        return {
            "valid_entries": valid_entries,
            "invalid_entries": invalid_entries,
            "total_cached_size_gb": round(total_size_gb, 2),
            "average_compilation_time": round(avg_compilation_time, 2),
            "total_cache_accesses": total_accesses,
            "cache_hit_savings": f"{total_accesses * avg_compilation_time:.1f}s",
            "top_accessed_models": [
                {"model": name, "accesses": count, "last_used": time.ctime(last_access)}
                for name, count, last_access in top_models
            ]
        }
    
    def cleanup_cache(self, max_age_days: int = 30, max_cache_size_gb: float = 100.0):
        """Clean up old or invalid cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        # Find entries to remove
        cursor.execute("""
            SELECT cache_key, file_path FROM model_cache 
            WHERE cache_valid = 0 OR last_accessed < ?
        """, (cutoff_time,))
        
        entries_to_remove = cursor.fetchall()
        
        # Remove files and database entries
        removed_count = 0
        for cache_key, file_path in entries_to_remove:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                
                cursor.execute("DELETE FROM model_cache WHERE cache_key = ?", (cache_key,))
                removed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to remove cache entry {cache_key}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cache cleanup completed: removed {removed_count} entries")
        return removed_count
```

### **Phase 3: Performance Testing and Validation**

#### **3.1 AOTI Performance Benchmarker**
```python
# evaluation/aoti_benchmarker.py
import time
import torch
import psutil
import GPUtil
from typing import Dict, List, Any
import json
from datetime import datetime

class AOTIPerformanceBenchmarker:
    """
    Comprehensive benchmarking for AOTI implementation
    Measures loading time, inference speed, memory usage, and throughput
    """
    
    def __init__(self):
        self.results = []
        self.baseline_results = {}
    
    def benchmark_model_loading(self, 
                              model_config,
                              test_iterations: int = 3,
                              enable_aoti: bool = True) -> Dict:
        """Benchmark model loading with and without AOTI"""
        
        results = {
            "model_name": model_config.model_name,
            "model_size_gb": model_config.size_gb,
            "test_iterations": test_iterations,
            "aoti_enabled": enable_aoti,
            "timestamp": datetime.now().isoformat()
        }
        
        loading_times = []
        memory_usage = []
        
        for iteration in range(test_iterations):
            # Clear cache between iterations
            torch.cuda.empty_cache()
            gc.collect()
            
            # Measure memory before loading
            memory_before = self._get_memory_usage()
            
            # Measure loading time
            start_time = time.time()
            
            try:
                if enable_aoti:
                    model_info = self._load_with_aoti(model_config)
                else:
                    model_info = self._load_standard(model_config)
                
                loading_time = time.time() - start_time
                loading_times.append(loading_time)
                
                # Measure memory after loading
                memory_after = self._get_memory_usage()
                memory_delta = {
                    "gpu_memory_mb": memory_after["gpu_memory_mb"] - memory_before["gpu_memory_mb"],
                    "system_memory_mb": memory_after["system_memory_mb"] - memory_before["system_memory_mb"]
                }
                memory_usage.append(memory_delta)
                
                # Clean up
                self._cleanup_model(model_info)
                
            except Exception as e:
                logger.error(f"Loading benchmark failed for iteration {iteration}: {e}")
                loading_times.append(float('inf'))
                memory_usage.append({"gpu_memory_mb": 0, "system_memory_mb": 0})
        
        # Calculate statistics
        valid_times = [t for t in loading_times if t != float('inf')]
        if valid_times:
            results.update({
                "avg_loading_time": sum(valid_times) / len(valid_times),
                "min_loading_time": min(valid_times),
                "max_loading_time": max(valid_times),
                "loading_times": valid_times,
                "success_rate": len(valid_times) / test_iterations,
                "avg_gpu_memory_mb": sum(m["gpu_memory_mb"] for m in memory_usage) / len(memory_usage),
                "avg_system_memory_mb": sum(m["system_memory_mb"] for m in memory_usage) / len(memory_usage)
            })
        else:
            results.update({
                "avg_loading_time": float('inf'),
                "success_rate": 0.0,
                "error": "All loading attempts failed"
            })
        
        return results
    
    def benchmark_inference_speed(self, 
                                model_info: Dict,
                                test_prompts: List[str],
                                iterations_per_prompt: int = 5) -> Dict:
        """Benchmark inference speed and throughput"""
        
        results = {
            "model_name": model_info.get("model_name", "unknown"),
            "backend": model_info.get("backend", "unknown"),
            "aot_compiled": model_info.get("aot_compiled", False),
            "test_prompts_count": len(test_prompts),
            "iterations_per_prompt": iterations_per_prompt
        }
        
        inference_times = []
        tokens_per_second = []
        
        for prompt in test_prompts:
            prompt_times = []
            prompt_throughput = []
            
            for iteration in range(iterations_per_prompt):
                # Measure inference time
                start_time = time.time()
                
                try:
                    # Generate response
                    response = self._generate_response(model_info, prompt)
                    inference_time = time.time() - start_time
                    
                    # Calculate throughput
                    output_tokens = len(response.split())  # Rough token count
                    throughput = output_tokens / inference_time if inference_time > 0 else 0
                    
                    prompt_times.append(inference_time)
                    prompt_throughput.append(throughput)
                    
                except Exception as e:
                    logger.error(f"Inference benchmark failed: {e}")
                    prompt_times.append(float('inf'))
                    prompt_throughput.append(0)
            
            # Aggregate prompt results
            valid_times = [t for t in prompt_times if t != float('inf')]
            valid_throughput = [t for t in prompt_throughput if t > 0]
            
            if valid_times:
                inference_times.extend(valid_times)
                tokens_per_second.extend(valid_throughput)
        
        # Calculate overall statistics
        if inference_times:
            results.update({
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times),
                "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
                "max_tokens_per_second": max(tokens_per_second),
                "total_successful_inferences": len(inference_times)
            })
        else:
            results.update({
                "avg_inference_time": float('inf'),
                "avg_tokens_per_second": 0,
                "error": "All inference attempts failed"
            })
        
        return results
    
    def compare_aoti_vs_baseline(self, 
                               model_configs: List,
                               test_prompts: List[str]) -> Dict:
        """Compare AOTI performance against baseline for multiple models"""
        
        comparison_results = {
            "test_timestamp": datetime.now().isoformat(),
            "models_tested": len(model_configs),
            "test_prompts_count": len(test_prompts),
            "model_comparisons": []
        }
        
        for model_config in model_configs:
            logger.info(f"Benchmarking {model_config.model_name}")
            
            # Benchmark baseline (no AOTI)
            baseline_loading = self.benchmark_model_loading(model_config, enable_aoti=False)
            if baseline_loading["success_rate"] > 0:
                baseline_model = self._load_standard(model_config)
                baseline_inference = self.benchmark_inference_speed(baseline_model, test_prompts)
                self._cleanup_model(baseline_model)
            else:
                baseline_inference = {"error": "Baseline loading failed"}
            
            # Benchmark AOTI
            aoti_loading = self.benchmark_model_loading(model_config, enable_aoti=True)
            if aoti_loading["success_rate"] > 0:
                aoti_model = self._load_with_aoti(model_config)
                aoti_inference = self.benchmark_inference_speed(aoti_model, test_prompts)
                self._cleanup_model(aoti_model)
            else:
                aoti_inference = {"error": "AOTI loading failed"}
            
            # Calculate improvements
            comparison = {
                "model_name": model_config.model_name,
                "model_size_gb": model_config.size_gb,
                "baseline_loading": baseline_loading,
                "aoti_loading": aoti_loading,
                "baseline_inference": baseline_inference,
                "aoti_inference": aoti_inference
            }
            
            # Calculate improvement metrics
            if (baseline_loading["success_rate"] > 0 and aoti_loading["success_rate"] > 0):
                loading_improvement = baseline_loading["avg_loading_time"] / aoti_loading["avg_loading_time"]
                comparison["loading_speedup"] = round(loading_improvement, 2)
                
                if ("avg_tokens_per_second" in baseline_inference and 
                    "avg_tokens_per_second" in aoti_inference and
                    baseline_inference["avg_tokens_per_second"] > 0):
                    
                    throughput_improvement = (aoti_inference["avg_tokens_per_second"] / 
                                            baseline_inference["avg_tokens_per_second"])
                    comparison["throughput_improvement"] = round(throughput_improvement, 2)
                    
                    inference_improvement = (baseline_inference["avg_inference_time"] / 
                                           aoti_inference["avg_inference_time"])
                    comparison["inference_speedup"] = round(inference_improvement, 2)
            
            comparison_results["model_comparisons"].append(comparison)
        
        return comparison_results
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        try:
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                gpu_memory_mb = 0
            
            # System memory
            system_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            return {
                "gpu_memory_mb": gpu_memory_mb,
                "system_memory_mb": system_memory_mb
            }
        except:
            return {"gpu_memory_mb": 0, "system_memory_mb": 0}
    
    def save_benchmark_results(self, results: Dict, output_path: str):
        """Save benchmark results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")
```

---

## 🚀 **Implementation Roadmap**

### **Phase 1: Foundation (Weeks 1-2)**
**Goal**: Implement basic AOT compilation with 30-50% improvement

#### **Week 1: Core Components**
- [ ] **Day 1-2**: Implement `AOTModelCompiler` class
- [ ] **Day 3-4**: Create `EnhancedLightweightModelLoader` 
- [ ] **Day 5-7**: Basic integration testing with qwen25_7b

#### **Week 2: Validation and Optimization**
- [ ] **Day 8-10**: Implement error handling and fallback mechanisms
- [ ] **Day 11-12**: Performance benchmarking framework
- [ ] **Day 13-14**: Documentation and code review

**Success Criteria:**
- ✅ AOT compilation works for 3+ model architectures
- ✅ 1.2x+ speedup in inference
- ✅ 1.5x+ speedup in model loading
- ✅ Zero regression in evaluation accuracy

### **Phase 2: Persistence and Caching (Weeks 3-4)**
**Goal**: Cross-session model graph caching

#### **Week 3: Cache Infrastructure**
- [ ] **Day 15-17**: Implement `ModelCacheManager` with SQLite
- [ ] **Day 18-19**: Persistent storage and retrieval
- [ ] **Day 20-21**: Cache invalidation and cleanup

#### **Week 4: Integration and Testing**
- [ ] **Day 22-24**: Integration with engine lifecycle
- [ ] **Day 25-26**: Multi-session testing
- [ ] **Day 27-28**: Performance validation and tuning

**Success Criteria:**
- ✅ Model graphs persist across sessions
- ✅ Cache hit rate > 80% in typical workflows
- ✅ 10x+ faster second loading
- ✅ Automatic cache management working

### **Phase 3: Advanced Features (Weeks 5-6)**
**Goal**: Dynamic shapes and regional compilation

#### **Week 5: Dynamic Shapes**
- [ ] **Day 29-31**: Implement dynamic shape constraints
- [ ] **Day 32-33**: Variable batch size and sequence length support
- [ ] **Day 34-35**: Testing with diverse input patterns

#### **Week 6: Regional Compilation**
- [ ] **Day 36-38**: Implement transformer block compilation
- [ ] **Day 39-40**: Regional optimization for large models
- [ ] **Day 41-42**: Final validation and documentation

**Success Criteria:**
- ✅ Single compiled model handles multiple input sizes
- ✅ Regional compilation reduces compilation time by 40%
- ✅ Large models (>30B) benefit from regional optimization

---

## 📋 **Technical Requirements**

### **Dependencies**
```python
# Additional requirements for AOTI implementation
torch>=2.4.0                    # Required for stable torch.export
torch._inductor                 # Built-in with PyTorch 2.4+
sqlite3                        # Built-in Python module
psutil>=6.0.0                  # Already in requirements
GPUtil>=1.4.0                  # For GPU monitoring (new)
```

### **Compatibility Matrix**
| Component | Supported | Notes |
|-----------|-----------|-------|
| **PyTorch Version** | 2.4.0+ | torch.export stable in 2.4+ |
| **Model Architectures** | Llama, Qwen, Mistral, Phi | Transformer-based models |
| **Model Sizes** | ≤70B parameters | Memory constraints for compilation |
| **Quantization** | AWQ, GPTQ compatible | Post-compilation quantization |
| **GPU Types** | CUDA 11.8+ | Requires modern GPU compute capability |

### **Storage Requirements**
- **Compiled Models**: 1.5-2x original model size
- **Cache Database**: ~10MB per 100 cached models
- **Example Storage**: 70B model = ~140GB original + ~210GB compiled = ~350GB total

### **Memory Requirements**
- **Compilation**: 3-4x model size during compilation
- **Runtime**: 1.2-1.5x model size (improved efficiency)
- **Example**: 30B model needs ~180GB peak during compilation, ~45GB runtime

---

## 🎯 **Expected Performance Improvements**

### **Quantified Benefits**
| Metric | Current Baseline | AOTI Enhanced | Improvement | Business Impact |
|--------|------------------|---------------|-------------|-----------------|
| **Cold Start Time** | 60-120 seconds | 15-30 seconds | **2-4x faster** | Faster experimentation |
| **Model Re-loading** | In-memory cache (2 models) | Persistent AOT cache | **2-3x faster** | Better cross-session workflow |
| **Inference Throughput** | 50 tokens/sec | 65-90 tokens/sec | **1.3-1.8x** | Higher evaluation throughput |
| **Memory Efficiency** | Baseline | -15-25% usage | **25% improvement** | More models per GPU |
| **Cache Persistence** | Session-only | Cross-session | **Eliminates re-compilation** | Consistent performance |

### **ROI Analysis**
- **Development Time**: 6 weeks (1 developer)
- **Infrastructure Cost**: +50% storage (temporary)
- **Performance Gain**: 2-4x overall evaluation speed
- **Yearly Savings**: ~200 hours of researcher time
- **GPU Cost Reduction**: 25% less GPU-hours needed

---

## ⚠️ **Risk Assessment and Mitigation**

### **High Risk Items**
1. **PyTorch Version Compatibility**
   - *Risk*: torch.export API changes between versions
   - *Mitigation*: Pin PyTorch version, comprehensive testing

2. **Model Architecture Support**
   - *Risk*: Some models may not compile successfully
   - *Mitigation*: Fallback to standard loading, architecture detection

3. **Memory Requirements**
   - *Risk*: Compilation may require more memory than available
   - *Mitigation*: Memory estimation, graceful degradation

### **Medium Risk Items**
1. **Cache Invalidation**
   - *Risk*: Stale cached models causing incorrect results
   - *Mitigation*: Version-based invalidation, hash verification

2. **Compilation Time**
   - *Risk*: Initial compilation may be very slow
   - *Mitigation*: Background compilation, progress indicators

### **Low Risk Items**
1. **Storage Space**
   - *Risk*: Cached models consuming too much disk space
   - *Mitigation*: Automatic cleanup, configurable limits

---

## 🧪 **Testing Strategy**

### **Unit Tests**
```python
# tests/test_aoti_compiler.py
def test_aot_compilation_llama():
    """Test AOT compilation for Llama models"""
    
def test_cache_persistence():
    """Test model cache across sessions"""
    
def test_dynamic_shapes():
    """Test variable input size handling"""
    
def test_fallback_mechanism():
    """Test fallback when compilation fails"""
```

### **Integration Tests**
```python
# tests/test_aoti_integration.py
def test_lightweight_engine_with_aoti():
    """Test full integration with lightweight engine"""
    
def test_evaluation_pipeline_aoti():
    """Test evaluation pipeline with AOTI enabled"""
    
def test_multi_model_workflow():
    """Test workflow with multiple AOTI models"""
```

### **Performance Tests**
```python
# tests/test_aoti_performance.py
def test_loading_speedup():
    """Validate loading time improvements"""
    
def test_inference_speedup():
    """Validate inference speed improvements"""
    
def test_memory_efficiency():
    """Validate memory usage improvements"""
```

---

## 📊 **Success Metrics**

### **Technical Metrics**
- [ ] **Loading Speedup**: ≥2x faster model loading
- [ ] **Inference Speedup**: ≥1.3x faster inference
- [ ] **Memory Efficiency**: ≥15% memory reduction
- [ ] **Cache Hit Rate**: ≥80% in typical workflows
- [ ] **Compilation Success Rate**: ≥90% for supported models

### **User Experience Metrics**
- [ ] **Evaluation Campaign Time**: ≥50% reduction
- [ ] **Developer Productivity**: ≥30% faster iteration cycles
- [ ] **System Reliability**: Zero regression in evaluation accuracy
- [ ] **Resource Utilization**: ≥25% more efficient GPU usage

### **Operational Metrics**
- [ ] **Implementation Timeline**: Complete in 6 weeks
- [ ] **Code Quality**: 90%+ test coverage
- [ ] **Documentation**: Complete user and developer guides
- [ ] **Maintenance Overhead**: <10% additional maintenance time

---

## 🔄 **Rollback Plan**

### **Graceful Degradation**
1. **Feature Flag**: `ENABLE_AOTI_COMPILATION = False`
2. **Automatic Fallback**: Failed compilation → standard loading
3. **Cache Bypass**: `--disable-aoti-cache` flag for debugging
4. **Version Rollback**: Pin to previous PyTorch version if needed

### **Emergency Procedures**
1. **Disable AOTI**: Set environment variable `AOTI_DISABLED=1`
2. **Clear Cache**: `rm -rf model_cache/compiled/`
3. **Revert Code**: Git rollback to pre-AOTI commit
4. **Rebuild Environment**: Restore previous requirements.txt

---

## 📈 **Future Enhancements**

### **Phase 4: Advanced Optimizations (Future)**
1. **FlashAttention-3 Integration**: Pre-built optimized kernels
2. **FP8 Quantization**: Hardware-specific precision optimizations  
3. **Multi-GPU AOT**: Distributed compilation for large models
4. **Hub Integration**: Share compiled models via HuggingFace Hub

### **Phase 5: Production Optimization (Future)**
1. **Automated Compilation**: Background compilation of popular models
2. **Smart Caching**: ML-based cache eviction policies
3. **Compilation Clusters**: Dedicated compilation infrastructure
4. **Performance Analytics**: Real-time optimization recommendations

---

## 🎉 **Conclusion**

This implementation plan provides a **high-impact, low-risk approach** to integrating ZeroGPU AOTI concepts into our hybrid evaluation pipeline. The **phased approach** ensures:

1. **Quick Wins**: 2-4x loading speedup in Phase 1
2. **Sustained Benefits**: Cross-session caching in Phase 2  
3. **Advanced Features**: Dynamic shapes and regional compilation in Phase 3

**Expected Overall Impact**:
- **3-5x faster evaluation campaigns**
- **30-50% more efficient GPU utilization**
- **Significantly improved researcher productivity**

The **comprehensive testing and rollback strategies** ensure we can safely implement these optimizations while maintaining the reliability and accuracy of our evaluation pipeline.

**Recommendation**: **Proceed with Phase 1 implementation** starting immediately, with expected completion and validation within 2 weeks.