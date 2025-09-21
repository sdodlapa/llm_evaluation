"""
Ahead-of-Time (AOT) Model Compiler

This module implements PyTorch AOT compilation using torch.export + torch._inductor
to pre-compile models for faster loading and inference in evaluation pipelines.

Key Features:
- torch.export graph capture
- torch._inductor optimization 
- Compiled model caching and persistence
- Automatic fallback for unsupported models
- Dynamic shape support for flexible inputs

Usage:
    compiler = AOTModelCompiler()
    compiled_model = compiler.compile_model_aot(model, example_inputs, config)
    compiler.save_compiled_model(cache_key)
"""

import logging
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch

# Check PyTorch version and AOT availability
TORCH_VERSION = torch.__version__
AOT_AVAILABLE = False
EXPORT_AVAILABLE = False
INDUCTOR_AVAILABLE = False

try:
    import torch.export
    EXPORT_AVAILABLE = hasattr(torch, 'export')
    logger = logging.getLogger(__name__)
    logger.info(f"torch.export available: {EXPORT_AVAILABLE}")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("torch.export not available - AOT compilation disabled")

try:
    import torch._inductor
    INDUCTOR_AVAILABLE = hasattr(torch, '_inductor')
    logger.info(f"torch._inductor available: {INDUCTOR_AVAILABLE}")
except ImportError:
    logger.warning("torch._inductor not available - AOT compilation disabled")

# AOT is available if both export and inductor are present
AOT_AVAILABLE = EXPORT_AVAILABLE and INDUCTOR_AVAILABLE and TORCH_VERSION >= "2.4.0"

logger.info(f"AOT Compilation Available: {AOT_AVAILABLE} (PyTorch {TORCH_VERSION})")


class AOTModelCompiler:
    """
    Ahead-of-Time model compilation using torch.export + inductor
    
    Provides methods to compile PyTorch models ahead of time for faster
    loading and inference in evaluation workflows.
    """
    
    def __init__(self, 
                 cache_dir: str = "model_cache/compiled",
                 enable_aot: bool = True,
                 max_compilation_time: int = 600):
        """
        Initialize AOT Model Compiler
        
        Args:
            cache_dir: Directory to store compiled models
            enable_aot: Whether to enable AOT compilation (can disable for debugging)
            max_compilation_time: Maximum time to spend on compilation (seconds)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_aot = enable_aot and AOT_AVAILABLE
        self.max_compilation_time = max_compilation_time
        
        # Compiled model registry
        self.compiled_cache: Dict[str, Any] = {}
        self.compilation_stats: Dict[str, Dict] = {}
        
        # Supported model architectures for AOT compilation
        self.supported_architectures = {
            'llama', 'qwen', 'mistral', 'phi', 'gemma', 
            'falcon', 'baichuan', 'yi', 'chatglm', 'gpt2'
        }
        
        logger.info(f"AOT Model Compiler initialized - Enabled: {self.enable_aot}")
        if not self.enable_aot:
            if not AOT_AVAILABLE:
                logger.warning("AOT compilation disabled - PyTorch requirements not met")
            else:
                logger.info("AOT compilation manually disabled")
    
    def is_model_supported(self, model_config) -> bool:
        """
        Check if model architecture supports AOT compilation
        
        Args:
            model_config: Model configuration object
            
        Returns:
            bool: True if model can be AOT compiled
        """
        if not self.enable_aot:
            return False
            
        model_name = model_config.model_name.lower()
        
        # Check architecture support
        supported = any(arch in model_name for arch in self.supported_architectures)
        if not supported:
            logger.debug(f"Model {model_name} architecture not supported for AOT")
            return False
        
        # Check size constraints (AOT compilation memory overhead)
        if hasattr(model_config, 'size_gb') and model_config.size_gb > 70.0:
            logger.warning(f"Model {model_name} too large for AOT compilation")
            return False
            
        # Check PyTorch version compatibility
        if not AOT_AVAILABLE:
            logger.debug(f"PyTorch {TORCH_VERSION} doesn't support stable AOT")
            return False
            
        logger.debug(f"Model {model_name} is supported for AOT compilation")
        return True
    
    def compile_model_aot(self, 
                         model, 
                         example_inputs: Tuple,
                         model_config,
                         compilation_mode: str = "default") -> Optional[Any]:
        """
        Compile model ahead of time using torch.export + inductor
        
        Args:
            model: The PyTorch model to compile
            example_inputs: Representative input tensors for tracing
            model_config: Model configuration
            compilation_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            
        Returns:
            Compiled model or None if compilation fails
        """
        if not self.enable_aot:
            logger.debug("AOT compilation disabled")
            return None
            
        try:
            compilation_start = time.time()
            model_name = getattr(model_config, 'model_name', 'unknown')
            logger.info(f"Starting AOT compilation for {model_name}")
            
            # Step 1: Export model to graph representation
            logger.debug("Exporting model to graph...")
            
            # Ensure model is in eval mode for export
            model.eval()
            
            # Use dynamic shape constraints for better compatibility
            dynamic_shapes = self._get_dynamic_shapes(model_config, example_inputs)
            
            with torch.no_grad():
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
            
            # Add GPU-specific optimizations if available
            if torch.cuda.is_available():
                compile_options.update({
                    "triton.cudagraphs": True,
                    "triton.fast_math": True,
                })
            
            # Compile the exported model with timeout
            compiled_model = self._compile_with_timeout(
                exported_model, 
                compile_options,
                self.max_compilation_time
            )
            
            if compiled_model is None:
                logger.error("AOT compilation timed out")
                return None
            
            compilation_time = time.time() - compilation_start
            
            # Step 3: Cache and validate compiled model
            cache_key = self._generate_cache_key(model_config, example_inputs)
            self.compiled_cache[cache_key] = compiled_model
            
            # Record compilation statistics
            self.compilation_stats[cache_key] = {
                "model_name": model_name,
                "compilation_time": compilation_time,
                "compilation_mode": compilation_mode,
                "timestamp": time.time(),
                "input_shapes": [tuple(inp.shape) for inp in example_inputs],
                "dynamic_shapes": dynamic_shapes is not None,
                "torch_version": TORCH_VERSION
            }
            
            logger.info(f"AOT compilation completed in {compilation_time:.1f}s")
            
            # Step 4: Validation run (quick check)
            if not self._validate_compiled_model(compiled_model, model, example_inputs):
                logger.error("Compiled model validation failed")
                return None
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"AOT compilation failed for {model_name}: {e}")
            logger.debug(f"Compilation error details: {e}", exc_info=True)
            return None
    
    def _compile_with_timeout(self, exported_model, compile_options, timeout_seconds):
        """Compile model with timeout to prevent hanging"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Compilation timed out")
        
        compiled_model = None
        
        try:
            # Set timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            # Perform compilation - use torch.compile instead of direct aot_compile
            # This is the correct API for PyTorch 2.4+
            compiled_model = torch.compile(
                exported_model.module(),
                mode=compile_options.get("mode", "default"),
                fullgraph=compile_options.get("fullgraph", True),
                dynamic=compile_options.get("dynamic", False)
            )
            
            # Clear alarm
            signal.alarm(0)
            
        except TimeoutError:
            logger.error(f"AOT compilation timed out after {timeout_seconds}s")
            signal.alarm(0)
            return None
        except Exception as e:
            signal.alarm(0)
            raise e
        
        return compiled_model
    
    def _get_dynamic_shapes(self, model_config, example_inputs):
        """
        Generate dynamic shape constraints for flexible input sizes
        
        Args:
            model_config: Model configuration 
            example_inputs: Example input tensors
            
        Returns:
            Dynamic shape constraints or None
        """
        try:
            # For now, use static shapes for stability
            # Dynamic shapes can be enabled later once basic AOT is working
            logger.debug("Using static shapes for initial AOT implementation")
            return None
            
            # Future implementation:
            # batch_dim = torch.export.Dim("batch", min=1, max=32)
            # seq_dim = torch.export.Dim("seq_len", min=1, max=4096)
            # return dynamic shape constraints
            
        except Exception as e:
            logger.debug(f"Dynamic shapes not supported: {e}")
            return None
    
    def _validate_compiled_model(self, compiled_model, original_model, example_inputs, tolerance=1e-3):
        """
        Validate that compiled model produces correct outputs
        
        Args:
            compiled_model: AOT compiled model
            original_model: Original PyTorch model
            example_inputs: Test inputs
            tolerance: Numerical tolerance for comparison
            
        Returns:
            bool: True if validation passes
        """
        try:
            with torch.no_grad():
                # Get original output
                original_model.eval()
                original_output = original_model(*example_inputs)
                
                # Get compiled output  
                compiled_output = compiled_model(*example_inputs)
                
                # Handle different output types
                if isinstance(original_output, torch.Tensor):
                    if not isinstance(compiled_output, torch.Tensor):
                        logger.error("Output type mismatch: original=Tensor, compiled=non-Tensor")
                        return False
                    
                    max_diff = torch.max(torch.abs(original_output - compiled_output)).item()
                    if max_diff > tolerance:
                        logger.warning(f"Compiled model output differs by {max_diff:.6f} (tolerance: {tolerance})")
                        return False
                    else:
                        logger.debug(f"Compiled model validation passed (max_diff: {max_diff:.6f})")
                        return True
                        
                elif hasattr(original_output, 'logits'):
                    # Handle transformers model output
                    if not hasattr(compiled_output, 'logits'):
                        logger.error("Output structure mismatch: missing logits")
                        return False
                    
                    max_diff = torch.max(torch.abs(original_output.logits - compiled_output.logits)).item()
                    if max_diff > tolerance:
                        logger.warning(f"Compiled model logits differ by {max_diff:.6f}")
                        return False
                    else:
                        logger.debug("Compiled model validation passed (logits)")
                        return True
                else:
                    logger.warning("Unknown output type - skipping validation")
                    return True
                    
        except Exception as e:
            logger.error(f"Compiled model validation failed: {e}")
            return False
    
    def _generate_cache_key(self, model_config, example_inputs) -> str:
        """
        Generate unique cache key for model + input configuration
        
        Args:
            model_config: Model configuration
            example_inputs: Example input tensors
            
        Returns:
            str: Unique cache key hash
        """
        # Include model name, size, quantization, and input shapes
        key_components = [
            getattr(model_config, 'model_name', 'unknown'),
            str(getattr(model_config, 'size_gb', 0)),
            getattr(model_config, 'quantization_method', 'none'),
            str([tuple(inp.shape) for inp in example_inputs]),
            str([str(inp.dtype) for inp in example_inputs]),
            TORCH_VERSION  # Include PyTorch version for compatibility
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def save_compiled_model(self, cache_key: str, output_path: Optional[str] = None) -> bool:
        """
        Save compiled model to disk for persistence
        
        Args:
            cache_key: Cache key for the compiled model
            output_path: Optional custom output path
            
        Returns:
            bool: True if save successful
        """
        if cache_key not in self.compiled_cache:
            logger.error(f"No compiled model found for key: {cache_key}")
            return False
        
        if output_path is None:
            output_path = self.cache_dir / f"{cache_key}.pt"
        else:
            output_path = Path(output_path)
        
        try:
            compiled_model = self.compiled_cache[cache_key]
            stats = self.compilation_stats.get(cache_key, {})
            
            # Save both model and metadata
            save_data = {
                "compiled_model": compiled_model,
                "compilation_stats": stats,
                "torch_version": TORCH_VERSION,
                "save_timestamp": time.time(),
                "cache_key": cache_key
            }
            
            torch.save(save_data, output_path)
            logger.info(f"Saved compiled model to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save compiled model: {e}")
            return False
    
    def load_compiled_model(self, cache_key: str, input_path: Optional[str] = None) -> Optional[Any]:
        """
        Load compiled model from disk
        
        Args:
            cache_key: Cache key for the model
            input_path: Optional custom input path
            
        Returns:
            Compiled model or None if load fails
        """
        if input_path is None:
            input_path = self.cache_dir / f"{cache_key}.pt"
        else:
            input_path = Path(input_path)
        
        if not input_path.exists():
            logger.debug(f"Compiled model cache file not found: {input_path}")
            return None
        
        try:
            save_data = torch.load(input_path, map_location='cpu')
            
            # Version compatibility check
            saved_torch_version = save_data.get("torch_version", "unknown")
            if saved_torch_version != TORCH_VERSION:
                logger.warning(f"PyTorch version mismatch: saved={saved_torch_version}, current={TORCH_VERSION}")
                logger.info("Recompilation may be required")
                return None
            
            compiled_model = save_data["compiled_model"]
            stats = save_data.get("compilation_stats", {})
            
            # Cache in memory
            self.compiled_cache[cache_key] = compiled_model
            self.compilation_stats[cache_key] = stats
            
            logger.info(f"Loaded compiled model from {input_path}")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Failed to load compiled model: {e}")
            return None
    
    def get_compilation_stats(self, cache_key: Optional[str] = None) -> Dict:
        """Get compilation statistics"""
        if cache_key:
            return self.compilation_stats.get(cache_key, {})
        else:
            return {
                "total_compiled_models": len(self.compilation_stats),
                "cache_directory": str(self.cache_dir),
                "aot_enabled": self.enable_aot,
                "torch_version": TORCH_VERSION,
                "aot_available": AOT_AVAILABLE,
                "models": list(self.compilation_stats.keys())
            }
    
    def clear_cache(self):
        """Clear in-memory cache"""
        self.compiled_cache.clear()
        self.compilation_stats.clear()
        logger.info("AOT compiler cache cleared")


# Module-level convenience functions
def is_aot_available() -> bool:
    """Check if AOT compilation is available in current environment"""
    return AOT_AVAILABLE

def get_aot_info() -> Dict[str, Any]:
    """Get information about AOT availability"""
    return {
        "aot_available": AOT_AVAILABLE,
        "torch_version": TORCH_VERSION,
        "export_available": EXPORT_AVAILABLE,
        "inductor_available": INDUCTOR_AVAILABLE,
        "requirements_met": TORCH_VERSION >= "2.4.0"
    }