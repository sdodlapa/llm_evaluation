"""
Custom JSON Serialization Framework
Handles vLLM, torch, and other ML framework objects for robust evaluation result persistence
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class MLObjectEncoder(json.JSONEncoder):
    """Custom JSON encoder for ML framework objects"""
    
    def default(self, obj):
        # Handle vLLM RequestOutput objects
        if hasattr(obj, '__class__') and 'RequestOutput' in str(type(obj)):
            return self._serialize_vllm_request_output(obj)
        
        # Handle vLLM CompletionOutput objects
        if hasattr(obj, '__class__') and 'CompletionOutput' in str(type(obj)):
            return self._serialize_completion_output(obj)
        
        # Handle torch tensors
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return self._serialize_torch_tensor(obj)
        except ImportError:
            pass  # torch not available
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
            
        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return self._serialize_numpy_array(obj)
        except ImportError:
            pass  # numpy not available
            
        # Handle other complex objects with __dict__
        if hasattr(obj, '__dict__'):
            return self._serialize_object_attributes(obj)
        
        # Handle objects with specific serialization methods
        if hasattr(obj, 'to_dict'):
            try:
                return obj.to_dict()
            except Exception:
                pass
        
        if hasattr(obj, '__json__'):
            try:
                return obj.__json__()
            except Exception:
                pass
        
        # Last resort: convert to string
        try:
            return str(obj)
        except Exception:
            return f"[UNSERIALIZABLE_{type(obj).__name__}]"
    
    def _serialize_vllm_request_output(self, obj):
        """Safely serialize vLLM RequestOutput objects"""
        try:
            return {
                "_type": "vLLM.RequestOutput",
                "request_id": getattr(obj, 'request_id', None),
                "outputs": self._serialize_completion_outputs(getattr(obj, 'outputs', [])),
                "finished": getattr(obj, 'finished', None),
                "metrics": getattr(obj, 'metrics', None),
                "prompt": getattr(obj, 'prompt', None),
                "prompt_token_ids": getattr(obj, 'prompt_token_ids', [])[:50] if hasattr(obj, 'prompt_token_ids') else None,  # Limit size
            }
        except Exception as e:
            logger.warning(f"Failed to serialize vLLM RequestOutput: {e}")
            return {
                "_type": "vLLM.RequestOutput",
                "_serialization_error": str(e),
                "request_id": getattr(obj, 'request_id', 'unknown')
            }
    
    def _serialize_completion_output(self, obj):
        """Safely serialize individual completion output"""
        try:
            return {
                "_type": "vLLM.CompletionOutput",
                "index": getattr(obj, 'index', None),
                "text": getattr(obj, 'text', None),
                "token_ids": getattr(obj, 'token_ids', [])[:100] if hasattr(obj, 'token_ids') else None,  # Limit size
                "cumulative_logprob": getattr(obj, 'cumulative_logprob', None),
                "logprobs": None,  # Skip complex logprob structures to avoid deep recursion
                "finish_reason": getattr(obj, 'finish_reason', None),
            }
        except Exception as e:
            logger.warning(f"Failed to serialize CompletionOutput: {e}")
            return {"_type": "vLLM.CompletionOutput", "_serialization_error": str(e)}
    
    def _serialize_completion_outputs(self, outputs):
        """Serialize list of completion output objects"""
        serialized = []
        for i, output in enumerate(outputs):
            if i >= 10:  # Limit number of outputs to prevent huge files
                serialized.append({"_note": f"... {len(outputs) - 10} more outputs truncated"})
                break
            try:
                serialized.append(self._serialize_completion_output(output))
            except Exception as e:
                logger.warning(f"Failed to serialize output {i}: {e}")
                serialized.append({"_serialization_error": str(e), "_index": i})
        return serialized
    
    def _serialize_torch_tensor(self, obj):
        """Safely serialize torch tensors"""
        try:
            tensor_info = {
                "_type": "torch.Tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "requires_grad": obj.requires_grad,
            }
            
            # Only include data for small tensors
            if obj.numel() < 100:
                tensor_info["data"] = obj.detach().cpu().numpy().tolist()
            else:
                tensor_info["data"] = f"[LARGE_TENSOR_{obj.numel()}_ELEMENTS]"
                
            return tensor_info
        except Exception as e:
            return {
                "_type": "torch.Tensor", 
                "_serialization_error": str(e),
                "shape": getattr(obj, 'shape', 'unknown')
            }
    
    def _serialize_numpy_array(self, obj):
        """Safely serialize numpy arrays"""
        try:
            array_info = {
                "_type": "numpy.ndarray",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
            
            # Only include data for small arrays
            if obj.size < 100:
                array_info["data"] = obj.tolist()
            else:
                array_info["data"] = f"[LARGE_ARRAY_{obj.size}_ELEMENTS]"
                
            return array_info
        except Exception as e:
            return {"_type": "numpy.ndarray", "_serialization_error": str(e)}
    
    def _serialize_object_attributes(self, obj):
        """Safely serialize object attributes"""
        result = {"_type": str(type(obj).__name__)}
        
        try:
            for key, value in obj.__dict__.items():
                if key.startswith('_'):  # Skip private attributes
                    continue
                    
                try:
                    # Test if value is JSON serializable
                    json.dumps(value, cls=MLObjectEncoder)
                    result[key] = value
                except (TypeError, ValueError, RecursionError):
                    # Handle non-serializable values
                    if hasattr(value, '__len__') and len(str(value)) > 1000:
                        result[key] = f"[LARGE_OBJECT_{type(value).__name__}]"
                    else:
                        result[key] = f"[NON_SERIALIZABLE_{type(value).__name__}]"
                except Exception as e:
                    result[key] = f"[ERROR_{str(e)}]"
                    
        except Exception as e:
            result["_serialization_error"] = str(e)
            
        return result

def safe_json_dump(data: Any, file_path: str, **kwargs) -> bool:
    """
    Safely dump data to JSON file with ML object support
    
    Args:
        data: Data to serialize
        file_path: Path to output file
        **kwargs: Additional arguments for json.dump
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=MLObjectEncoder, ensure_ascii=False, **kwargs)
        
        logger.debug(f"Successfully serialized data to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to serialize to {file_path}: {e}")
        
        # Try to save a minimal fallback version
        try:
            fallback_data = {
                "_serialization_fallback": True,
                "_original_error": str(e),
                "_data_type": str(type(data)),
                "_timestamp": datetime.now().isoformat()
            }
            
            # Try to extract some basic info
            if isinstance(data, dict):
                fallback_data["_keys"] = list(data.keys())[:10]  # First 10 keys
                
            fallback_path = str(file_path).replace('.json', '.fallback.json')
            with open(fallback_path, 'w') as f:
                json.dump(fallback_data, f, indent=2)
                
            logger.warning(f"Saved fallback data to {fallback_path}")
            
        except Exception as fallback_error:
            logger.error(f"Even fallback serialization failed: {fallback_error}")
            
        return False

def safe_json_dumps(data: Any, **kwargs) -> Optional[str]:
    """
    Safely serialize data to JSON string with ML object support
    
    Args:
        data: Data to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        str: JSON string if successful, None otherwise
    """
    try:
        return json.dumps(data, cls=MLObjectEncoder, ensure_ascii=False, **kwargs)
    except Exception as e:
        logger.error(f"Failed to serialize to JSON string: {e}")
        return None

def validate_serialization(data: Any) -> Dict[str, Any]:
    """
    Validate that data can be serialized without actually saving to file
    
    Args:
        data: Data to validate
        
    Returns:
        dict: Validation results with success status and details
    """
    result = {
        "success": False,
        "data_type": str(type(data)),
        "issues_found": [],
        "serialized_size": None
    }
    
    try:
        serialized = json.dumps(data, cls=MLObjectEncoder)
        result["success"] = True
        result["serialized_size"] = len(serialized)
        
        # Check for any warning patterns in serialized data
        if "_serialization_error" in serialized:
            result["issues_found"].append("Contains serialization errors")
        if "[NON_SERIALIZABLE_" in serialized:
            result["issues_found"].append("Contains non-serializable objects")
        if "[LARGE_" in serialized:
            result["issues_found"].append("Contains large objects (truncated)")
            
    except Exception as e:
        result["issues_found"].append(f"Serialization failed: {e}")
        
    return result