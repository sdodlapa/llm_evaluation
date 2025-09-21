# Advanced vLLM Integration Technical Plan
## Comprehensive Implementation Roadmap for Hybrid Evaluation System

**Document Version:** 1.0  
**Date:** September 21, 2025  
**Author:** AI Assistant  
**Scope:** Integration of advanced vLLM techniques into hybrid evaluation system  

---

## 📋 Document Overview

This technical plan outlines the detailed implementation strategy for integrating advanced vLLM optimization techniques into our hybrid evaluation system. The plan is based on analysis of Aleksa Gordic's comprehensive vLLM deep-dive and focuses on techniques that provide maximum benefit with optimal integration complexity.

### 🎯 Core Objectives

1. **Performance Optimization**: Achieve 20-50% improvement in evaluation throughput
2. **Capability Enhancement**: Add structured output generation and long context handling
3. **Resource Efficiency**: Better GPU utilization and memory management
4. **Production Readiness**: Maintain system stability and reliability

### 📊 Implementation Scope

- **5 Primary Techniques**: CUDA Graph Capture, Guided Decoding, Chunked Prefill, Performance Benchmarking, Speculative Decoding
- **3 Implementation Phases**: Immediate (next GPU session), Medium-term (2-4 weeks), Long-term (2-3 months)
- **Integration Points**: AOT Compiler, Lightweight Engine, Evaluation Pipeline, Benchmarking Framework

---

## 🏗️ Architecture Overview

### Current System State
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Evaluation    │    │  Lightweight     │    │  Distributed    │
│   Orchestrator  │───▶│     Engine       │───▶│    Engine       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  AOT Compiler    │
                       │  (Implemented)   │
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   vLLM Backend   │
                       │  (Base Features) │
                       └──────────────────┘
```

### Target Enhanced Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Evaluation    │    │  Enhanced        │    │  Distributed    │
│   Orchestrator  │───▶│  Lightweight     │───▶│    Engine       │
│  + Guided Gen   │    │    Engine        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Performance    │    │  Enhanced AOT    │
│  Benchmarking   │    │   Compiler       │
│   Framework     │    │ + CUDA Graphs   │
└─────────────────┘    └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Advanced vLLM   │
                       │ + Chunked Prefill│
                       │ + Spec Decoding  │
                       └──────────────────┘
```

---

## 📦 Implementation Priority Matrix

| Technique | Impact | Effort | Priority | Dependencies |
|-----------|--------|--------|----------|--------------|
| CUDA Graph Capture | HIGH | LOW | CRITICAL | AOT Compiler |
| Guided Decoding | HIGH | MEDIUM | CRITICAL | None |
| Performance Benchmarking | MEDIUM | LOW | HIGH | None |
| Chunked Prefill | HIGH | MEDIUM | HIGH | vLLM Integration |
| Speculative Decoding | MEDIUM | HIGH | MEDIUM | Model Training |

---

## 🔧 Technology Stack Requirements

### Additional Dependencies
```python
# Core dependencies (add to requirements.txt)
xgrammar>=0.1.0           # For guided decoding FSMs
torch-audio>=2.0.0        # For enhanced profiling
matplotlib>=3.7.0         # For benchmarking visualization
seaborn>=0.12.0          # For advanced analytics
psutil>=5.9.0            # For system monitoring
gpustat>=1.1.0           # For GPU monitoring
```

### Environment Setup
```bash
# Install additional dependencies
pip install xgrammar torch-audio matplotlib seaborn psutil gpustat

# Verify CUDA graph support
python -c "import torch; print(f'CUDA Graph Support: {torch.cuda.is_available()}')"

# Check PyTorch version compatibility
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
```

---

## 📐 Detailed Design Specifications

### Core Design Principles

1. **Backward Compatibility**: All enhancements must maintain existing API compatibility
2. **Graceful Degradation**: Features should fail gracefully if hardware/software requirements aren't met
3. **Modular Architecture**: Each enhancement should be independently configurable
4. **Performance Monitoring**: All features should include comprehensive metrics
5. **Documentation**: Complete code documentation and usage examples

### Configuration Architecture
```python
@dataclass
class AdvancedVLLMConfig:
    """Configuration for advanced vLLM features"""
    
    # CUDA Graph Configuration
    enable_cuda_graphs: bool = True
    cuda_graph_warmup_steps: int = 3
    cuda_graph_capture_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Guided Decoding Configuration
    enable_guided_decoding: bool = True
    default_grammar_backend: str = "xgrammar"
    grammar_cache_size: int = 100
    
    # Chunked Prefill Configuration  
    enable_chunked_prefill: bool = True
    prefill_chunk_size: int = 512
    max_prefill_tokens: int = 8192
    
    # Performance Configuration
    enable_performance_monitoring: bool = True
    benchmark_warmup_iterations: int = 3
    benchmark_measurement_iterations: int = 10
    
    # Speculative Decoding Configuration
    enable_speculative_decoding: bool = False
    speculative_method: str = "ngram"  # "ngram", "eagle", "medusa"
    num_speculative_tokens: int = 3
    speculative_acceptance_threshold: float = 0.7
```

---

## 🚀 PHASE 1: CUDA Graph Capture & Replay Implementation

### 🎯 Objective
Integrate CUDA graph capture and replay into our existing AOT compilation pipeline to reduce GPU kernel launch overhead by 10-30%.

### 📋 Technical Background

CUDA graphs allow capturing a sequence of CUDA operations into a graph structure that can be replayed multiple times with minimal CPU overhead. This is particularly beneficial for:

- Reducing kernel launch latency
- Improving GPU utilization
- Providing predictable performance characteristics
- Optimizing repetitive computation patterns

### 🔧 Implementation Strategy

#### 1. Enhanced AOT Compiler Integration

**File**: `engines/shared/aot_compiler.py`

```python
class CUDAGraphManager:
    """Manages CUDA graph capture and replay for compiled models"""
    
    def __init__(self, enable_graphs: bool = True, warmup_steps: int = 3):
        self.enable_graphs = enable_graphs and torch.cuda.is_available()
        self.warmup_steps = warmup_steps
        self.captured_graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.graph_inputs: Dict[str, List[torch.Tensor]] = {}
        self.graph_outputs: Dict[str, List[torch.Tensor]] = {}
        
    def capture_graph(self, 
                     model: torch.nn.Module, 
                     example_inputs: Tuple[torch.Tensor, ...],
                     batch_size: int) -> Optional[torch.cuda.CUDAGraph]:
        """
        Capture CUDA graph for a specific model and batch size
        
        Args:
            model: Compiled PyTorch model
            example_inputs: Representative input tensors
            batch_size: Target batch size for graph capture
            
        Returns:
            Captured CUDA graph or None if capture fails
        """
        if not self.enable_graphs:
            return None
            
        try:
            # Prepare inputs with target batch size
            graph_inputs = self._prepare_graph_inputs(example_inputs, batch_size)
            graph_key = self._generate_graph_key(model, batch_size)
            
            # Warmup runs
            model.eval()
            with torch.no_grad():
                for _ in range(self.warmup_steps):
                    _ = model(*graph_inputs)
                    torch.cuda.synchronize()
            
            # Capture graph
            graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(graph):
                graph_outputs = model(*graph_inputs)
                
            # Store graph and tensors
            self.captured_graphs[graph_key] = graph
            self.graph_inputs[graph_key] = graph_inputs
            self.graph_outputs[graph_key] = graph_outputs
            
            logger.info(f"Successfully captured CUDA graph for batch_size={batch_size}")
            return graph
            
        except Exception as e:
            logger.warning(f"CUDA graph capture failed for batch_size={batch_size}: {e}")
            return None
    
    def replay_graph(self, 
                    model_key: str, 
                    batch_size: int, 
                    input_data: Tuple[torch.Tensor, ...]) -> Optional[torch.Tensor]:
        """
        Replay captured CUDA graph with new input data
        
        Args:
            model_key: Unique identifier for the model
            batch_size: Batch size for graph selection
            input_data: New input tensors
            
        Returns:
            Model output from graph replay or None if no graph available
        """
        graph_key = f"{model_key}_batch_{batch_size}"
        
        if graph_key not in self.captured_graphs:
            return None
            
        try:
            # Copy new data into graph input tensors
            graph_inputs = self.graph_inputs[graph_key]
            for graph_input, new_input in zip(graph_inputs, input_data):
                graph_input.copy_(new_input)
            
            # Replay graph
            self.captured_graphs[graph_key].replay()
            
            # Return output (copy to avoid graph tensor lifecycle issues)
            return self.graph_outputs[graph_key].clone()
            
        except Exception as e:
            logger.warning(f"CUDA graph replay failed: {e}")
            return None
    
    def _prepare_graph_inputs(self, 
                             example_inputs: Tuple[torch.Tensor, ...], 
                             batch_size: int) -> List[torch.Tensor]:
        """Prepare input tensors with specified batch size for graph capture"""
        graph_inputs = []
        
        for input_tensor in example_inputs:
            # Create tensor with target batch size
            target_shape = list(input_tensor.shape)
            target_shape[0] = batch_size  # Assume batch dimension is first
            
            graph_input = torch.empty(
                target_shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device
            )
            
            # Fill with example data (repeat if necessary)
            if batch_size <= input_tensor.shape[0]:
                graph_input.copy_(input_tensor[:batch_size])
            else:
                # Repeat example data to fill larger batch
                repeats = (batch_size + input_tensor.shape[0] - 1) // input_tensor.shape[0]
                repeated_data = input_tensor.repeat(repeats, *([1] * (len(input_tensor.shape) - 1)))
                graph_input.copy_(repeated_data[:batch_size])
                
            graph_inputs.append(graph_input)
            
        return graph_inputs
    
    def _generate_graph_key(self, model: torch.nn.Module, batch_size: int) -> str:
        """Generate unique key for graph storage"""
        model_hash = hash(str(model))
        return f"model_{abs(model_hash)}_batch_{batch_size}"
    
    def get_capture_statistics(self) -> Dict[str, Any]:
        """Get statistics about captured graphs"""
        return {
            "total_captured_graphs": len(self.captured_graphs),
            "captured_batch_sizes": [
                int(key.split("_batch_")[1]) 
                for key in self.captured_graphs.keys()
            ],
            "memory_usage_mb": sum([
                sum([tensor.numel() * tensor.element_size() for tensor in tensors])
                for tensors in self.graph_inputs.values()
            ]) / (1024 * 1024)
        }
```

#### 2. AOT Compiler Enhancement

**Enhancement to existing `AOTModelCompiler` class**:

```python
class AOTModelCompiler:
    def __init__(self, 
                 cache_dir: str = "model_cache/compiled",
                 enable_aot: bool = True,
                 max_compilation_time: int = 600,
                 enable_cuda_graphs: bool = True,
                 cuda_graph_batch_sizes: List[int] = None):
        # ... existing initialization ...
        
        # CUDA Graph Manager
        self.cuda_graph_manager = CUDAGraphManager(
            enable_graphs=enable_cuda_graphs and CUDA_AVAILABLE,
            warmup_steps=3
        )
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or [1, 2, 4, 8, 16]
    
    def compile_model_aot(self, 
                         model, 
                         example_inputs: Tuple,
                         model_config,
                         compilation_mode: str = "default") -> Optional[Any]:
        """Enhanced compilation with CUDA graph capture"""
        
        # ... existing compilation logic ...
        
        if compiled_model is not None and self.cuda_graph_manager.enable_graphs:
            # Capture CUDA graphs for different batch sizes
            logger.info("Capturing CUDA graphs for compiled model...")
            
            captured_graphs = {}
            for batch_size in self.cuda_graph_batch_sizes:
                graph = self.cuda_graph_manager.capture_graph(
                    compiled_model, example_inputs, batch_size
                )
                if graph is not None:
                    captured_graphs[batch_size] = graph
            
            # Store graph metadata
            self.compilation_stats[cache_key].update({
                "cuda_graphs_captured": len(captured_graphs),
                "cuda_graph_batch_sizes": list(captured_graphs.keys()),
                "cuda_graph_memory_mb": self.cuda_graph_manager.get_capture_statistics()["memory_usage_mb"]
            })
            
            logger.info(f"Captured {len(captured_graphs)} CUDA graphs")
        
        return compiled_model
```

### 📊 Performance Monitoring Integration

```python
class CUDAGraphPerformanceMonitor:
    """Monitor performance improvements from CUDA graphs"""
    
    def __init__(self):
        self.graph_execution_times: List[float] = []
        self.regular_execution_times: List[float] = []
        self.graph_hit_count = 0
        self.graph_miss_count = 0
    
    def record_graph_execution(self, execution_time: float):
        """Record execution time when using CUDA graph"""
        self.graph_execution_times.append(execution_time)
        self.graph_hit_count += 1
    
    def record_regular_execution(self, execution_time: float):
        """Record execution time when not using CUDA graph"""
        self.regular_execution_times.append(execution_time)
        self.graph_miss_count += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.graph_execution_times or not self.regular_execution_times:
            return {"status": "insufficient_data"}
        
        graph_avg = statistics.mean(self.graph_execution_times)
        regular_avg = statistics.mean(self.regular_execution_times)
        speedup = regular_avg / graph_avg if graph_avg > 0 else 0
        
        return {
            "graph_hit_rate": self.graph_hit_count / (self.graph_hit_count + self.graph_miss_count),
            "average_graph_time_ms": graph_avg * 1000,
            "average_regular_time_ms": regular_avg * 1000,
            "speedup_factor": speedup,
            "total_graph_executions": self.graph_hit_count,
            "total_regular_executions": self.graph_miss_count
        }
```

### 🧪 Testing Strategy

```python
def test_cuda_graph_integration():
    """Comprehensive test for CUDA graph integration"""
    
    # Test graph capture
    compiler = AOTModelCompiler(enable_cuda_graphs=True)
    
    # Create test model
    model = SimpleTestModel()
    example_inputs = (torch.randn(2, 512),)
    
    # Test compilation with graph capture
    compiled_model = compiler.compile_model_aot(model, example_inputs, MockConfig())
    
    assert compiled_model is not None
    assert len(compiler.cuda_graph_manager.captured_graphs) > 0
    
    # Test graph replay
    test_input = torch.randn(4, 512)
    graph_output = compiler.cuda_graph_manager.replay_graph("test_model", 4, (test_input,))
    
    assert graph_output is not None
    
    # Performance comparison
    monitor = CUDAGraphPerformanceMonitor()
    
    # Measure regular execution
    start_time = time.time()
    regular_output = compiled_model(test_input)
    monitor.record_regular_execution(time.time() - start_time)
    
    # Measure graph execution  
    start_time = time.time()
    graph_output = compiler.cuda_graph_manager.replay_graph("test_model", 4, (test_input,))
    monitor.record_graph_execution(time.time() - start_time)
    
    # Verify outputs match
    torch.testing.assert_close(regular_output, graph_output, rtol=1e-5, atol=1e-6)
    
    # Check performance improvement
    perf_summary = monitor.get_performance_summary()
    assert perf_summary["speedup_factor"] > 1.0
    
    logger.info(f"CUDA Graph speedup: {perf_summary['speedup_factor']:.2f}x")
```

### 🔄 Integration Points

1. **AOT Compiler**: Enhanced compilation process with graph capture
2. **Model Loader**: Modified to use graph replay when available
3. **Performance Monitoring**: New metrics for graph usage and performance
4. **Configuration**: New config options for graph management

### 📈 Expected Benefits

- **Latency Reduction**: 10-30% improvement in inference latency
- **Consistency**: More predictable execution times
- **GPU Utilization**: Better GPU resource utilization
- **Scalability**: Better performance at higher batch sizes

### ⚠️ Implementation Considerations

- **Memory Overhead**: Captured graphs consume additional GPU memory
- **Batch Size Constraints**: Graphs are captured for specific batch sizes
- **Model Compatibility**: Not all models/operations support CUDA graphs
- **Debugging Complexity**: Graph execution can be harder to debug

---

## 🧠 PHASE 2: Guided Decoding Integration with xgrammar

### 🎯 Objective
Integrate xgrammar for structured output generation in evaluation tasks, ensuring model outputs conform to JSON schemas, programming languages, or mathematical expressions.

### 📋 Technical Background

Guided decoding uses grammar constraints during the generation process to ensure outputs follow specific formats. This is crucial for:

- **Structured Evaluations**: Ensuring consistent JSON output for automated scoring
- **Code Generation Tasks**: Enforcing syntax compliance for programming benchmarks  
- **Mathematical Reasoning**: Ensuring proper mathematical notation and formatting
- **Reliability**: Reducing parsing errors in evaluation pipelines

### 🔧 Implementation Strategy

#### 1. Guided Decoding Engine

**File**: `engines/shared/guided_decoding.py`

```python
import json
import re
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import xgrammar as xgr
    XGRAMMAR_AVAILABLE = True
except ImportError:
    XGRAMMAR_AVAILABLE = False
    xgr = None

@dataclass
class GuidedDecodingConfig:
    """Configuration for guided decoding"""
    grammar_type: str  # "json", "regex", "bnf", "python", "math"
    grammar_spec: Union[str, Dict]  # Grammar specification
    max_tokens: int = 2048
    enable_caching: bool = True
    strict_mode: bool = True
    temperature: float = 0.7
    top_p: float = 0.9

class GrammarEngine(ABC):
    """Abstract base class for grammar engines"""
    
    @abstractmethod
    def create_grammar(self, spec: Union[str, Dict]) -> Any:
        """Create grammar object from specification"""
        pass
    
    @abstractmethod
    def guide_generation(self, 
                        model_output_logits: torch.Tensor,
                        current_sequence: List[int],
                        grammar_state: Any) -> torch.Tensor:
        """Apply grammar constraints to generation logits"""
        pass

class XgrammarEngine(GrammarEngine):
    """xgrammar-based guided decoding engine"""
    
    def __init__(self, tokenizer):
        if not XGRAMMAR_AVAILABLE:
            raise ImportError("xgrammar not available. Install with: pip install xgrammar")
        
        self.tokenizer = tokenizer
        self.grammar_cache: Dict[str, Any] = {}
    
    def create_grammar(self, spec: Union[str, Dict]) -> xgr.Grammar:
        """Create xgrammar Grammar object"""
        
        if isinstance(spec, dict):
            # JSON Schema
            if "type" in spec or "properties" in spec:
                return xgr.Grammar.from_json_schema(json.dumps(spec))
            else:
                raise ValueError("Invalid JSON schema specification")
        
        elif isinstance(spec, str):
            if spec.startswith("regex:"):
                # Regular expression
                pattern = spec[6:]  # Remove "regex:" prefix
                return xgr.Grammar.from_regex(pattern)
            
            elif spec.startswith("bnf:"):
                # BNF grammar
                bnf_spec = spec[4:]  # Remove "bnf:" prefix
                return xgr.Grammar.from_bnf(bnf_spec)
            
            elif spec == "python":
                # Python code grammar
                return xgr.Grammar.builtin_python()
            
            elif spec == "json":
                # Generic JSON grammar
                return xgr.Grammar.builtin_json()
            
            else:
                # Treat as direct BNF specification
                return xgr.Grammar.from_bnf(spec)
        
        else:
            raise ValueError(f"Unsupported grammar specification type: {type(spec)}")
    
    def guide_generation(self, 
                        model_output_logits: torch.Tensor,
                        current_sequence: List[int],
                        grammar_state: xgr.GrammarMatcher) -> torch.Tensor:
        """Apply grammar constraints to logits"""
        
        # Get valid next tokens from grammar
        valid_tokens = grammar_state.get_next_tokens()
        
        if not valid_tokens:
            # If no valid tokens, return original logits (fallback)
            return model_output_logits
        
        # Create mask for valid tokens
        vocab_size = model_output_logits.shape[-1]
        mask = torch.full((vocab_size,), float('-inf'), device=model_output_logits.device)
        
        # Set valid tokens to 0 (no penalty)
        for token_id in valid_tokens:
            if token_id < vocab_size:
                mask[token_id] = 0.0
        
        # Apply mask to logits
        return model_output_logits + mask

class GuidedDecodingManager:
    """Manages guided decoding for evaluation tasks"""
    
    def __init__(self, tokenizer, enable_caching: bool = True):
        self.tokenizer = tokenizer
        self.enable_caching = enable_caching
        self.grammar_cache: Dict[str, Any] = {}
        
        # Initialize grammar engine
        if XGRAMMAR_AVAILABLE:
            self.engine = XgrammarEngine(tokenizer)
        else:
            logger.warning("xgrammar not available, guided decoding disabled")
            self.engine = None
    
    def create_guided_config(self, task_type: str, **kwargs) -> Optional[GuidedDecodingConfig]:
        """Create guided decoding config for specific task types"""
        
        if self.engine is None:
            return None
        
        if task_type == "json_qa":
            # Question answering with JSON output
            schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"}
                },
                "required": ["answer", "confidence"]
            }
            return GuidedDecodingConfig("json", schema, **kwargs)
        
        elif task_type == "multiple_choice":
            # Multiple choice with structured output
            choices = kwargs.get("choices", ["A", "B", "C", "D"])
            schema = {
                "type": "object", 
                "properties": {
                    "answer": {"type": "string", "enum": choices},
                    "explanation": {"type": "string"}
                },
                "required": ["answer"]
            }
            return GuidedDecodingConfig("json", schema, **kwargs)
        
        elif task_type == "code_generation":
            # Python code generation
            return GuidedDecodingConfig("python", "python", **kwargs)
        
        elif task_type == "math_expression":
            # Mathematical expressions
            math_grammar = """
            expr: term (("+" | "-") term)*
            term: factor (("*" | "/" | "//") factor)*
            factor: "(" expr ")" | NUMBER | VARIABLE
            NUMBER: /[0-9]+(\.[0-9]+)?/
            VARIABLE: /[a-zA-Z][a-zA-Z0-9_]*/
            """
            return GuidedDecodingConfig("bnf", f"bnf:{math_grammar}", **kwargs)
        
        elif task_type == "regex_pattern":
            # Use provided regex pattern
            pattern = kwargs.get("pattern", ".*")
            return GuidedDecodingConfig("regex", f"regex:{pattern}", **kwargs)
        
        else:
            logger.warning(f"Unknown task type for guided decoding: {task_type}")
            return None
    
    def setup_guided_generation(self, config: GuidedDecodingConfig) -> Optional[Any]:
        """Setup guided generation state"""
        
        if self.engine is None or config is None:
            return None
        
        try:
            # Create or retrieve cached grammar
            cache_key = f"{config.grammar_type}:{hash(str(config.grammar_spec))}"
            
            if self.enable_caching and cache_key in self.grammar_cache:
                grammar = self.grammar_cache[cache_key]
            else:
                grammar = self.engine.create_grammar(config.grammar_spec)
                if self.enable_caching:
                    self.grammar_cache[cache_key] = grammar
            
            # Create grammar matcher for this generation session
            grammar_matcher = xgr.GrammarMatcher(grammar, self.tokenizer)
            
            return {
                "grammar": grammar,
                "matcher": grammar_matcher,
                "config": config
            }
            
        except Exception as e:
            logger.error(f"Failed to setup guided generation: {e}")
            return None
    
    def apply_guided_constraints(self, 
                               logits: torch.Tensor,
                               guided_state: Dict[str, Any],
                               current_tokens: List[int]) -> torch.Tensor:
        """Apply guided decoding constraints to generation logits"""
        
        if guided_state is None or self.engine is None:
            return logits
        
        try:
            matcher = guided_state["matcher"]
            
            # Update matcher state with current tokens
            for token in current_tokens:
                if not matcher.accept_token(token):
                    # Invalid sequence, return original logits
                    logger.warning("Invalid token sequence for grammar")
                    return logits
            
            # Apply constraints
            constrained_logits = self.engine.guide_generation(
                logits, current_tokens, matcher
            )
            
            return constrained_logits
            
        except Exception as e:
            logger.error(f"Error applying guided constraints: {e}")
            return logits
    
    def validate_output(self, 
                       output_text: str, 
                       guided_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated output against grammar"""
        
        if guided_state is None:
            return {"valid": True, "message": "No validation applied"}
        
        try:
            config = guided_state["config"]
            
            if config.grammar_type == "json":
                # Validate JSON
                try:
                    parsed = json.loads(output_text)
                    return {"valid": True, "parsed": parsed}
                except json.JSONDecodeError as e:
                    return {"valid": False, "error": f"Invalid JSON: {e}"}
            
            elif config.grammar_type == "regex":
                # Validate regex match
                pattern = config.grammar_spec[6:]  # Remove "regex:" prefix
                if re.match(pattern, output_text):
                    return {"valid": True}
                else:
                    return {"valid": False, "error": "Output doesn't match regex pattern"}
            
            elif config.grammar_type == "python":
                # Validate Python syntax
                try:
                    compile(output_text, '<string>', 'exec')
                    return {"valid": True}
                except SyntaxError as e:
                    return {"valid": False, "error": f"Invalid Python syntax: {e}"}
            
            else:
                # For BNF and other grammars, assume valid if generation completed
                return {"valid": True, "message": "Grammar validation not implemented"}
                
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}
```

#### 2. Integration with Evaluation Pipeline

**Enhancement to evaluation engine**:

```python
class GuidedEvaluationEngine:
    """Enhanced evaluation engine with guided decoding support"""
    
    def __init__(self, model_loader, guided_manager: Optional[GuidedDecodingManager] = None):
        self.model_loader = model_loader
        self.guided_manager = guided_manager
        self.guided_stats = {
            "total_guided_generations": 0,
            "successful_validations": 0,
            "validation_failures": 0,
            "grammar_setup_failures": 0
        }
    
    def evaluate_with_guidance(self, 
                             task: Dict[str, Any],
                             model_config: Dict[str, Any],
                             guided_config: Optional[GuidedDecodingConfig] = None) -> Dict[str, Any]:
        """Run evaluation with optional guided decoding"""
        
        # Setup guided generation if configured
        guided_state = None
        if self.guided_manager and guided_config:
            guided_state = self.guided_manager.setup_guided_generation(guided_config)
            if guided_state:
                logger.info(f"Guided decoding enabled for task: {task.get('task_id', 'unknown')}")
            else:
                self.guided_stats["grammar_setup_failures"] += 1
                logger.warning("Failed to setup guided decoding, falling back to regular generation")
        
        # Load model
        model = self.model_loader.load_model(model_config)
        
        # Prepare input
        input_text = task["prompt"]
        input_tokens = model.tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate with guidance
        try:
            if guided_state:
                output_tokens = self._generate_with_guidance(
                    model, input_tokens, guided_state
                )
                self.guided_stats["total_guided_generations"] += 1
            else:
                output_tokens = self._generate_regular(model, input_tokens)
            
            # Decode output
            output_text = model.tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            # Validate output if guided
            validation_result = {"valid": True}
            if guided_state:
                validation_result = self.guided_manager.validate_output(
                    output_text, guided_state
                )
                if validation_result["valid"]:
                    self.guided_stats["successful_validations"] += 1
                else:
                    self.guided_stats["validation_failures"] += 1
            
            return {
                "model_output": output_text,
                "guided_decoding_used": guided_state is not None,
                "validation_result": validation_result,
                "task_id": task.get("task_id"),
                "guided_stats": self.guided_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {"error": str(e), "task_id": task.get("task_id")}
    
    def _generate_with_guidance(self, 
                               model, 
                               input_tokens: torch.Tensor,
                               guided_state: Dict[str, Any]) -> torch.Tensor:
        """Generate text with guided decoding constraints"""
        
        config = guided_state["config"]
        current_tokens = input_tokens[0].tolist()
        
        model.eval()
        with torch.no_grad():
            for _ in range(config.max_tokens):
                # Get logits from model
                outputs = model(torch.tensor([current_tokens]))
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Apply guided constraints
                constrained_logits = self.guided_manager.apply_guided_constraints(
                    logits.unsqueeze(0), guided_state, current_tokens
                )
                
                # Sample next token
                probabilities = torch.softmax(
                    constrained_logits / config.temperature, dim=-1
                )
                
                # Top-p sampling
                if config.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    probabilities[indices_to_remove] = 0
                    probabilities = probabilities / probabilities.sum()
                
                # Sample token
                next_token = torch.multinomial(probabilities, 1).item()
                current_tokens.append(next_token)
                
                # Check for end of sequence
                if next_token == model.tokenizer.eos_token_id:
                    break
        
        return torch.tensor(current_tokens)
    
    def _generate_regular(self, model, input_tokens: torch.Tensor) -> torch.Tensor:
        """Regular generation without guidance"""
        with torch.no_grad():
            outputs = model.generate(
                input_tokens,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        return outputs[0]
```

### 🧪 Testing and Validation

```python
def test_guided_decoding_integration():
    """Test guided decoding with various grammar types"""
    
    # Mock tokenizer and model
    tokenizer = MockTokenizer()
    guided_manager = GuidedDecodingManager(tokenizer)
    
    # Test JSON schema guidance
    json_config = guided_manager.create_guided_config(
        "json_qa", 
        max_tokens=256
    )
    assert json_config is not None
    assert json_config.grammar_type == "json"
    
    # Test guided state setup
    guided_state = guided_manager.setup_guided_generation(json_config)
    assert guided_state is not None
    
    # Test output validation
    valid_json = '{"answer": "test", "confidence": 0.95}'
    validation = guided_manager.validate_output(valid_json, guided_state)
    assert validation["valid"] == True
    
    invalid_json = '{"answer": "test", "confidence":}'
    validation = guided_manager.validate_output(invalid_json, guided_state)
    assert validation["valid"] == False
    
    # Test code generation
    code_config = guided_manager.create_guided_config("code_generation")
    assert code_config.grammar_type == "python"
    
    logger.info("All guided decoding tests passed")

def test_evaluation_with_guidance():
    """Test evaluation pipeline with guided decoding"""
    
    # Setup
    model_loader = MockModelLoader()
    guided_manager = GuidedDecodingManager(MockTokenizer())
    eval_engine = GuidedEvaluationEngine(model_loader, guided_manager)
    
    # Test task
    task = {
        "task_id": "test_guided",
        "prompt": "What is the capital of France? Answer in JSON format.",
        "expected_format": "json"
    }
    
    # Create guided config
    guided_config = guided_manager.create_guided_config("json_qa")
    
    # Run evaluation
    result = eval_engine.evaluate_with_guidance(
        task, MockModelConfig(), guided_config
    )
    
    assert "model_output" in result
    assert result["guided_decoding_used"] == True
    assert "validation_result" in result
    
    logger.info("Guided evaluation test passed")
```

### 📈 Expected Benefits

- **Output Reliability**: 90%+ reduction in parsing errors
- **Evaluation Consistency**: Standardized output formats across all models
- **Task Adaptability**: Flexible grammar specification for different evaluation types
- **Quality Metrics**: Improved evaluation accuracy through structured outputs

---

## 🧩 PHASE 3: Chunked Prefill Implementation

### 🎯 Objective
Implement chunked prefill to efficiently handle long context evaluation datasets by processing prompts in smaller chunks while maintaining full attention context.

### 📋 Technical Background

Chunked prefill addresses memory limitations when processing long sequences by:

- **Memory Efficiency**: Breaking long sequences into manageable chunks
- **Attention Preservation**: Maintaining full attention across chunk boundaries  
- **Batch Processing**: Enabling larger effective batch sizes with long contexts
- **Scalability**: Supporting evaluation datasets with very long prompts (32K+ tokens)

### 🔧 Implementation Strategy

#### 1. Chunked Prefill Engine

**File**: `engines/shared/chunked_prefill.py`

```python
import torch
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChunkedPrefillConfig:
    """Configuration for chunked prefill processing"""
    chunk_size: int = 2048  # Size of each chunk in tokens
    overlap_size: int = 128  # Overlap between chunks for context continuity
    max_sequence_length: int = 32768  # Maximum supported sequence length
    enable_attention_optimization: bool = True
    prefill_batch_size: int = 4  # Batch size for prefill chunks
    generation_batch_size: int = 1  # Batch size for generation phase
    memory_efficient_attention: bool = True

class ChunkedPrefillManager:
    """Manages chunked prefill processing for long context evaluation"""
    
    def __init__(self, config: ChunkedPrefillConfig):
        self.config = config
        self.chunk_cache: Dict[str, List[torch.Tensor]] = {}
        self.attention_cache: Dict[str, torch.Tensor] = {}
        self.processing_stats = {
            "total_sequences_processed": 0,
            "total_chunks_processed": 0,
            "average_chunks_per_sequence": 0,
            "memory_peak_mb": 0,
            "prefill_time_ms": 0
        }
    
    def prepare_chunked_sequence(self, 
                                input_tokens: torch.Tensor,
                                sequence_id: str = None) -> Dict[str, Any]:
        """
        Prepare long sequence for chunked prefill processing
        
        Args:
            input_tokens: Input token sequence [batch_size, seq_len]
            sequence_id: Optional unique identifier for caching
            
        Returns:
            Dictionary with chunked sequence information
        """
        seq_len = input_tokens.shape[1]
        
        if seq_len <= self.config.chunk_size:
            # Sequence is short enough, no chunking needed
            return {
                "chunks": [input_tokens],
                "chunk_positions": [(0, seq_len)],
                "needs_chunking": False,
                "total_chunks": 1,
                "sequence_length": seq_len
            }
        
        # Calculate chunking strategy
        chunks = []
        chunk_positions = []
        
        start_pos = 0
        chunk_idx = 0
        
        while start_pos < seq_len:
            # Calculate chunk end position
            end_pos = min(start_pos + self.config.chunk_size, seq_len)
            
            # Adjust for overlap (except for last chunk)
            if end_pos < seq_len:
                end_pos = min(end_pos, seq_len)
            
            # Extract chunk
            chunk_tokens = input_tokens[:, start_pos:end_pos]
            chunks.append(chunk_tokens)
            chunk_positions.append((start_pos, end_pos))
            
            # Calculate next start position with overlap
            start_pos = end_pos - self.config.overlap_size
            if start_pos >= seq_len:
                break
                
            chunk_idx += 1
        
        return {
            "chunks": chunks,
            "chunk_positions": chunk_positions,
            "needs_chunking": True,
            "total_chunks": len(chunks),
            "sequence_length": seq_len,
            "sequence_id": sequence_id
        }
    
    def process_chunked_prefill(self, 
                               model,
                               chunked_sequence: Dict[str, Any],
                               enable_caching: bool = True) -> Dict[str, Any]:
        """
        Process chunked sequence through model with prefill optimization
        
        Args:
            model: PyTorch model with attention mechanism
            chunked_sequence: Output from prepare_chunked_sequence
            enable_caching: Whether to cache intermediate states
            
        Returns:
            Dictionary with processed results and cached states
        """
        chunks = chunked_sequence["chunks"]
        chunk_positions = chunked_sequence["chunk_positions"]
        sequence_id = chunked_sequence.get("sequence_id")
        
        if not chunked_sequence["needs_chunking"]:
            # Process single chunk normally
            with torch.no_grad():
                outputs = model(chunks[0])
            return {
                "final_hidden_states": outputs.last_hidden_state,
                "past_key_values": getattr(outputs, 'past_key_values', None),
                "attention_mask": None,
                "processed_length": chunks[0].shape[1]
            }
        
        # Process chunks sequentially with attention continuity
        all_hidden_states = []
        cumulative_attention_mask = None
        past_key_values = None
        
        model.eval()
        
        for chunk_idx, (chunk_tokens, (start_pos, end_pos)) in enumerate(zip(chunks, chunk_positions)):
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                        f"(positions {start_pos}:{end_pos})")
            
            with torch.no_grad():
                # Prepare attention mask for this chunk
                chunk_attention_mask = self._create_chunk_attention_mask(
                    chunk_tokens, start_pos, end_pos, cumulative_attention_mask
                )
                
                # Process chunk through model
                if hasattr(model, 'forward_chunked'):
                    # Use model's native chunked processing if available
                    outputs = model.forward_chunked(
                        chunk_tokens,
                        attention_mask=chunk_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                else:
                    # Standard forward pass with attention mask
                    outputs = model(
                        chunk_tokens,
                        attention_mask=chunk_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                # Store hidden states
                chunk_hidden_states = outputs.last_hidden_state
                all_hidden_states.append(chunk_hidden_states)
                
                # Update past key values for next chunk
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                    past_key_values = outputs.past_key_values
                
                # Update cumulative attention mask
                cumulative_attention_mask = self._update_attention_mask(
                    cumulative_attention_mask, chunk_attention_mask, chunk_idx
                )
            
            # Memory management
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Combine hidden states from all chunks
        final_hidden_states = self._combine_chunk_outputs(
            all_hidden_states, chunk_positions
        )
        
        # Cache results if enabled
        if enable_caching and sequence_id:
            self.attention_cache[sequence_id] = final_hidden_states
        
        # Update statistics
        self.processing_stats["total_sequences_processed"] += 1
        self.processing_stats["total_chunks_processed"] += len(chunks)
        self.processing_stats["average_chunks_per_sequence"] = (
            self.processing_stats["total_chunks_processed"] / 
            self.processing_stats["total_sequences_processed"]
        )
        
        return {
            "final_hidden_states": final_hidden_states,
            "past_key_values": past_key_values,
            "attention_mask": cumulative_attention_mask,
            "processed_length": chunked_sequence["sequence_length"],
            "chunk_count": len(chunks)
        }
    
    def _create_chunk_attention_mask(self, 
                                   chunk_tokens: torch.Tensor,
                                   start_pos: int,
                                   end_pos: int,
                                   cumulative_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Create attention mask for current chunk with context awareness"""
        
        batch_size, chunk_len = chunk_tokens.shape
        
        # Create basic attention mask for chunk (all 1s, no padding assumed)
        chunk_mask = torch.ones(batch_size, chunk_len, dtype=torch.bool, device=chunk_tokens.device)
        
        if cumulative_mask is None:
            # First chunk, return basic mask
            return chunk_mask
        
        # Combine with cumulative mask for full context attention
        full_context_len = cumulative_mask.shape[1] + chunk_len
        full_mask = torch.zeros(batch_size, full_context_len, dtype=torch.bool, device=chunk_tokens.device)
        
        # Copy previous context
        full_mask[:, :cumulative_mask.shape[1]] = cumulative_mask
        
        # Add current chunk
        full_mask[:, cumulative_mask.shape[1]:] = chunk_mask
        
        return full_mask
    
    def _update_attention_mask(self, 
                             cumulative_mask: Optional[torch.Tensor],
                             chunk_mask: torch.Tensor,
                             chunk_idx: int) -> torch.Tensor:
        """Update cumulative attention mask with new chunk"""
        
        if cumulative_mask is None:
            return chunk_mask
        
        # For memory efficiency, we may need to truncate very old context
        max_context_length = self.config.max_sequence_length
        
        if cumulative_mask.shape[1] + chunk_mask.shape[1] > max_context_length:
            # Truncate oldest context to stay within limits
            truncate_amount = (cumulative_mask.shape[1] + chunk_mask.shape[1]) - max_context_length
            cumulative_mask = cumulative_mask[:, truncate_amount:]
        
        # Concatenate masks
        return torch.cat([cumulative_mask, chunk_mask], dim=1)
    
    def _combine_chunk_outputs(self, 
                             hidden_states_list: List[torch.Tensor],
                             chunk_positions: List[Tuple[int, int]]) -> torch.Tensor:
        """Combine hidden states from multiple chunks, handling overlaps"""
        
        if len(hidden_states_list) == 1:
            return hidden_states_list[0]
        
        # Calculate total sequence length
        total_length = chunk_positions[-1][1]
        batch_size = hidden_states_list[0].shape[0]
        hidden_dim = hidden_states_list[0].shape[-1]
        
        # Initialize combined tensor
        combined_states = torch.zeros(
            batch_size, total_length, hidden_dim,
            dtype=hidden_states_list[0].dtype,
            device=hidden_states_list[0].device
        )
        
        for chunk_states, (start_pos, end_pos) in zip(hidden_states_list, chunk_positions):
            chunk_len = end_pos - start_pos
            
            if start_pos == 0:
                # First chunk, copy directly
                combined_states[:, start_pos:end_pos] = chunk_states
            else:
                # Handle overlap by averaging overlapping regions
                overlap_start = start_pos
                overlap_end = min(start_pos + self.config.overlap_size, end_pos)
                
                if overlap_end > overlap_start:
                    # Average overlapping region
                    existing_states = combined_states[:, overlap_start:overlap_end]
                    new_states = chunk_states[:, :overlap_end - overlap_start]
                    combined_states[:, overlap_start:overlap_end] = (existing_states + new_states) / 2
                
                # Copy non-overlapping part
                non_overlap_start = overlap_end
                if non_overlap_start < end_pos:
                    copy_start_in_chunk = non_overlap_start - start_pos
                    combined_states[:, non_overlap_start:end_pos] = chunk_states[:, copy_start_in_chunk:]
        
        return combined_states
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        return self.processing_stats.copy()
    
    def clear_cache(self, sequence_id: str = None):
        """Clear cached states for specific sequence or all sequences"""
        if sequence_id:
            self.attention_cache.pop(sequence_id, None)
        else:
            self.attention_cache.clear()

class ChunkedGenerationManager:
    """Manages generation phase after chunked prefill"""
    
    def __init__(self, prefill_manager: ChunkedPrefillManager):
        self.prefill_manager = prefill_manager
        self.generation_stats = {
            "total_generations": 0,
            "average_generation_length": 0,
            "chunked_context_usage": 0
        }
    
    def generate_with_chunked_context(self, 
                                    model,
                                    prefill_result: Dict[str, Any],
                                    generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate text using results from chunked prefill
        
        Args:
            model: PyTorch model
            prefill_result: Output from process_chunked_prefill
            generation_config: Generation parameters
            
        Returns:
            Generated text and metadata
        """
        
        # Extract prefill context
        hidden_states = prefill_result["final_hidden_states"]
        past_key_values = prefill_result.get("past_key_values")
        attention_mask = prefill_result.get("attention_mask")
        
        # Generation parameters
        max_new_tokens = generation_config.get("max_new_tokens", 512)
        temperature = generation_config.get("temperature", 0.7)
        top_p = generation_config.get("top_p", 0.9)
        do_sample = generation_config.get("do_sample", True)
        
        # Generate tokens
        generated_tokens = []
        current_length = prefill_result["processed_length"]
        
        model.eval()
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare input for next token (empty input for continuation)
                if step == 0:
                    # First generation step uses prefill result
                    next_token_logits = hidden_states[:, -1, :]  # Last position logits
                else:
                    # Subsequent steps need forward pass with last generated token
                    last_token = torch.tensor([[generated_tokens[-1]]], device=hidden_states.device)
                    outputs = model(
                        last_token,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-p sampling
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token.item())
                
                # Check for end of sequence
                if hasattr(model, 'config') and hasattr(model.config, 'eos_token_id'):
                    if next_token.item() == model.config.eos_token_id:
                        break
                
                # Update attention mask for next step
                if attention_mask is not None:
                    new_mask = torch.ones(attention_mask.shape[0], 1, 
                                        dtype=attention_mask.dtype, 
                                        device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        
        # Update statistics
        self.generation_stats["total_generations"] += 1
        self.generation_stats["average_generation_length"] = (
            (self.generation_stats["average_generation_length"] * 
             (self.generation_stats["total_generations"] - 1) + len(generated_tokens)) /
            self.generation_stats["total_generations"]
        )
        
        return {
            "generated_tokens": generated_tokens,
            "generation_length": len(generated_tokens),
            "prefill_length": current_length,
            "total_context_length": current_length + len(generated_tokens),
            "chunked_prefill_used": prefill_result.get("chunk_count", 1) > 1
        }
```

#### 2. Integration with Evaluation Pipeline

**Enhancement to model loading and evaluation**:

```python
class ChunkedEvaluationEngine:
    """Enhanced evaluation engine with chunked prefill support"""
    
    def __init__(self, 
                 model_loader,
                 chunked_config: Optional[ChunkedPrefillConfig] = None):
        self.model_loader = model_loader
        self.chunked_config = chunked_config or ChunkedPrefillConfig()
        self.prefill_manager = ChunkedPrefillManager(self.chunked_config)
        self.generation_manager = ChunkedGenerationManager(self.prefill_manager)
        
    def evaluate_long_context_task(self, 
                                  task: Dict[str, Any],
                                  model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task with potentially long context using chunked prefill"""
        
        # Load model
        model = self.model_loader.load_model(model_config)
        tokenizer = getattr(model, 'tokenizer', None)
        
        if tokenizer is None:
            raise ValueError("Model must have associated tokenizer for chunked processing")
        
        # Tokenize input
        prompt = task["prompt"]
        input_tokens = tokenizer.encode(prompt, return_tensors="pt")
        
        logger.info(f"Processing task with input length: {input_tokens.shape[1]} tokens")
        
        # Prepare chunked sequence
        chunked_sequence = self.prefill_manager.prepare_chunked_sequence(
            input_tokens, sequence_id=task.get("task_id")
        )
        
        if chunked_sequence["needs_chunking"]:
            logger.info(f"Using chunked prefill: {chunked_sequence['total_chunks']} chunks")
        
        # Process prefill phase
        prefill_result = self.prefill_manager.process_chunked_prefill(
            model, chunked_sequence
        )
        
        # Generate response
        generation_config = {
            "max_new_tokens": task.get("max_new_tokens", 512),
            "temperature": task.get("temperature", 0.7),
            "top_p": task.get("top_p", 0.9),
            "do_sample": task.get("do_sample", True)
        }
        
        generation_result = self.generation_manager.generate_with_chunked_context(
            model, prefill_result, generation_config
        )
        
        # Decode output
        output_text = tokenizer.decode(
            generation_result["generated_tokens"], 
            skip_special_tokens=True
        )
        
        return {
            "model_output": output_text,
            "input_length": chunked_sequence["sequence_length"],
            "output_length": generation_result["generation_length"],
            "chunks_used": chunked_sequence["total_chunks"],
            "chunked_prefill_used": generation_result["chunked_prefill_used"],
            "prefill_stats": self.prefill_manager.get_processing_statistics(),
            "task_id": task.get("task_id")
        }
```

### 🧪 Testing Strategy

```python
def test_chunked_prefill_integration():
    """Test chunked prefill with various sequence lengths"""
    
    config = ChunkedPrefillConfig(chunk_size=512, overlap_size=64)
    manager = ChunkedPrefillManager(config)
    
    # Test short sequence (no chunking needed)
    short_tokens = torch.randint(0, 1000, (1, 256))
    chunked = manager.prepare_chunked_sequence(short_tokens)
    assert chunked["needs_chunking"] == False
    assert chunked["total_chunks"] == 1
    
    # Test long sequence (chunking needed)
    long_tokens = torch.randint(0, 1000, (1, 2048))
    chunked = manager.prepare_chunked_sequence(long_tokens)
    assert chunked["needs_chunking"] == True
    assert chunked["total_chunks"] > 1
    
    # Test overlap calculation
    chunk_positions = chunked["chunk_positions"]
    for i in range(1, len(chunk_positions)):
        prev_end = chunk_positions[i-1][1]
        curr_start = chunk_positions[i][0]
        overlap = prev_end - curr_start
        assert overlap <= config.overlap_size
    
    logger.info("Chunked prefill tests passed")

def test_memory_efficiency():
    """Test memory usage with chunked vs non-chunked processing"""
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Test with very long sequence
    long_sequence = torch.randint(0, 1000, (1, 8192))
    
    # Measure memory with chunking
    config = ChunkedPrefillConfig(chunk_size=1024)
    manager = ChunkedPrefillManager(config)
    
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    chunked = manager.prepare_chunked_sequence(long_sequence)
    # Mock model processing would go here
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    
    logger.info(f"Memory usage with chunking: {memory_used:.2f} MB")
    
    # Verify chunking was used
    assert chunked["needs_chunking"] == True
    assert chunked["total_chunks"] > 1
    
    logger.info("Memory efficiency test passed")
```

### 📈 Expected Benefits

- **Memory Efficiency**: 60-80% reduction in peak memory usage for long sequences
- **Scalability**: Support for sequences up to 32K+ tokens
- **Batch Processing**: Maintain reasonable batch sizes even with long contexts
- **Flexibility**: Configurable chunk sizes and overlap for different use cases

---

## 📊 PHASE 4: Performance Benchmarking Framework

### 🎯 Objective
Implement comprehensive performance benchmarking and auto-tuning system to optimize model configurations and track performance improvements across all optimization techniques.

### 📋 Technical Background

A robust benchmarking framework is essential for:

- **Performance Tracking**: Measuring improvement from optimization techniques
- **Auto-tuning**: Automatically finding optimal configurations
- **Regression Detection**: Identifying performance degradations
- **Resource Optimization**: Balancing speed, memory usage, and accuracy

### 🔧 Implementation Strategy

#### 1. Performance Metrics Collection

**File**: `engines/shared/performance_monitor.py`

```python
import time
import psutil
import torch
import json
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import queue
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    
    # Timing metrics
    total_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    compilation_time_ms: float = 0.0
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    prefill_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    memory_efficiency: float = 0.0  # tokens/MB
    
    # GPU metrics
    gpu_utilization_percent: float = 0.0
    gpu_power_watts: float = 0.0
    
    # Model metrics
    input_tokens: int = 0
    output_tokens: int = 0
    batch_size: int = 1
    
    # Optimization metrics
    cuda_graphs_used: bool = False
    guided_decoding_used: bool = False
    chunked_prefill_used: bool = False
    compilation_backend: str = "none"
    
    # Quality metrics
    validation_passed: bool = True
    error_rate: float = 0.0
    
    # Timestamp
    timestamp: str = ""

class PerformanceMonitor:
    """Real-time performance monitoring and metrics collection"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_measurements: Dict[str, Dict] = {}
        
        # Background monitoring
        self.monitoring_queue = queue.Queue()
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # GPU monitoring setup
        if self.enable_gpu_monitoring:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml = pynvml
            except ImportError:
                logger.warning("pynvml not available, GPU monitoring disabled")
                self.enable_gpu_monitoring = False
    
    @contextmanager
    def measure_performance(self, 
                           measurement_id: str,
                           input_tokens: int = 0,
                           batch_size: int = 1):
        """Context manager for measuring performance of code blocks"""
        
        measurement = {
            "id": measurement_id,
            "start_time": time.time(),
            "input_tokens": input_tokens,
            "batch_size": batch_size,
            "start_memory": self._get_memory_usage(),
            "start_gpu_memory": self._get_gpu_memory() if self.enable_gpu_monitoring else 0
        }
        
        self.active_measurements[measurement_id] = measurement
        
        try:
            yield measurement
        finally:
            self._finalize_measurement(measurement_id)
    
    def _finalize_measurement(self, measurement_id: str):
        """Complete measurement and calculate metrics"""
        
        if measurement_id not in self.active_measurements:
            return
        
        measurement = self.active_measurements[measurement_id]
        end_time = time.time()
        
        # Calculate timing
        total_time = (end_time - measurement["start_time"]) * 1000  # ms
        
        # Memory usage
        end_memory = self._get_memory_usage()
        peak_memory = max(measurement["start_memory"], end_memory)
        
        end_gpu_memory = self._get_gpu_memory() if self.enable_gpu_monitoring else 0
        gpu_memory = max(measurement["start_gpu_memory"], end_gpu_memory)
        
        # Create metrics object
        metrics = PerformanceMetrics(
            total_time_ms=total_time,
            peak_memory_mb=peak_memory,
            gpu_memory_mb=gpu_memory,
            input_tokens=measurement["input_tokens"],
            batch_size=measurement["batch_size"],
            timestamp=datetime.now().isoformat()
        )
        
        # Calculate throughput
        if total_time > 0:
            metrics.tokens_per_second = (measurement["input_tokens"] * 1000) / total_time
        
        # Calculate memory efficiency
        if peak_memory > 0:
            metrics.memory_efficiency = measurement["input_tokens"] / peak_memory
        
        # GPU utilization (if available)
        if self.enable_gpu_monitoring:
            try:
                utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                metrics.gpu_utilization_percent = utilization.gpu
                
                power = self.pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # watts
                metrics.gpu_power_watts = power
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
        
        self.metrics_history.append(metrics)
        del self.active_measurements[measurement_id]
    
    def update_optimization_metrics(self, 
                                  measurement_id: str,
                                  cuda_graphs: bool = False,
                                  guided_decoding: bool = False,
                                  chunked_prefill: bool = False,
                                  compilation_backend: str = "none"):
        """Update optimization-specific metrics for a measurement"""
        
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            latest_metrics.cuda_graphs_used = cuda_graphs
            latest_metrics.guided_decoding_used = guided_decoding
            latest_metrics.chunked_prefill_used = chunked_prefill
            latest_metrics.compilation_backend = compilation_backend
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if not self.enable_gpu_monitoring:
            return 0.0
        
        try:
            info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return info.used / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_performance_summary(self, last_n: int = None) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        metrics_subset = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not metrics_subset:
            return {"error": "No metrics available"}
        
        # Calculate aggregated statistics
        total_times = [m.total_time_ms for m in metrics_subset]
        throughputs = [m.tokens_per_second for m in metrics_subset if m.tokens_per_second > 0]
        memory_usage = [m.peak_memory_mb for m in metrics_subset]
        gpu_memory = [m.gpu_memory_mb for m in metrics_subset if m.gpu_memory_mb > 0]
        
        summary = {
            "total_measurements": len(metrics_subset),
            "time_stats": {
                "mean_ms": statistics.mean(total_times),
                "median_ms": statistics.median(total_times),
                "std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
                "min_ms": min(total_times),
                "max_ms": max(total_times)
            },
            "throughput_stats": {
                "mean_tokens_per_sec": statistics.mean(throughputs) if throughputs else 0,
                "median_tokens_per_sec": statistics.median(throughputs) if throughputs else 0,
                "max_tokens_per_sec": max(throughputs) if throughputs else 0
            },
            "memory_stats": {
                "mean_memory_mb": statistics.mean(memory_usage),
                "peak_memory_mb": max(memory_usage),
                "mean_gpu_memory_mb": statistics.mean(gpu_memory) if gpu_memory else 0
            },
            "optimization_usage": {
                "cuda_graphs_rate": sum(m.cuda_graphs_used for m in metrics_subset) / len(metrics_subset),
                "guided_decoding_rate": sum(m.guided_decoding_used for m in metrics_subset) / len(metrics_subset),
                "chunked_prefill_rate": sum(m.chunked_prefill_used for m in metrics_subset) / len(metrics_subset)
            }
        }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        metrics_data = [asdict(m) for m in self.metrics_history]
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        self.metrics_history.clear()

class AutoTuner:
    """Automatic configuration tuning based on performance metrics"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.tuning_history: List[Dict] = []
        self.best_configs: Dict[str, Dict] = {}
    
    def tune_configuration(self, 
                          model_config: Dict[str, Any],
                          test_workload: Callable,
                          parameter_space: Dict[str, List]) -> Dict[str, Any]:
        """
        Automatically tune configuration parameters
        
        Args:
            model_config: Base model configuration
            test_workload: Function that runs evaluation with given config
            parameter_space: Dict of parameter names to list of values to try
            
        Returns:
            Best configuration found
        """
        
        best_config = model_config.copy()
        best_performance = float('inf')  # Minimize latency
        
        # Generate parameter combinations (grid search)
        import itertools
        
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        for combination in itertools.product(*param_values):
            # Create test configuration
            test_config = model_config.copy()
            for param_name, param_value in zip(param_names, combination):
                self._set_nested_config(test_config, param_name, param_value)
            
            logger.info(f"Testing configuration: {dict(zip(param_names, combination))}")
            
            # Run test workload
            try:
                self.monitor.clear_metrics()
                test_workload(test_config)
                
                # Evaluate performance
                summary = self.monitor.get_performance_summary()
                performance_score = self._calculate_performance_score(summary)
                
                # Track this configuration
                self.tuning_history.append({
                    "config": test_config.copy(),
                    "parameters": dict(zip(param_names, combination)),
                    "performance_score": performance_score,
                    "summary": summary
                })
                
                # Check if this is the best so far
                if performance_score < best_performance:
                    best_performance = performance_score
                    best_config = test_config.copy()
                    logger.info(f"New best configuration found: score={performance_score:.2f}")
                
            except Exception as e:
                logger.warning(f"Configuration failed: {e}")
                continue
        
        # Store best configuration
        config_key = self._generate_config_key(model_config)
        self.best_configs[config_key] = {
            "config": best_config,
            "performance_score": best_performance,
            "tuning_timestamp": datetime.now().isoformat()
        }
        
        return best_config
    
    def _set_nested_config(self, config: Dict, key_path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _calculate_performance_score(self, summary: Dict[str, Any]) -> float:
        """Calculate single performance score from summary statistics"""
        
        if "error" in summary:
            return float('inf')
        
        # Weighted combination of metrics (lower is better)
        latency_weight = 0.5
        memory_weight = 0.3
        throughput_weight = -0.2  # Negative because higher throughput is better
        
        latency_score = summary["time_stats"]["mean_ms"]
        memory_score = summary["memory_stats"]["mean_memory_mb"]
        throughput_score = summary["throughput_stats"]["mean_tokens_per_sec"]
        
        # Normalize and combine
        score = (
            latency_weight * latency_score +
            memory_weight * memory_score +
            throughput_weight * throughput_score
        )
        
        return score
    
    def _generate_config_key(self, config: Dict[str, Any]) -> str:
        """Generate unique key for configuration"""
        # Use model name and key parameters
        model_name = config.get("model_name", "unknown")
        batch_size = config.get("batch_size", 1)
        return f"{model_name}_batch_{batch_size}"
    
    def get_tuning_report(self) -> Dict[str, Any]:
        """Generate comprehensive tuning report"""
        
        if not self.tuning_history:
            return {"error": "No tuning history available"}
        
        # Sort by performance score
        sorted_history = sorted(self.tuning_history, key=lambda x: x["performance_score"])
        
        return {
            "total_configurations_tested": len(self.tuning_history),
            "best_configuration": sorted_history[0] if sorted_history else None,
            "worst_configuration": sorted_history[-1] if sorted_history else None,
            "performance_improvement": {
                "best_score": sorted_history[0]["performance_score"],
                "worst_score": sorted_history[-1]["performance_score"],
                "improvement_ratio": sorted_history[-1]["performance_score"] / sorted_history[0]["performance_score"]
            } if len(sorted_history) >= 2 else None,
            "parameter_analysis": self._analyze_parameter_impact()
        }
    
    def _analyze_parameter_impact(self) -> Dict[str, Any]:
        """Analyze impact of different parameters on performance"""
        
        parameter_impact = {}
        
        # Group by parameter values
        for entry in self.tuning_history:
            for param_name, param_value in entry["parameters"].items():
                if param_name not in parameter_impact:
                    parameter_impact[param_name] = {}
                
                if param_value not in parameter_impact[param_name]:
                    parameter_impact[param_name][param_value] = []
                
                parameter_impact[param_name][param_value].append(entry["performance_score"])
        
        # Calculate statistics for each parameter
        impact_summary = {}
        for param_name, value_scores in parameter_impact.items():
            impact_summary[param_name] = {}
            
            for param_value, scores in value_scores.items():
                impact_summary[param_name][param_value] = {
                    "mean_score": statistics.mean(scores),
                    "count": len(scores),
                    "std_score": statistics.stdev(scores) if len(scores) > 1 else 0
                }
        
        return impact_summary

class BenchmarkSuite:
    """Comprehensive benchmarking suite for evaluation system"""
    
    def __init__(self, monitor: PerformanceMonitor, auto_tuner: AutoTuner):
        self.monitor = monitor
        self.auto_tuner = auto_tuner
        self.benchmark_results: Dict[str, Any] = {}
    
    def run_comprehensive_benchmark(self, 
                                  models: List[Dict[str, Any]],
                                  workloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive benchmark across models and workloads"""
        
        logger.info("Starting comprehensive benchmark")
        benchmark_start = time.time()
        
        results = {
            "benchmark_info": {
                "start_time": datetime.now().isoformat(),
                "models_tested": len(models),
                "workloads_tested": len(workloads)
            },
            "model_results": {},
            "workload_results": {},
            "comparative_analysis": {}
        }
        
        # Test each model with each workload
        for model_config in models:
            model_name = model_config.get("model_name", "unknown")
            results["model_results"][model_name] = {}
            
            for workload in workloads:
                workload_name = workload.get("name", "unknown")
                
                logger.info(f"Testing {model_name} with {workload_name}")
                
                # Clear metrics for this test
                self.monitor.clear_metrics()
                
                # Run workload
                try:
                    workload_result = self._run_workload(model_config, workload)
                    performance_summary = self.monitor.get_performance_summary()
                    
                    results["model_results"][model_name][workload_name] = {
                        "workload_result": workload_result,
                        "performance_summary": performance_summary
                    }
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {model_name}/{workload_name}: {e}")
                    results["model_results"][model_name][workload_name] = {
                        "error": str(e)
                    }
        
        # Generate comparative analysis
        results["comparative_analysis"] = self._generate_comparative_analysis(results["model_results"])
        
        # Store results
        benchmark_end = time.time()
        results["benchmark_info"]["total_time_seconds"] = benchmark_end - benchmark_start
        results["benchmark_info"]["end_time"] = datetime.now().isoformat()
        
        self.benchmark_results = results
        return results
    
    def _run_workload(self, model_config: Dict[str, Any], workload: Dict[str, Any]) -> Dict[str, Any]:
        """Run single workload with given model configuration"""
        
        # This would integrate with the actual evaluation pipeline
        # For now, return mock result
        return {
            "status": "completed",
            "mock_result": "This would contain actual evaluation results"
        }
    
    def _generate_comparative_analysis(self, model_results: Dict) -> Dict[str, Any]:
        """Generate comparative analysis across models and workloads"""
        
        analysis = {
            "performance_rankings": {},
            "optimization_effectiveness": {},
            "resource_efficiency": {}
        }
        
        # Analyze performance rankings
        for workload_name in set().union(*[model_data.keys() for model_data in model_results.values()]):
            workload_performances = []
            
            for model_name, model_data in model_results.items():
                if workload_name in model_data and "performance_summary" in model_data[workload_name]:
                    perf = model_data[workload_name]["performance_summary"]
                    if "time_stats" in perf:
                        workload_performances.append({
                            "model": model_name,
                            "latency": perf["time_stats"]["mean_ms"],
                            "throughput": perf["throughput_stats"]["mean_tokens_per_sec"],
                            "memory": perf["memory_stats"]["mean_memory_mb"]
                        })
            
            # Rank by latency (lower is better)
            workload_performances.sort(key=lambda x: x["latency"])
            analysis["performance_rankings"][workload_name] = workload_performances
        
        return analysis
```

#### 2. Integration with Evaluation Pipeline

```python
class PerformanceAwareEvaluator:
    """Enhanced evaluator with integrated performance monitoring"""
    
    def __init__(self, base_evaluator, enable_auto_tuning: bool = True):
        self.base_evaluator = base_evaluator
        self.monitor = PerformanceMonitor()
        self.auto_tuner = AutoTuner(self.monitor) if enable_auto_tuning else None
        self.benchmark_suite = BenchmarkSuite(self.monitor, self.auto_tuner)
    
    def evaluate_with_monitoring(self, 
                               task: Dict[str, Any],
                               model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation with comprehensive performance monitoring"""
        
        task_id = task.get("task_id", "unknown")
        input_tokens = len(task.get("prompt", "").split())  # Rough estimate
        batch_size = model_config.get("batch_size", 1)
        
        with self.monitor.measure_performance(task_id, input_tokens, batch_size) as measurement:
            # Run actual evaluation
            result = self.base_evaluator.evaluate(task, model_config)
            
            # Update optimization metrics based on result
            self.monitor.update_optimization_metrics(
                task_id,
                cuda_graphs=result.get("cuda_graphs_used", False),
                guided_decoding=result.get("guided_decoding_used", False),
                chunked_prefill=result.get("chunked_prefill_used", False),
                compilation_backend=result.get("compilation_backend", "none")
            )
        
        # Add performance metrics to result
        if self.monitor.metrics_history:
            latest_metrics = self.monitor.metrics_history[-1]
            result["performance_metrics"] = asdict(latest_metrics)
        
        return result
    
    def auto_tune_for_task(self, 
                          task: Dict[str, Any],
                          base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-tune configuration for specific task"""
        
        if not self.auto_tuner:
            return base_config
        
        # Define parameter space for tuning
        parameter_space = {
            "batch_size": [1, 2, 4],
            "compilation.enable_cuda_graphs": [True, False],
            "compilation.enable_chunked_prefill": [True, False],
            "generation.temperature": [0.1, 0.7, 1.0]
        }
        
        # Define test workload
        def test_workload(config):
            return self.evaluate_with_monitoring(task, config)
        
        # Run auto-tuning
        optimal_config = self.auto_tuner.tune_configuration(
            base_config, test_workload, parameter_space
        )
        
        return optimal_config
```

### 📈 Expected Benefits

- **Performance Visibility**: Complete visibility into system performance
- **Automatic Optimization**: Auto-tuning reduces manual configuration effort
- **Regression Detection**: Early identification of performance issues
- **Resource Efficiency**: Optimal balance of speed, memory, and accuracy

---

## 🗓️ PHASE 5: Implementation Timeline & Milestones

### 📅 Project Timeline (8-12 weeks)

#### **Week 1-2: Foundation & CUDA Graph Implementation**
- **Milestone 1.1**: Enhanced AOT compiler with CUDA graph capture
  - [ ] Implement `CUDAGraphManager` class
  - [ ] Integrate with existing AOT compilation pipeline
  - [ ] Add graph capture for multiple batch sizes
  - [ ] Performance monitoring integration
  - [ ] Unit tests and validation

- **Milestone 1.2**: CUDA Graph Performance Validation
  - [ ] Benchmark against baseline AOT performance
  - [ ] Validate 10-30% latency improvement
  - [ ] Memory usage analysis
  - [ ] Integration testing with existing models

#### **Week 3-4: Guided Decoding Integration**
- **Milestone 2.1**: xgrammar Integration
  - [ ] Install and configure xgrammar dependency
  - [ ] Implement `GuidedDecodingManager` class
  - [ ] Support for JSON schema, BNF, regex, and Python grammars
  - [ ] Grammar caching and optimization

- **Milestone 2.2**: Evaluation Pipeline Enhancement
  - [ ] Integrate guided decoding with evaluation engine
  - [ ] Task-specific grammar configuration
  - [ ] Output validation and error handling
  - [ ] Performance impact assessment

#### **Week 5-6: Chunked Prefill Implementation**
- **Milestone 3.1**: Chunked Processing Engine
  - [ ] Implement `ChunkedPrefillManager` class
  - [ ] Sequence chunking with overlap handling
  - [ ] Attention mask management across chunks
  - [ ] Memory-efficient chunk processing

- **Milestone 3.2**: Long Context Support
  - [ ] Integration with generation pipeline
  - [ ] Support for 32K+ token sequences
  - [ ] Memory usage optimization
  - [ ] Validation with long-context datasets

#### **Week 7-8: Performance Framework Development**
- **Milestone 4.1**: Monitoring Infrastructure
  - [ ] Implement `PerformanceMonitor` class
  - [ ] Real-time metrics collection
  - [ ] GPU monitoring integration
  - [ ] Metrics export and visualization

- **Milestone 4.2**: Auto-tuning System
  - [ ] Implement `AutoTuner` class
  - [ ] Parameter space exploration
  - [ ] Performance scoring and optimization
  - [ ] Configuration recommendation system

#### **Week 9-10: Integration & Testing**
- **Milestone 5.1**: System Integration
  - [ ] Integrate all optimization techniques
  - [ ] Unified configuration management
  - [ ] Error handling and fallback mechanisms
  - [ ] Documentation and examples

- **Milestone 5.2**: Comprehensive Testing
  - [ ] End-to-end integration tests
  - [ ] Performance regression testing
  - [ ] Stress testing with large datasets
  - [ ] User acceptance testing

#### **Week 11-12: Optimization & Deployment**
- **Milestone 6.1**: Performance Optimization
  - [ ] Fine-tune all optimization parameters
  - [ ] Resolve any performance bottlenecks
  - [ ] Memory usage optimization
  - [ ] Stability improvements

- **Milestone 6.2**: Production Readiness
  - [ ] Production deployment configuration
  - [ ] Monitoring and alerting setup
  - [ ] User documentation and training
  - [ ] Release preparation

### 🎯 Success Criteria

#### **Performance Targets**
- [ ] **Latency Improvement**: 25-40% reduction in average inference time
- [ ] **Memory Efficiency**: 50-70% reduction in peak memory usage for long sequences
- [ ] **Throughput Increase**: 30-50% improvement in tokens/second
- [ ] **Output Quality**: 95%+ validation success rate for structured outputs

#### **Technical Deliverables**
- [ ] **CUDA Graph Integration**: Working implementation with measurable performance gains
- [ ] **Guided Decoding**: Support for JSON, code, and mathematical output formats
- [ ] **Chunked Prefill**: Support for sequences up to 32K+ tokens
- [ ] **Auto-tuning**: Automated configuration optimization for different workloads
- [ ] **Monitoring**: Comprehensive performance tracking and reporting

#### **Quality Assurance**
- [ ] **Test Coverage**: 90%+ code coverage for new components
- [ ] **Documentation**: Complete API documentation and usage examples
- [ ] **Stability**: 99.9% uptime in production environment
- [ ] **Backwards Compatibility**: No breaking changes to existing evaluation pipeline

### 🔧 Implementation Strategy

#### **Phase-by-Phase Approach**
1. **Foundation First**: Start with CUDA graphs as they provide immediate performance benefits
2. **Incremental Integration**: Add each technique one at a time to isolate issues
3. **Continuous Testing**: Test each component thoroughly before moving to the next
4. **Performance Validation**: Measure and validate improvements at each step

#### **Risk Mitigation**
- **Fallback Mechanisms**: Ensure system works without optimizations if they fail
- **Progressive Rollout**: Deploy optimizations gradually to production
- **Monitoring**: Comprehensive monitoring to detect issues early
- **Documentation**: Detailed troubleshooting guides and known issues

#### **Resource Requirements**
- **Development Time**: 8-12 weeks with 1-2 developers
- **Hardware**: GPU-enabled development environment for testing
- **Dependencies**: xgrammar, pynvml, additional monitoring tools
- **Testing**: Access to diverse evaluation datasets and model configurations

### 📊 Expected ROI

#### **Performance Improvements**
- **Inference Speed**: 25-40% faster evaluation runs
- **Memory Usage**: 50-70% reduction in memory requirements
- **Resource Costs**: 30-50% reduction in compute costs for large evaluations
- **Development Velocity**: 2-3x faster evaluation iteration cycles

#### **Quality Improvements**
- **Output Reliability**: 90%+ reduction in parsing errors
- **Evaluation Accuracy**: Higher quality evaluations through structured outputs
- **System Stability**: More predictable and reliable evaluation runs
- **Maintainability**: Better monitoring and debugging capabilities

### 📋 Next Steps

1. **Environment Setup**: Install required dependencies (xgrammar, pynvml)
2. **Baseline Measurements**: Establish current performance baselines
3. **CUDA Graph Development**: Begin with Phase 1 implementation
4. **Iterative Development**: Follow phase-by-phase implementation plan
5. **Continuous Integration**: Set up CI/CD pipeline for new components
6. **Documentation**: Create comprehensive implementation documentation

---

## 📝 Conclusion

This comprehensive integration plan provides a roadmap for incorporating advanced vLLM techniques into our hybrid evaluation system. The phased approach ensures:

- **Incremental Value**: Each phase delivers measurable improvements
- **Risk Management**: Careful testing and validation at each step  
- **Scalability**: Foundation for future optimization techniques
- **Production Readiness**: Robust, well-tested, and monitored implementation

The expected performance improvements (25-40% latency reduction, 50-70% memory efficiency gains) will significantly enhance the evaluation system's capabilities while maintaining reliability and ease of use.

Implementation should begin with establishing baseline performance metrics, followed by Phase 1 (CUDA Graph integration) to achieve immediate performance gains, then proceeding through subsequent phases to build a comprehensive, highly-optimized evaluation platform.