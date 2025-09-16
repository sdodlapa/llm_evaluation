# Evaluation Datasets Documentation

## Overview

This document provides comprehensive information about the evaluation datasets used for LLM testing, their characteristics, and the functionality built to manage them. Our dataset collection is designed to evaluate key capabilities for modern LLMs, particularly focusing on coding, reasoning, function calling, and instruction following.

## Dataset Collection Summary

- **Total Datasets**: 12 curated benchmarks
- **Total Size**: ~6.5GB (well under 10GB limit)
- **Task Categories**: 5 primary types
- **License Coverage**: MIT, Apache-2.0, CC-BY-4.0
- **Sample Range**: 10-14,042 samples per dataset

## Dataset Categories

### 1. Function Calling & Agent Tasks

#### Berkeley Function Calling Leaderboard (BFCL)
- **Size**: 50MB (subset)
- **Samples**: ~2,000
- **License**: Apache-2.0
- **Source**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- **Description**: Comprehensive benchmark for evaluating function calling capabilities with real API scenarios
- **Task Format**: 
  - Input: Natural language request + function definitions
  - Output: Structured function calls (JSON format)
  - Evaluation: Function name accuracy + parameter correctness
- **Use Case**: Testing agent capabilities for tool usage and API interaction

#### ToolLLaMA
- **Size**: 100MB (subset)
- **Samples**: ~1,500
- **License**: Apache-2.0
- **Source**: `ToolBench/ToolBench`
- **Description**: Tool learning dataset for LLMs with diverse API usage scenarios
- **Task Format**:
  - Input: Complex multi-step tasks requiring tool usage
  - Output: Sequence of tool calls with reasoning
  - Evaluation: Task completion accuracy + tool selection appropriateness
- **Use Case**: Advanced agent workflows and multi-tool coordination

### 2. Code Generation & Understanding

#### HumanEval
- **Size**: 5MB
- **Samples**: 164
- **License**: MIT
- **Source**: `openai_humaneval`
- **Description**: Hand-written programming problems designed to test code generation capabilities
- **Task Format**:
  - Input: Function signature + docstring + examples
  - Output: Complete function implementation
  - Evaluation: Unit test execution (pass/fail)
- **Key Features**:
  - High-quality, human-written problems
  - Comprehensive test suites
  - Standard benchmark in code generation
- **Example**:
  ```python
  def has_close_elements(numbers: List[float], threshold: float) -> bool:
      """ Check if in given list of numbers, are any two numbers closer to each other than
      given threshold.
      >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
      False
      >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
      True
      """
  ```

#### MBPP (Mostly Basic Python Problems)
- **Size**: 10MB
- **Samples**: 974
- **License**: CC-BY-4.0
- **Source**: `mbpp`
- **Description**: Basic Python programming problems for beginners to intermediate level
- **Task Format**:
  - Input: Problem description in natural language
  - Output: Python code solution
  - Evaluation: Test case execution + code quality
- **Difficulty Levels**: Basic to intermediate Python programming
- **Coverage**: Data structures, algorithms, string manipulation, math

#### CodeT5
- **Size**: 150MB
- **Samples**: ~5,000
- **License**: Apache-2.0
- **Source**: `code_x_glue_ct_code_to_text`
- **Description**: Code understanding and generation tasks across multiple programming languages
- **Task Format**:
  - Input: Code snippets or natural language descriptions
  - Output: Code generation or natural language explanation
  - Evaluation: BLEU score + functional correctness
- **Languages**: Python, Java, JavaScript, PHP, Ruby, Go

### 3. Reasoning & Problem Solving

#### GSM8K (Grade School Math)
- **Size**: 3MB
- **Samples**: 1,319
- **License**: MIT
- **Source**: `gsm8k`
- **Description**: Grade school math word problems requiring multi-step reasoning
- **Task Format**:
  - Input: Natural language math word problem
  - Output: Step-by-step solution with final numerical answer
  - Evaluation: Final answer correctness + reasoning step quality
- **Key Features**:
  - Requires chain-of-thought reasoning
  - Real-world problem contexts
  - Numerical and logical reasoning combined
- **Example**:
  ```
  Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
  ```

#### ARC-Challenge (AI2 Reasoning Challenge)
- **Size**: 2MB
- **Samples**: 1,172
- **License**: CC-BY-SA-4.0
- **Source**: `ai2_arc`
- **Description**: Multiple-choice science questions requiring complex reasoning
- **Task Format**:
  - Input: Science question + 4 multiple choice options
  - Output: Selected answer (A, B, C, or D)
  - Evaluation: Answer accuracy
- **Difficulty**: Challenging subset of ARC dataset
- **Topics**: Physics, chemistry, biology, earth science

#### HellaSwag
- **Size**: 20MB
- **Samples**: 10,042
- **License**: MIT
- **Source**: `hellaswag`
- **Description**: Commonsense reasoning about physical situations and everyday activities
- **Task Format**:
  - Input: Context sentence + 4 possible continuations
  - Output: Most plausible continuation selection
  - Evaluation: Selection accuracy
- **Focus**: Common sense reasoning and world knowledge

### 4. Instruction Following

#### AlpacaEval
- **Size**: 15MB
- **Samples**: 805
- **License**: Apache-2.0
- **Source**: `tatsu-lab/alpaca_eval`
- **Description**: Comprehensive evaluation of instruction-following capabilities
- **Task Format**:
  - Input: Diverse instructions and prompts
  - Output: Appropriate responses following instructions
  - Evaluation: Response quality and instruction adherence
- **Categories**: Creative writing, analysis, coding, math, roleplay

#### MT-Bench
- **Size**: 5MB
- **Samples**: 160
- **License**: Apache-2.0
- **Source**: `lmsys/mt_bench_human_judgments`
- **Description**: Multi-turn conversation benchmark for chat capabilities
- **Task Format**:
  - Input: Multi-turn conversation scenarios
  - Output: Contextually appropriate responses
  - Evaluation: Conversation coherence and quality
- **Features**: Tests conversation memory and context understanding

### 5. Knowledge & Question Answering

#### MMLU (Massive Multitask Language Understanding)
- **Size**: 200MB
- **Samples**: 14,042
- **License**: MIT
- **Source**: `cais/mmlu`
- **Description**: Comprehensive knowledge evaluation across 57 academic subjects
- **Task Format**:
  - Input: Multiple choice questions from various academic domains
  - Output: Selected answer (A, B, C, or D)
  - Evaluation: Accuracy across subjects
- **Subjects**: STEM, humanities, social sciences, and more
- **Difficulty**: Undergraduate to graduate level

#### TruthfulQA
- **Size**: 5MB
- **Samples**: 817
- **License**: Apache-2.0
- **Source**: `truthful_qa`
- **Description**: Questions designed to test for truthful and accurate responses
- **Task Format**:
  - Input: Questions that humans often answer incorrectly due to misconceptions
  - Output: Truthful and accurate answers
  - Evaluation: Truthfulness and informativeness
- **Purpose**: Tests resistance to generating false but plausible information

## Dataset Management Functionality

### Core Management System

#### EvaluationDatasetManager Class
The central class for handling all dataset operations:

```python
manager = EvaluationDatasetManager(
    cache_dir="evaluation_data",
    max_total_size_gb=10.0
)
```

**Key Methods**:
- `download_dataset(dataset_name)`: Download and cache specific dataset
- `download_recommended_datasets()`: Download all recommended datasets
- `load_cached_dataset(dataset_name)`: Load previously cached dataset
- `get_dataset_summary()`: Get comprehensive overview
- `get_recommended_datasets()`: Get list based on task types and constraints

#### Automatic Data Processing
- **Standardization**: Converts all datasets to unified format
- **Sampling Control**: Limits large datasets for efficiency
- **Error Handling**: Robust error handling with detailed logging
- **Metadata Extraction**: Automatic analysis of dataset characteristics

### Command Line Interface

#### Dataset Management CLI (`manage_datasets.py`)

```bash
# View comprehensive summary
python manage_datasets.py --summary

# Download recommended datasets
python manage_datasets.py --download-recommended

# Download specific datasets
python manage_datasets.py --download humaneval gsm8k mbpp

# List available datasets
python manage_datasets.py --list
python manage_datasets.py --list --task-type coding

# Analyze cached dataset
python manage_datasets.py --analyze humaneval
```

**Features**:
- Color-coded status indicators (✅ cached, ⬜ available)
- Size and sample count information
- License and description details
- Task-type filtering
- Detailed dataset analysis

### Data Processing Pipeline

#### 1. Download Phase
```python
def download_dataset(self, dataset_name: str) -> Dict[str, Any]:
    # Check cache first
    # Download from HuggingFace or direct source
    # Handle special dataset configurations
    # Process and standardize format
    # Save to cache with metadata
```

#### 2. Standardization Phase
Each dataset type is converted to a standard format:

**Coding Tasks**:
```python
{
    "id": "task_id",
    "prompt": "function signature + description",
    "expected_output": "correct implementation",
    "test_cases": ["list of test cases"],
    "difficulty": "easy/medium/hard"
}
```

**Function Calling**:
```python
{
    "id": "unique_id",
    "prompt": "natural language request",
    "functions": ["available function definitions"],
    "expected_calls": ["expected function invocations"],
    "category": "task category"
}
```

**Reasoning Tasks**:
```python
{
    "id": "unique_id",
    "question": "problem statement",
    "choices": ["A", "B", "C", "D"],  # if multiple choice
    "answer": "correct answer",
    "explanation": "reasoning steps"
}
```

#### 3. Analysis Phase
Automatic analysis includes:
- **Sample Statistics**: Count, size, completeness
- **Content Analysis**: Text length distributions, empty fields
- **Quality Metrics**: Unique IDs, label consistency
- **Memory Estimation**: Storage requirements and processing time

### Integration with Evaluation System

#### Evaluation Metrics per Dataset Type

**Code Execution Accuracy**:
```python
def code_execution_accuracy(predictions, test_cases):
    # Execute code against test cases
    # Safe execution with timeout
    # Syntax and runtime error handling
    # Pass/fail determination
```

**Function Calling Accuracy**:
```python
def function_calling_accuracy(predictions, expected_calls):
    # Extract function calls from text
    # Compare function names and parameters
    # Support JSON and XML formats
    # Flexible parameter matching
```

**Multiple Choice Accuracy**:
```python
def multiple_choice_accuracy(predictions, references):
    # Extract choice letters (A, B, C, D)
    # Handle various response formats
    # Pattern matching for robustness
```

#### Prompt Generation
Automatic prompt creation based on task type:

```python
def _create_prompt_from_sample(self, sample: Dict, task_type: str) -> str:
    if task_type == "coding":
        return sample.get("prompt", "")
    elif task_type == "function_calling":
        prompt = sample.get("prompt", "")
        functions = sample.get("functions", [])
        # Add function definitions to prompt
    elif task_type == "reasoning":
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        # Format multiple choice or open-ended
```

## Usage Patterns

### Quick Start
```bash
# 1. Download essential datasets
python manage_datasets.py --download humaneval gsm8k mbpp

# 2. Run evaluation with real datasets
python evaluation/run_evaluation.py --preset balanced

# 3. View results
ls results/dataset_results/
```

### Development Workflow
```bash
# 1. Check what's available
python manage_datasets.py --summary

# 2. Download specific task types
python manage_datasets.py --download-recommended --task-type coding

# 3. Test with synthetic data first
python evaluation/run_evaluation.py --synthetic-only --quick-test

# 4. Run full evaluation with real datasets
python evaluation/run_evaluation.py --datasets-only --preset performance
```

### Analysis Workflow
```bash
# 1. Analyze downloaded datasets
python manage_datasets.py --analyze humaneval

# 2. List datasets by category
python manage_datasets.py --list --task-type reasoning

# 3. Check cache status
python manage_datasets.py --summary
```

## Dataset Selection Strategy

### High Priority Datasets (Default)
1. **HumanEval** - Code generation gold standard
2. **GSM8K** - Math reasoning benchmark
3. **BFCL** - Function calling evaluation
4. **MBPP** - Python programming basics
5. **AlpacaEval** - Instruction following

### Task-Specific Recommendations

**For Agent Development**:
- BFCL (function calling)
- ToolLLaMA (multi-tool workflows)
- AlpacaEval (instruction following)

**For Code Generation**:
- HumanEval (gold standard)
- MBPP (broader coverage)
- CodeT5 (multi-language)

**For Reasoning**:
- GSM8K (mathematical reasoning)
- ARC-Challenge (scientific reasoning)
- HellaSwag (commonsense reasoning)

**For Knowledge Testing**:
- MMLU (comprehensive knowledge)
- TruthfulQA (factual accuracy)

## Performance Considerations

### Memory Management
- **Streaming Processing**: Large datasets processed in chunks
- **Cache Management**: Automatic cleanup of old cached data
- **Size Limits**: Configurable total cache size (default 10GB)

### Processing Efficiency
- **Sample Limiting**: Large datasets limited to 1000 samples for quick iteration
- **Parallel Processing**: Multiple datasets can be processed simultaneously
- **Incremental Loading**: Only load needed portions of datasets

### Evaluation Speed
- **Preset-Based Sampling**: Different presets use different dataset subsets
  - Memory Optimized: 3 datasets, 50 samples each
  - Balanced: 5 datasets, 100 samples each  
  - Performance: All datasets, full samples

## Error Handling and Robustness

### Download Resilience
- **Network Error Recovery**: Automatic retry with exponential backoff
- **Partial Download Handling**: Resume interrupted downloads
- **Format Validation**: Verify dataset integrity after download

### Processing Robustness
- **Schema Validation**: Ensure consistent data format
- **Error Isolation**: Failed samples don't break entire evaluation
- **Graceful Degradation**: Continue with available datasets if some fail

### Monitoring and Logging
- **Detailed Logging**: All operations logged with timestamps
- **Progress Tracking**: Real-time progress for long operations
- **Error Reporting**: Comprehensive error messages and stack traces

## Future Extensions

### Planned Additions
1. **Multilingual Datasets**: Support for non-English evaluations
2. **Domain-Specific Benchmarks**: Medical, legal, financial domains
3. **Adversarial Testing**: Robustness and safety evaluations
4. **Custom Dataset Integration**: Support for user-provided datasets

### Technical Improvements
1. **Streaming Evaluation**: Process datasets without full loading
2. **Distributed Processing**: Multi-node dataset processing
3. **Advanced Metrics**: Semantic similarity, coherence scoring
4. **Real-time Updates**: Automatic dataset version management

This comprehensive dataset management system provides a solid foundation for rigorous LLM evaluation across multiple dimensions of capability, with robust tooling for efficient dataset handling and analysis.