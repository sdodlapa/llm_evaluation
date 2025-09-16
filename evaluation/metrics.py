"""
Evaluation metrics for different LLM task types
Provides standardized scoring functions for model evaluation
"""

import re
import json
import ast
import string
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of an evaluation metric"""
    score: float
    total_samples: int
    successful_samples: int
    metric_name: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

class EvaluationMetrics:
    """Collection of evaluation metrics for different task types"""
    
    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> EvaluationResult:
        """Exact string match accuracy"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip().lower() == ref.strip().lower())
        
        return EvaluationResult(
            score=matches / len(predictions) if predictions else 0.0,
            total_samples=len(predictions),
            successful_samples=matches,
            metric_name="exact_match",
            details={"case_sensitive": False}
        )
    
    @staticmethod
    def multiple_choice_accuracy(predictions: List[str], references: List[str], 
                               choices: List[List[str]] = None) -> EvaluationResult:
        """Multiple choice accuracy (A, B, C, D or 0, 1, 2, 3)"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        correct = 0
        processed_predictions = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Clean and normalize prediction
            pred_clean = EvaluationMetrics._extract_choice(pred)
            ref_clean = ref.strip().upper()
            
            processed_predictions.append(pred_clean)
            if pred_clean == ref_clean:
                correct += 1
        
        return EvaluationResult(
            score=correct / len(predictions) if predictions else 0.0,
            total_samples=len(predictions),
            successful_samples=correct,
            metric_name="multiple_choice_accuracy",
            details={
                "processed_predictions": processed_predictions[:10],  # First 10 for debugging
                "reference_format": references[:10] if references else []
            }
        )
    
    @staticmethod
    def _extract_choice(text: str) -> str:
        """Extract choice letter from model output"""
        text = text.strip()
        
        # Look for patterns like "Answer: A", "The answer is B", "(C)", etc.
        patterns = [
            r'(?:answer|choice)(?:\s*is)?:?\s*([A-D])',
            r'^\s*([A-D])\s*[.:)]',
            r'\(([A-D])\)',
            r'\b([A-D])\b(?:\s*[.:)]|\s*$)',
            r'option\s*([A-D])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, check if it's just a single letter
        if len(text) == 1 and text.upper() in 'ABCD':
            return text.upper()
        
        # Default to first letter if found
        for char in text.upper():
            if char in 'ABCD':
                return char
        
        return ""  # No valid choice found
    
    @staticmethod
    def code_execution_accuracy(predictions: List[str], test_cases: List[List[Dict]], 
                              timeout: int = 5) -> EvaluationResult:
        """Evaluate code by running test cases"""
        if len(predictions) != len(test_cases):
            raise ValueError("Predictions and test cases must have same length")
        
        results = []
        successful = 0
        
        for i, (code, tests) in enumerate(zip(predictions, test_cases)):
            try:
                test_result = EvaluationMetrics._execute_code_tests(code, tests, timeout)
                results.append(test_result)
                if test_result["passed"]:
                    successful += 1
            except Exception as e:
                results.append({
                    "passed": False,
                    "error": str(e),
                    "tests_run": 0,
                    "tests_passed": 0
                })
        
        return EvaluationResult(
            score=successful / len(predictions) if predictions else 0.0,
            total_samples=len(predictions),
            successful_samples=successful,
            metric_name="code_execution_accuracy",
            details={
                "test_results": results[:5],  # First 5 for debugging
                "timeout_seconds": timeout
            }
        )
    
    @staticmethod
    def _execute_code_tests(code: str, test_cases: List[Dict], timeout: int) -> Dict:
        """Execute code against test cases safely"""
        import signal
        import contextlib
        import io
        import sys
        
        # Extract function from code
        try:
            # Parse the code to find the main function
            tree = ast.parse(code)
            function_name = None
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break
            
            if not function_name:
                return {"passed": False, "error": "No function found", "tests_run": 0, "tests_passed": 0}
            
        except SyntaxError as e:
            return {"passed": False, "error": f"Syntax error: {e}", "tests_run": 0, "tests_passed": 0}
        
        # Create execution environment
        exec_globals = {}
        
        try:
            # Execute the code
            exec(code, exec_globals)
            
            if function_name not in exec_globals:
                return {"passed": False, "error": f"Function {function_name} not found after execution", 
                       "tests_run": 0, "tests_passed": 0}
            
            func = exec_globals[function_name]
            
            # Run test cases
            tests_passed = 0
            tests_run = 0
            
            for test_case in test_cases:
                tests_run += 1
                try:
                    # Handle different test case formats
                    if "input" in test_case and "output" in test_case:
                        input_val = test_case["input"]
                        expected = test_case["output"]
                        
                        if isinstance(input_val, list):
                            result = func(*input_val)
                        else:
                            result = func(input_val)
                        
                        if result == expected:
                            tests_passed += 1
                    elif "assert" in test_case:
                        # Handle assertion-style tests
                        assertion = test_case["assert"].replace(function_name, "func")
                        if eval(assertion, {"func": func}):
                            tests_passed += 1
                            
                except Exception as e:
                    # Test failed with exception
                    continue
            
            passed = tests_passed == tests_run and tests_run > 0
            
            return {
                "passed": passed,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "error": None
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "tests_run": 0,
                "tests_passed": 0
            }
    
    @staticmethod
    def function_calling_accuracy(predictions: List[str], expected_calls: List[List[Dict]]) -> EvaluationResult:
        """Evaluate function calling accuracy"""
        if len(predictions) != len(expected_calls):
            raise ValueError("Predictions and expected calls must have same length")
        
        correct = 0
        call_results = []
        
        for pred, expected in zip(predictions, expected_calls):
            try:
                # Extract function calls from prediction
                extracted_calls = EvaluationMetrics._extract_function_calls(pred)
                
                # Compare with expected calls
                call_correct = EvaluationMetrics._compare_function_calls(extracted_calls, expected)
                call_results.append({
                    "extracted_calls": extracted_calls,
                    "expected_calls": expected,
                    "correct": call_correct
                })
                
                if call_correct:
                    correct += 1
                    
            except Exception as e:
                call_results.append({
                    "error": str(e),
                    "correct": False
                })
        
        return EvaluationResult(
            score=correct / len(predictions) if predictions else 0.0,
            total_samples=len(predictions),
            successful_samples=correct,
            metric_name="function_calling_accuracy",
            details={
                "call_results": call_results[:5],  # First 5 for debugging
            }
        )
    
    @staticmethod
    def _extract_function_calls(text: str) -> List[Dict]:
        """Extract function calls from model output"""
        calls = []
        
        # Look for JSON-style function calls
        json_pattern = r'\{[^}]*"function_call"[^}]*\{[^}]*\}[^}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if "function_call" in parsed:
                    calls.append(parsed["function_call"])
            except json.JSONDecodeError:
                continue
        
        # Look for XML-style function calls
        xml_pattern = r'<function_call>\s*<name>([^<]+)</name>\s*<arguments>([^<]*)</arguments>\s*</function_call>'
        xml_matches = re.findall(xml_pattern, text, re.DOTALL)
        
        for name, args in xml_matches:
            try:
                parsed_args = json.loads(args) if args.strip() else {}
                calls.append({
                    "name": name.strip(),
                    "arguments": parsed_args
                })
            except json.JSONDecodeError:
                calls.append({
                    "name": name.strip(),
                    "arguments": {"raw": args.strip()}
                })
        
        # Look for natural language function calls
        # This is more heuristic and would need domain-specific patterns
        
        return calls
    
    @staticmethod
    def _compare_function_calls(extracted: List[Dict], expected: List[Dict]) -> bool:
        """Compare extracted function calls with expected ones"""
        if len(extracted) != len(expected):
            return False
        
        for ext_call, exp_call in zip(extracted, expected):
            # Check function name
            if ext_call.get("name", "").lower() != exp_call.get("name", "").lower():
                return False
            
            # Check arguments (more flexible comparison)
            ext_args = ext_call.get("arguments", {})
            exp_args = exp_call.get("arguments", {})
            
            # For now, just check if key arguments are present
            for key, expected_value in exp_args.items():
                if key not in ext_args:
                    return False
                # Could add more sophisticated comparison here
        
        return True
    
    @staticmethod
    def reasoning_score(predictions: List[str], references: List[str], 
                       expected_steps: List[List[str]] = None) -> EvaluationResult:
        """Evaluate reasoning quality"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        scores = []
        step_scores = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Basic answer correctness
            answer_score = 1.0 if EvaluationMetrics._normalize_answer(pred) == EvaluationMetrics._normalize_answer(ref) else 0.0
            
            # Reasoning steps if provided
            step_score = 0.0
            if expected_steps and i < len(expected_steps):
                step_score = EvaluationMetrics._score_reasoning_steps(pred, expected_steps[i])
            
            # Combined score (70% answer, 30% reasoning if available)
            if expected_steps and i < len(expected_steps):
                final_score = 0.7 * answer_score + 0.3 * step_score
            else:
                final_score = answer_score
            
            scores.append(final_score)
            step_scores.append(step_score)
        
        return EvaluationResult(
            score=np.mean(scores) if scores else 0.0,
            total_samples=len(predictions),
            successful_samples=sum(1 for s in scores if s > 0.5),
            metric_name="reasoning_score",
            details={
                "individual_scores": scores[:10],
                "step_scores": step_scores[:10],
                "has_step_evaluation": expected_steps is not None
            }
        )
    
    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer for comparison"""
        # Extract numeric answers
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return str(float(numbers[-1]))  # Take the last number
            except ValueError:
                pass
        
        # Clean text answer
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    @staticmethod
    def _score_reasoning_steps(prediction: str, expected_steps: List[str]) -> float:
        """Score the reasoning steps in a prediction"""
        pred_lower = prediction.lower()
        found_steps = 0
        
        for step in expected_steps:
            if step.lower() in pred_lower:
                found_steps += 1
        
        return found_steps / len(expected_steps) if expected_steps else 0.0
    
    @staticmethod
    def instruction_following_score(predictions: List[str], instructions: List[str], 
                                  requirements: List[List[str]] = None) -> EvaluationResult:
        """Evaluate instruction following quality"""
        if len(predictions) != len(instructions):
            raise ValueError("Predictions and instructions must have same length")
        
        scores = []
        requirement_scores = []
        
        for i, pred in enumerate(predictions):
            score = 0.0
            req_score = 0.0
            
            if requirements and i < len(requirements):
                # Check specific requirements
                req_score = EvaluationMetrics._check_requirements(pred, requirements[i])
                
                # Basic instruction coherence (simplified)
                coherence_score = EvaluationMetrics._score_coherence(pred, instructions[i])
                
                # Combined score
                score = 0.6 * req_score + 0.4 * coherence_score
            else:
                # Just coherence if no specific requirements
                score = EvaluationMetrics._score_coherence(pred, instructions[i])
            
            scores.append(score)
            requirement_scores.append(req_score)
        
        return EvaluationResult(
            score=np.mean(scores) if scores else 0.0,
            total_samples=len(predictions),
            successful_samples=sum(1 for s in scores if s > 0.7),
            metric_name="instruction_following_score",
            details={
                "individual_scores": scores[:10],
                "requirement_scores": requirement_scores[:10],
                "has_requirements": requirements is not None
            }
        )
    
    @staticmethod
    def _check_requirements(prediction: str, requirements: List[str]) -> float:
        """Check if prediction meets specific requirements"""
        pred_lower = prediction.lower()
        met_requirements = 0
        
        for req in requirements:
            req_lower = req.lower()
            
            # Simple keyword checking (could be more sophisticated)
            if req_lower in pred_lower:
                met_requirements += 1
            # Special cases
            elif req_lower == "json" and ('{' in prediction and '}' in prediction):
                met_requirements += 1
            elif req_lower == "python" and 'def ' in prediction:
                met_requirements += 1
            elif req_lower == "docstring" and ('"""' in prediction or "'''" in prediction):
                met_requirements += 1
        
        return met_requirements / len(requirements) if requirements else 0.0
    
    @staticmethod
    def _score_coherence(prediction: str, instruction: str) -> float:
        """Score coherence between prediction and instruction (simplified)"""
        # Very basic coherence scoring
        # In practice, you might use embedding similarity or other NLP metrics
        
        pred_words = set(prediction.lower().split())
        inst_words = set(instruction.lower().split())
        
        if not inst_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(pred_words & inst_words)
        relevance = overlap / len(inst_words)
        
        # Basic length appropriateness (not too short, not too long)
        pred_len = len(prediction.split())
        if pred_len < 5:
            length_score = 0.5
        elif pred_len > 500:
            length_score = 0.7
        else:
            length_score = 1.0
        
        return min(1.0, relevance + 0.3 * length_score)

def evaluate_dataset_predictions(dataset_type: str, predictions: List[str], 
                               dataset_samples: List[Dict]) -> Dict[str, EvaluationResult]:
    """Evaluate predictions against a dataset"""
    metrics = EvaluationMetrics()
    results = {}
    
    if dataset_type == "coding":
        # Extract test cases and expected outputs
        test_cases = [sample.get("test_cases", []) for sample in dataset_samples]
        expected_outputs = [sample.get("expected_output", "") for sample in dataset_samples]
        
        if any(test_cases):
            results["code_execution"] = metrics.code_execution_accuracy(predictions, test_cases)
        
        if expected_outputs:
            results["exact_match"] = metrics.exact_match(predictions, expected_outputs)
    
    elif dataset_type == "function_calling":
        expected_calls = [sample.get("expected_calls", []) for sample in dataset_samples]
        results["function_calling"] = metrics.function_calling_accuracy(predictions, expected_calls)
    
    elif dataset_type == "reasoning":
        answers = [sample.get("answer", "") for sample in dataset_samples]
        expected_steps = [sample.get("explanation", "").split() for sample in dataset_samples]
        
        if answers:
            # Check if it's multiple choice
            if all(len(ans) == 1 and ans in 'ABCD' for ans in answers if ans):
                results["multiple_choice"] = metrics.multiple_choice_accuracy(predictions, answers)
            else:
                results["reasoning"] = metrics.reasoning_score(predictions, answers, expected_steps)
    
    elif dataset_type == "instruction_following":
        instructions = [sample.get("instruction", "") for sample in dataset_samples]
        requirements = [sample.get("requirements", []) for sample in dataset_samples if "requirements" in sample]
        
        if instructions:
            results["instruction_following"] = metrics.instruction_following_score(
                predictions, instructions, requirements if requirements else None)
    
    elif dataset_type == "qa":
        answers = [sample.get("answer", "") for sample in dataset_samples]
        choices = [sample.get("choices", []) for sample in dataset_samples]
        
        if answers:
            # Check if it's multiple choice
            if all(len(ans) == 1 and ans in 'ABCD' for ans in answers if ans):
                results["multiple_choice"] = metrics.multiple_choice_accuracy(predictions, answers)
            else:
                results["exact_match"] = metrics.exact_match(predictions, answers)
    
    return results

def print_evaluation_summary(results: Dict[str, EvaluationResult]):
    """Print a formatted summary of evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    for metric_name, result in results.items():
        print(f"\n📊 {metric_name.upper()}")
        print(f"   Score: {result.score:.3f} ({result.score*100:.1f}%)")
        print(f"   Successful: {result.successful_samples}/{result.total_samples}")
        
        if result.details:
            print(f"   Details: {len(result.details)} additional metrics")
    
    # Overall summary
    if results:
        avg_score = np.mean([r.score for r in results.values()])
        print(f"\n🎯 OVERALL AVERAGE: {avg_score:.3f} ({avg_score*100:.1f}%)")
    
    print("="*60)

if __name__ == "__main__":
    # Example usage
    metrics = EvaluationMetrics()
    
    # Test exact match
    preds = ["Paris", "London", "Tokyo"]
    refs = ["Paris", "london", "Tokyo"]
    result = metrics.exact_match(preds, refs)
    print(f"Exact match: {result.score:.2f}")
    
    # Test multiple choice
    mc_preds = ["The answer is A", "B", "(C)", "I think D is correct"]
    mc_refs = ["A", "B", "C", "D"]
    mc_result = metrics.multiple_choice_accuracy(mc_preds, mc_refs)
    print(f"Multiple choice: {mc_result.score:.2f}")