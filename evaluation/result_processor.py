"""
Result Processor - Analysis and aggregation of evaluation results
Handles result formatting, comparison analysis, and report generation
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ResultProcessor:
    """Process and analyze evaluation results"""
    
    def __init__(self):
        self.metric_weights = {
            'accuracy': 0.3,
            'consistency': 0.2,
            'relevance': 0.2,
            'coherence': 0.15,
            'completeness': 0.15
        }
    
    def generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of evaluation results"""
        logger.info("Generating evaluation summary")
        
        summary = {
            'overview': {
                'total_models': len(results),
                'models_evaluated': list(results.keys()),
                'timestamp': datetime.now().isoformat()
            },
            'model_summaries': {},
            'dataset_performance': {},
            'preset_analysis': {},
            'overall_rankings': {}
        }
        
        # Analyze each model
        all_model_scores = {}
        dataset_scores = {}
        preset_scores = {}
        
        for model_name, model_results in results.items():
            model_summary = self._analyze_model_results(model_name, model_results)
            summary['model_summaries'][model_name] = model_summary
            
            # Collect scores for rankings
            if 'overall_score' in model_summary:
                all_model_scores[model_name] = model_summary['overall_score']
            
            # Collect dataset performance
            for preset, preset_data in model_results.items():
                if 'dataset_results' in preset_data:
                    for dataset, dataset_result in preset_data['dataset_results'].items():
                        if dataset not in dataset_scores:
                            dataset_scores[dataset] = {}
                        
                        # Extract performance metrics
                        metrics = self._extract_performance_metrics(dataset_result)
                        if metrics:
                            dataset_scores[dataset][f"{model_name}_{preset}"] = metrics
                
                # Collect preset performance
                preset_key = f"{model_name}_{preset}"
                preset_scores[preset_key] = self._calculate_preset_score(preset_data)
        
        # Generate dataset performance summary
        summary['dataset_performance'] = self._analyze_dataset_performance(dataset_scores)
        
        # Generate preset analysis
        summary['preset_analysis'] = self._analyze_preset_performance(preset_scores)
        
        # Generate overall rankings
        summary['overall_rankings'] = self._generate_rankings(all_model_scores)
        
        logger.info(f"Summary generated for {len(results)} models")
        return summary
    
    def _analyze_model_results(self, model_name: str, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results for a single model"""
        model_summary = {
            'model_name': model_name,
            'presets_tested': list(model_results.keys()),
            'preset_performance': {},
            'average_scores': {},
            'strengths': [],
            'weaknesses': [],
            'overall_score': 0.0
        }
        
        all_scores = []
        preset_scores = {}
        
        for preset, preset_data in model_results.items():
            preset_analysis = self._analyze_preset_results(preset_data)
            model_summary['preset_performance'][preset] = preset_analysis
            
            if 'average_score' in preset_analysis:
                preset_scores[preset] = preset_analysis['average_score']
                all_scores.append(preset_analysis['average_score'])
        
        # Calculate overall model performance
        if all_scores:
            model_summary['overall_score'] = statistics.mean(all_scores)
            model_summary['score_variance'] = statistics.variance(all_scores) if len(all_scores) > 1 else 0.0
        
        # Identify strengths and weaknesses
        model_summary['strengths'], model_summary['weaknesses'] = self._identify_model_patterns(model_results)
        
        return model_summary
    
    def _analyze_preset_results(self, preset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results for a specific preset"""
        preset_analysis = {
            'datasets_evaluated': [],
            'performance_metrics': {},
            'execution_metrics': {},
            'average_score': 0.0,
            'consistency_score': 0.0
        }
        
        if 'dataset_results' not in preset_data:
            return preset_analysis
        
        dataset_scores = []
        execution_times = []
        
        for dataset_name, dataset_result in preset_data['dataset_results'].items():
            preset_analysis['datasets_evaluated'].append(dataset_name)
            
            # Extract performance metrics
            metrics = self._extract_performance_metrics(dataset_result)
            if metrics:
                preset_analysis['performance_metrics'][dataset_name] = metrics
                
                # Calculate dataset score
                dataset_score = self._calculate_dataset_score(metrics)
                dataset_scores.append(dataset_score)
                
                # Collect execution times
                if 'average_execution_time' in dataset_result:
                    execution_times.append(dataset_result['average_execution_time'])
        
        # Calculate aggregate metrics
        if dataset_scores:
            preset_analysis['average_score'] = statistics.mean(dataset_scores)
            preset_analysis['consistency_score'] = 1.0 - (statistics.stdev(dataset_scores) / statistics.mean(dataset_scores)) if len(dataset_scores) > 1 and statistics.mean(dataset_scores) > 0 else 1.0
        
        if execution_times:
            preset_analysis['execution_metrics'] = {
                'average_time': statistics.mean(execution_times),
                'median_time': statistics.median(execution_times),
                'time_variance': statistics.variance(execution_times) if len(execution_times) > 1 else 0.0
            }
        
        # Add performance metrics from benchmark if available
        if 'performance_metrics' in preset_data:
            benchmark = preset_data['performance_metrics']
            preset_analysis['benchmark_metrics'] = {
                'average_response_time': benchmark.get('average_response_time', 0),
                'total_tests': benchmark.get('total_tests', 0),
                'memory_estimation': benchmark.get('memory_estimation', {})
            }
        
        return preset_analysis
    
    def _extract_performance_metrics(self, dataset_result: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract numerical performance metrics from dataset result"""
        if 'error' in dataset_result:
            return None
        
        if 'evaluation_metrics' not in dataset_result:
            return None
        
        metrics = dataset_result['evaluation_metrics']
        
        # Convert metrics to numerical values
        numerical_metrics = {}
        
        # Standard metrics
        for metric in ['accuracy', 'consistency', 'relevance', 'coherence', 'completeness']:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (int, float)):
                    numerical_metrics[metric] = float(value)
                elif isinstance(value, dict) and 'score' in value:
                    numerical_metrics[metric] = float(value['score'])
        
        # Special handling for pass_rate
        if 'pass_rate' in metrics:
            numerical_metrics['accuracy'] = float(metrics['pass_rate'])
        
        # Extract BLEU scores
        if 'bleu_score' in metrics:
            numerical_metrics['bleu'] = float(metrics['bleu_score'])
        
        # Extract ROUGE scores
        if 'rouge_scores' in metrics and isinstance(metrics['rouge_scores'], dict):
            for rouge_type, score in metrics['rouge_scores'].items():
                if isinstance(score, (int, float)):
                    numerical_metrics[f'rouge_{rouge_type}'] = float(score)
        
        return numerical_metrics if numerical_metrics else None
    
    def _calculate_dataset_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted score for a dataset"""
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            weight = self.metric_weights.get(metric, 0.1)  # Default weight for unknown metrics
            score += value * weight
            total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_preset_score(self, preset_data: Dict[str, Any]) -> float:
        """Calculate overall score for a preset"""
        if 'dataset_results' not in preset_data:
            return 0.0
        
        scores = []
        for dataset_result in preset_data['dataset_results'].values():
            metrics = self._extract_performance_metrics(dataset_result)
            if metrics:
                score = self._calculate_dataset_score(metrics)
                scores.append(score)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _identify_model_patterns(self, model_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses of a model"""
        strengths = []
        weaknesses = []
        
        # Analyze across presets and datasets
        all_metrics = {}
        
        for preset, preset_data in model_results.items():
            if 'dataset_results' not in preset_data:
                continue
                
            for dataset, dataset_result in preset_data['dataset_results'].items():
                metrics = self._extract_performance_metrics(dataset_result)
                if metrics:
                    for metric, value in metrics.items():
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(value)
        
        # Identify patterns
        for metric, values in all_metrics.items():
            if values:
                avg_value = statistics.mean(values)
                if avg_value > 0.8:
                    strengths.append(f"Strong {metric} performance (avg: {avg_value:.3f})")
                elif avg_value < 0.4:
                    weaknesses.append(f"Weak {metric} performance (avg: {avg_value:.3f})")
        
        # Check for consistency
        for metric, values in all_metrics.items():
            if len(values) > 1:
                consistency = 1.0 - (statistics.stdev(values) / statistics.mean(values)) if statistics.mean(values) > 0 else 0.0
                if consistency > 0.9:
                    strengths.append(f"Consistent {metric} across evaluations")
                elif consistency < 0.6:
                    weaknesses.append(f"Inconsistent {metric} across evaluations")
        
        return strengths[:5], weaknesses[:5]  # Limit to top 5
    
    def _analyze_dataset_performance(self, dataset_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance across datasets"""
        analysis = {}
        
        for dataset, scores in dataset_scores.items():
            if not scores:
                continue
                
            dataset_analysis = {
                'models_tested': len(scores),
                'average_performance': {},
                'best_performer': None,
                'performance_spread': {}
            }
            
            # Calculate average performance per metric
            all_metrics = set()
            for model_scores in scores.values():
                if isinstance(model_scores, dict):
                    all_metrics.update(model_scores.keys())
            
            for metric in all_metrics:
                values = []
                for model_scores in scores.values():
                    if isinstance(model_scores, dict) and metric in model_scores:
                        values.append(model_scores[metric])
                
                if values:
                    dataset_analysis['average_performance'][metric] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values)
                    }
            
            # Find best performer
            if scores:
                best_model = max(scores.keys(), key=lambda x: self._calculate_dataset_score(scores[x]) if isinstance(scores[x], dict) else 0)
                best_score = self._calculate_dataset_score(scores[best_model]) if isinstance(scores[best_model], dict) else 0
                dataset_analysis['best_performer'] = {
                    'model': best_model,
                    'score': best_score
                }
            
            analysis[dataset] = dataset_analysis
        
        return analysis
    
    def _analyze_preset_performance(self, preset_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance across presets"""
        if not preset_scores:
            return {}
        
        # Group by preset type
        preset_groups = {}
        for preset_key, score in preset_scores.items():
            parts = preset_key.split('_')
            if len(parts) >= 2:
                preset_type = parts[-1]  # Last part is preset name
                if preset_type not in preset_groups:
                    preset_groups[preset_type] = []
                preset_groups[preset_type].append(score)
        
        analysis = {}
        for preset_type, scores in preset_groups.items():
            analysis[preset_type] = {
                'average_score': statistics.mean(scores),
                'median_score': statistics.median(scores),
                'score_range': max(scores) - min(scores),
                'models_tested': len(scores)
            }
        
        return analysis
    
    def _generate_rankings(self, model_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate model rankings"""
        if not model_scores:
            return {}
        
        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings = {
            'ranking_order': [model for model, score in sorted_models],
            'scores': dict(sorted_models),
            'performance_tiers': {}
        }
        
        # Create performance tiers
        scores = [score for _, score in sorted_models]
        if scores:
            score_range = max(scores) - min(scores)
            high_threshold = max(scores) - score_range * 0.25
            low_threshold = min(scores) + score_range * 0.25
            
            rankings['performance_tiers'] = {
                'top_tier': [model for model, score in sorted_models if score >= high_threshold],
                'middle_tier': [model for model, score in sorted_models if low_threshold <= score < high_threshold],
                'lower_tier': [model for model, score in sorted_models if score < low_threshold]
            }
        
        return rankings
    
    def generate_model_comparison(self, results: Dict[str, Any], preset: str, datasets: List[str]) -> Dict[str, Any]:
        """Generate detailed comparison between models"""
        comparison = {
            'preset': preset,
            'datasets': datasets,
            'models_compared': list(results.keys()),
            'head_to_head': {},
            'dataset_winners': {},
            'overall_comparison': {},
            'statistical_analysis': {}
        }
        
        # Extract scores for comparison
        model_dataset_scores = {}
        for model_name, model_results in results.items():
            if preset in model_results and 'dataset_results' in model_results[preset]:
                model_dataset_scores[model_name] = {}
                for dataset in datasets:
                    if dataset in model_results[preset]['dataset_results']:
                        metrics = self._extract_performance_metrics(
                            model_results[preset]['dataset_results'][dataset]
                        )
                        if metrics:
                            model_dataset_scores[model_name][dataset] = self._calculate_dataset_score(metrics)
        
        # Head-to-head comparisons
        models = list(model_dataset_scores.keys())
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                wins_model1 = 0
                wins_model2 = 0
                ties = 0
                
                for dataset in datasets:
                    score1 = model_dataset_scores.get(model1, {}).get(dataset, 0)
                    score2 = model_dataset_scores.get(model2, {}).get(dataset, 0)
                    
                    if abs(score1 - score2) < 0.01:  # Tie threshold
                        ties += 1
                    elif score1 > score2:
                        wins_model1 += 1
                    else:
                        wins_model2 += 1
                
                comparison['head_to_head'][f"{model1}_vs_{model2}"] = {
                    'wins_model1': wins_model1,
                    'wins_model2': wins_model2,
                    'ties': ties,
                    'winner': model1 if wins_model1 > wins_model2 else model2 if wins_model2 > wins_model1 else 'tie'
                }
        
        # Dataset winners
        for dataset in datasets:
            dataset_scores = {}
            for model in models:
                score = model_dataset_scores.get(model, {}).get(dataset, 0)
                dataset_scores[model] = score
            
            if dataset_scores:
                winner = max(dataset_scores, key=dataset_scores.get)
                comparison['dataset_winners'][dataset] = {
                    'winner': winner,
                    'score': dataset_scores[winner],
                    'all_scores': dataset_scores
                }
        
        # Overall comparison
        overall_scores = {}
        for model in models:
            scores = [model_dataset_scores.get(model, {}).get(dataset, 0) for dataset in datasets]
            if scores:
                overall_scores[model] = statistics.mean(scores)
        
        if overall_scores:
            comparison['overall_comparison'] = {
                'scores': overall_scores,
                'ranking': sorted(overall_scores.keys(), key=overall_scores.get, reverse=True),
                'winner': max(overall_scores, key=overall_scores.get)
            }
        
        return comparison
    
    def generate_preset_comparison(self, model_results: Dict[str, Any], datasets: List[str]) -> Dict[str, Any]:
        """Generate comparison between presets for a single model"""
        comparison = {
            'model': 'model_name',  # Will be filled by caller
            'presets_compared': list(model_results.keys()),
            'datasets': datasets,
            'preset_performance': {},
            'dataset_analysis': {},
            'recommendation': {}
        }
        
        # Analyze each preset
        preset_scores = {}
        for preset, preset_data in model_results.items():
            if 'dataset_results' not in preset_data:
                continue
            
            preset_analysis = {
                'dataset_scores': {},
                'average_score': 0.0,
                'consistency': 0.0,
                'execution_time': 0.0
            }
            
            scores = []
            exec_times = []
            
            for dataset in datasets:
                if dataset in preset_data['dataset_results']:
                    result = preset_data['dataset_results'][dataset]
                    metrics = self._extract_performance_metrics(result)
                    
                    if metrics:
                        score = self._calculate_dataset_score(metrics)
                        preset_analysis['dataset_scores'][dataset] = score
                        scores.append(score)
                    
                    if 'average_execution_time' in result:
                        exec_times.append(result['average_execution_time'])
            
            if scores:
                preset_analysis['average_score'] = statistics.mean(scores)
                preset_analysis['consistency'] = 1.0 - (statistics.stdev(scores) / statistics.mean(scores)) if len(scores) > 1 and statistics.mean(scores) > 0 else 1.0
            
            if exec_times:
                preset_analysis['execution_time'] = statistics.mean(exec_times)
            
            comparison['preset_performance'][preset] = preset_analysis
            preset_scores[preset] = preset_analysis['average_score']
        
        # Generate recommendation
        if preset_scores:
            best_performance = max(preset_scores, key=preset_scores.get)
            fastest_preset = min(comparison['preset_performance'].keys(), 
                               key=lambda x: comparison['preset_performance'][x]['execution_time'])
            
            comparison['recommendation'] = {
                'best_performance': best_performance,
                'fastest_execution': fastest_preset,
                'balanced_choice': self._recommend_balanced_preset(comparison['preset_performance'])
            }
        
        return comparison
    
    def _recommend_balanced_preset(self, preset_performance: Dict[str, Dict]) -> str:
        """Recommend balanced preset considering performance and speed"""
        if not preset_performance:
            return 'balanced'
        
        # Score each preset on performance/speed balance
        balanced_scores = {}
        
        for preset, perf in preset_performance.items():
            score = perf.get('average_score', 0)
            exec_time = perf.get('execution_time', float('inf'))
            
            # Normalize execution time (lower is better)
            max_time = max(p.get('execution_time', 0) for p in preset_performance.values())
            normalized_time = 1 - (exec_time / max_time) if max_time > 0 else 1
            
            # Balanced score: 70% performance, 30% speed
            balanced_scores[preset] = 0.7 * score + 0.3 * normalized_time
        
        return max(balanced_scores, key=balanced_scores.get)