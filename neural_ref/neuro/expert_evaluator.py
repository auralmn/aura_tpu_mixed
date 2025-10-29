#!/usr/bin/env python3
"""
AURA Expert Evaluation System
Multi-dimensional framework for evaluating expert quality in Liquid-MoE systems

Based on composite utility scoring with:
- Validation accuracy/loss
- Routing confidence & usage frequency  
- Marginal contribution (Shapley-style)
- Calibration & uncertainty
- Diversity and specialization
- Computational cost
- Lifespan and stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, deque
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExpertMetrics:
    """Container for expert evaluation metrics"""
    validation_loss: float = 0.0
    routing_confidence: float = 0.0
    usage_frequency: float = 0.0
    marginal_contribution: float = 0.0
    calibration_error: float = 0.0
    diversity_score: float = 0.0
    computational_cost: float = 0.0
    performance_variance: float = 0.0
    utility_score: float = 0.0


class ExpertEvaluator:
    """
    Multi-dimensional expert evaluation system for Liquid-MoE
    
    Implements composite utility scoring based on:
    1. Validation Accuracy (or Loss)
    2. Routing Confidence & Usage Frequency
    3. Marginal Contribution (Shapley-style)
    4. Calibration & Uncertainty
    5. Diversity and Specialization
    6. Computational Cost
    7. Lifespan and Stability
    """
    
    def __init__(self, 
                 num_experts: int,
                 evaluation_window: int = 100,
                 utility_weights: Optional[Dict[str, float]] = None):
        """
        Initialize expert evaluator
        
        Args:
            num_experts: Number of experts in the MoE system
            evaluation_window: Number of evaluations to keep for rolling averages
            utility_weights: Weights for composite utility score
        """
        self.num_experts = num_experts
        self.evaluation_window = evaluation_window
        
        # Default utility weights (tunable based on system priorities)
        self.utility_weights = utility_weights or {
            'performance': 0.3,      # w1: validation accuracy
            'cost': 0.2,             # w2: computational cost (negative)
            'marginal': 0.2,         # w3: marginal contribution
            'diversity': 0.15,       # w4: diversity score
            'calibration': 0.15      # w5: calibration error (negative)
        }
        
        # Expert metrics history
        self.expert_metrics = defaultdict(lambda: deque(maxlen=evaluation_window))
        self.routing_history = defaultdict(lambda: deque(maxlen=evaluation_window))
        self.performance_history = defaultdict(lambda: deque(maxlen=evaluation_window))
        
        # Current expert states
        self.current_metrics = {i: ExpertMetrics() for i in range(num_experts)}
        self.expert_usage_counts = defaultdict(int)
        self.total_evaluations = 0
        
        # Performance tracking
        self.ensemble_performance_history = deque(maxlen=evaluation_window)
        self.expert_ablation_results = {}
        
        logger.info(f"ExpertEvaluator initialized for {num_experts} experts")
    
    def update_routing_confidence(self, expert_id: int, gating_weights: torch.Tensor, 
                                selected_experts: torch.Tensor):
        """
        Update routing confidence metrics for an expert
        
        Args:
            expert_id: Expert identifier
            gating_weights: Softmax gating weights [batch, num_experts]
            selected_experts: Binary mask of selected experts [batch, num_experts]
        """
        if expert_id >= self.num_experts:
            return
            
        # Calculate confidence when this expert is selected
        expert_mask = selected_experts[:, expert_id].bool()
        if expert_mask.any():
            expert_confidence = gating_weights[expert_mask, expert_id].mean().item()
            self.routing_history[expert_id].append(expert_confidence)
            
            # Update usage frequency
            usage_freq = expert_mask.float().mean().item()
            self.expert_usage_counts[expert_id] += expert_mask.sum().item()
            
            # Update current metrics
            self.current_metrics[expert_id].routing_confidence = np.mean(self.routing_history[expert_id])
            self.current_metrics[expert_id].usage_frequency = usage_freq
    
    def update_validation_performance(self, expert_id: int, loss: float, accuracy: float = None):
        """
        Update validation performance metrics
        
        Args:
            expert_id: Expert identifier
            loss: Validation loss
            accuracy: Validation accuracy (optional)
        """
        if expert_id >= self.num_experts:
            return
            
        # Store performance history
        self.performance_history[expert_id].append({
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': time.time()
        })
        
        # Update current metrics
        recent_losses = [p['loss'] for p in self.performance_history[expert_id]]
        self.current_metrics[expert_id].validation_loss = np.mean(recent_losses)
        
        # Calculate performance variance
        if len(recent_losses) > 1:
            self.current_metrics[expert_id].performance_variance = np.var(recent_losses)
    
    def calculate_marginal_contribution(self, expert_id: int, 
                                      ensemble_performance: float,
                                      ablation_performance: float):
        """
        Calculate marginal contribution using Shapley-style ablation
        
        Args:
            expert_id: Expert identifier
            ensemble_performance: Performance with all experts
            ablation_performance: Performance without this expert
        """
        if expert_id >= self.num_experts:
            return
            
        # Marginal contribution = performance drop when expert is removed
        marginal_contrib = ensemble_performance - ablation_performance
        self.current_metrics[expert_id].marginal_contribution = marginal_contrib
        
        # Store for ensemble tracking
        self.ensemble_performance_history.append(ensemble_performance)
        self.expert_ablation_results[expert_id] = ablation_performance
    
    def calculate_calibration_error(self, expert_id: int, 
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor):
        """
        Calculate calibration error (Expected Calibration Error)
        
        Args:
            expert_id: Expert identifier
            predictions: Expert predictions [batch, num_classes]
            targets: Ground truth targets [batch]
        """
        if expert_id >= self.num_experts:
            return
            
        try:
            # Convert to probabilities
            probs = F.softmax(predictions, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]
            predictions_class = torch.argmax(probs, dim=-1)
            
            # Calculate accuracy for each confidence bin
            n_bins = 10
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.float().mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = (predictions_class[in_bin] == targets[in_bin]).float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            self.current_metrics[expert_id].calibration_error = ece.item()
            
        except Exception as e:
            logger.warning(f"Calibration error calculation failed for expert {expert_id}: {e}")
            self.current_metrics[expert_id].calibration_error = 1.0  # Worst case
    
    def calculate_diversity_score(self, expert_id: int, 
                                expert_outputs: Dict[int, torch.Tensor]):
        """
        Calculate diversity score based on KL divergence with other experts
        
        Args:
            expert_id: Expert identifier
            expert_outputs: Dictionary of expert outputs {expert_id: output_tensor}
        """
        if expert_id >= self.num_experts or expert_id not in expert_outputs:
            return
            
        try:
            current_expert_output = expert_outputs[expert_id]
            kl_divergences = []
            
            for other_expert_id, other_output in expert_outputs.items():
                if other_expert_id != expert_id:
                    # Calculate KL divergence
                    p = F.softmax(current_expert_output, dim=-1)
                    q = F.softmax(other_output, dim=-1)
                    
                    # Add small epsilon to avoid log(0)
                    p = torch.clamp(p, min=1e-8)
                    q = torch.clamp(q, min=1e-8)
                    
                    kl_div = F.kl_div(p.log(), q, reduction='batchmean')
                    kl_divergences.append(kl_div.item())
            
            if kl_divergences:
                # Higher KL divergence = more diverse
                diversity_score = np.mean(kl_divergences)
                self.current_metrics[expert_id].diversity_score = diversity_score
            else:
                self.current_metrics[expert_id].diversity_score = 0.0
                
        except Exception as e:
            logger.warning(f"Diversity calculation failed for expert {expert_id}: {e}")
            self.current_metrics[expert_id].diversity_score = 0.0
    
    def calculate_computational_cost(self, expert_id: int, 
                                   parameter_count: int,
                                   flops: float,
                                   latency: float):
        """
        Calculate computational cost metrics
        
        Args:
            expert_id: Expert identifier
            parameter_count: Number of parameters
            flops: Floating point operations per forward pass
            latency: Inference latency in seconds
        """
        if expert_id >= self.num_experts:
            return
            
        # Normalize costs (higher = more expensive)
        # This is a simplified cost model - can be enhanced based on specific hardware
        param_cost = parameter_count / 1e6  # Normalize by 1M parameters
        flop_cost = flops / 1e9  # Normalize by 1G FLOPs
        latency_cost = latency * 1000  # Convert to ms
        
        # Composite cost score
        cost_score = 0.4 * param_cost + 0.4 * flop_cost + 0.2 * latency_cost
        self.current_metrics[expert_id].computational_cost = cost_score
    
    def calculate_utility_score(self, expert_id: int) -> float:
        """
        Calculate composite utility score for an expert
        
        Formula: U_i = w1×Perf_i - w2×Cost_i + w3×Marginal_i + w4×Diversity_i - w5×CalibError_i
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            Utility score (higher = better expert)
        """
        if expert_id >= self.num_experts:
            return 0.0
            
        metrics = self.current_metrics[expert_id]
        weights = self.utility_weights
        
        # Normalize metrics to [0, 1] range
        # Performance: convert loss to accuracy (lower loss = higher performance)
        perf_score = max(0, 1.0 - metrics.validation_loss)
        
        # Cost: normalize by max cost across all experts
        max_cost = max((m.computational_cost for m in self.current_metrics.values()), default=1.0)
        cost_score = metrics.computational_cost / max_cost if max_cost > 0 else 0.0
        
        # Marginal contribution: normalize by max marginal contribution
        max_marginal = max((m.marginal_contribution for m in self.current_metrics.values()), default=1.0)
        marginal_score = metrics.marginal_contribution / max_marginal if max_marginal > 0 else 0.0
        
        # Diversity: normalize by max diversity
        max_diversity = max((m.diversity_score for m in self.current_metrics.values()), default=1.0)
        diversity_score = metrics.diversity_score / max_diversity if max_diversity > 0 else 0.0
        
        # Calibration error: normalize by max calibration error
        max_calib = max((m.calibration_error for m in self.current_metrics.values()), default=1.0)
        calib_score = metrics.calibration_error / max_calib if max_calib > 0 else 0.0
        
        # Calculate composite utility score
        utility_score = (
            weights['performance'] * perf_score -
            weights['cost'] * cost_score +
            weights['marginal'] * marginal_score +
            weights['diversity'] * diversity_score -
            weights['calibration'] * calib_score
        )
        
        # Store and return
        self.current_metrics[expert_id].utility_score = utility_score
        return utility_score
    
    def get_expert_rankings(self) -> List[Tuple[int, float]]:
        """
        Get experts ranked by utility score
        
        Returns:
            List of (expert_id, utility_score) tuples sorted by utility (descending)
        """
        rankings = []
        for expert_id in range(self.num_experts):
            utility_score = self.calculate_utility_score(expert_id)
            rankings.append((expert_id, utility_score))
        
        # Sort by utility score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_pruning_candidates(self, threshold: float = 0.3) -> List[int]:
        """
        Get experts that should be considered for pruning
        
        Args:
            threshold: Utility score threshold below which experts are candidates for pruning
            
        Returns:
            List of expert IDs to consider for pruning
        """
        rankings = self.get_expert_rankings()
        pruning_candidates = []
        
        for expert_id, utility_score in rankings:
            if utility_score < threshold:
                pruning_candidates.append(expert_id)
        
        return pruning_candidates
    
    def get_expert_report(self, expert_id: int) -> Dict[str, Any]:
        """
        Get detailed report for a specific expert
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            Dictionary containing all metrics for the expert
        """
        if expert_id >= self.num_experts:
            return {}
            
        metrics = self.current_metrics[expert_id]
        
        return {
            'expert_id': expert_id,
            'validation_loss': metrics.validation_loss,
            'routing_confidence': metrics.routing_confidence,
            'usage_frequency': metrics.usage_frequency,
            'marginal_contribution': metrics.marginal_contribution,
            'calibration_error': metrics.calibration_error,
            'diversity_score': metrics.diversity_score,
            'computational_cost': metrics.computational_cost,
            'performance_variance': metrics.performance_variance,
            'utility_score': metrics.utility_score,
            'total_usage': self.expert_usage_counts[expert_id],
            'evaluation_count': len(self.performance_history[expert_id])
        }
    
    def get_system_report(self) -> Dict[str, Any]:
        """
        Get comprehensive system report
        
        Returns:
            Dictionary containing system-wide statistics
        """
        rankings = self.get_expert_rankings()
        pruning_candidates = self.get_pruning_candidates()
        
        return {
            'total_experts': self.num_experts,
            'total_evaluations': self.total_evaluations,
            'expert_rankings': rankings,
            'pruning_candidates': pruning_candidates,
            'average_utility': np.mean([score for _, score in rankings]),
            'utility_std': np.std([score for _, score in rankings]),
            'best_expert': rankings[0] if rankings else None,
            'worst_expert': rankings[-1] if rankings else None,
            'utility_weights': self.utility_weights
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update utility score weights
        
        Args:
            new_weights: New weight configuration
        """
        self.utility_weights.update(new_weights)
        logger.info(f"Updated utility weights: {self.utility_weights}")
    
    def reset_expert(self, expert_id: int):
        """
        Reset metrics for a specific expert
        
        Args:
            expert_id: Expert identifier to reset
        """
        if expert_id < self.num_experts:
            self.expert_metrics[expert_id].clear()
            self.routing_history[expert_id].clear()
            self.performance_history[expert_id].clear()
            self.current_metrics[expert_id] = ExpertMetrics()
            self.expert_usage_counts[expert_id] = 0
            logger.info(f"Reset metrics for expert {expert_id}")


def create_expert_evaluator(num_experts: int, 
                           evaluation_window: int = 100,
                           utility_weights: Optional[Dict[str, float]] = None) -> ExpertEvaluator:
    """Factory function to create an ExpertEvaluator"""
    return ExpertEvaluator(num_experts, evaluation_window, utility_weights)


# Example usage and testing
if __name__ == "__main__":
    # Test the expert evaluator
    evaluator = ExpertEvaluator(num_experts=8)
    
    # Simulate some metrics
    for expert_id in range(8):
        evaluator.update_validation_performance(expert_id, loss=0.5 + expert_id * 0.1)
        evaluator.calculate_computational_cost(expert_id, 1000000, 1e9, 0.001)
    
    # Get rankings
    rankings = evaluator.get_expert_rankings()
    print("Expert Rankings:")
    for expert_id, score in rankings:
        print(f"  Expert {expert_id}: {score:.4f}")
    
    # Get system report
    report = evaluator.get_system_report()
    print(f"\nSystem Report:")
    print(f"  Average Utility: {report['average_utility']:.4f}")
    print(f"  Pruning Candidates: {report['pruning_candidates']}")
