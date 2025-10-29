#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Neuroplasticity Engine for AURA
Implements Hebbian learning and synaptic plasticity for dynamic expert connections
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass


@dataclass
class PlasticityConfig:
    """Configuration for neuroplasticity mechanisms."""
    hebbian_rate: float = 0.01
    decay_rate: float = 0.001
    homeostatic_target: float = 0.1
    homeostatic_rate: float = 0.001
    consolidation_threshold: float = 0.8
    pruning_threshold: float = 0.01


class HebbianLearning:
    """Implements Hebbian learning: 'Neurons that fire together, wire together'."""
    
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.connection_strengths = {}
        self.activity_history = {}
        
    def update_connections(self, 
                         pre_activity: jnp.ndarray, 
                         post_activity: jnp.ndarray,
                         connection_id: str) -> jnp.ndarray:
        """
        Update connection strengths based on correlated activity.
        
        Args:
            pre_activity: Pre-synaptic activity [batch, pre_neurons]
            post_activity: Post-synaptic activity [batch, post_neurons] 
            connection_id: Unique identifier for this connection
            
        Returns:
            Updated connection strength matrix [pre_neurons, post_neurons]
        """
        # Initialize connection matrix if not exists
        if connection_id not in self.connection_strengths:
            pre_size, post_size = pre_activity.shape[-1], post_activity.shape[-1]
            self.connection_strengths[connection_id] = jnp.ones((pre_size, post_size)) * 0.1
        
        current_weights = self.connection_strengths[connection_id]
        
        # Compute correlation-based weight updates
        # Average over batch dimension
        pre_mean = jnp.mean(pre_activity, axis=0, keepdims=True)  # [1, pre_neurons]
        post_mean = jnp.mean(post_activity, axis=0, keepdims=True)  # [1, post_neurons]
        
        # Hebbian update: ΔW = η * pre * post
        correlation = jnp.outer(pre_mean.squeeze(), post_mean.squeeze())
        hebbian_update = self.config.hebbian_rate * correlation
        
        # Weight decay to prevent runaway growth
        decay = self.config.decay_rate * current_weights
        
        # Update weights
        new_weights = current_weights + hebbian_update - decay
        new_weights = jnp.clip(new_weights, 0.0, 2.0)  # Prevent negative/excessive weights
        
        self.connection_strengths[connection_id] = new_weights
        return new_weights
    
    def get_connection_strength(self, connection_id: str) -> Optional[jnp.ndarray]:
        """Get current connection strength matrix."""
        return self.connection_strengths.get(connection_id)


class HomeostaticRegulation:
    """Implements homeostatic plasticity for maintaining optimal activity levels."""
    
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.target_activities = {}
        self.scaling_factors = {}
    
    def regulate_activity(self, 
                         activity: jnp.ndarray,
                         neuron_group_id: str) -> jnp.ndarray:
        """
        Apply homeostatic scaling to maintain target activity levels.
        
        Args:
            activity: Neural activity [batch, neurons]
            neuron_group_id: Identifier for this group of neurons
            
        Returns:
            Regulated activity with homeostatic scaling applied
        """
        # Track average activity over time
        current_avg = jnp.mean(activity)
        
        if neuron_group_id not in self.target_activities:
            self.target_activities[neuron_group_id] = current_avg
            self.scaling_factors[neuron_group_id] = 1.0
        
        # Compute deviation from target
        target = self.config.homeostatic_target
        deviation = target - current_avg
        
        # Update scaling factor
        current_scaling = self.scaling_factors[neuron_group_id]
        new_scaling = current_scaling + self.config.homeostatic_rate * deviation
        new_scaling = jnp.clip(new_scaling, 0.1, 10.0)  # Prevent extreme scaling
        
        self.scaling_factors[neuron_group_id] = float(new_scaling)
        
        # Apply scaling
        return activity * new_scaling


class SynapticConsolidation:
    """Consolidates important synaptic connections for long-term memory."""
    
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.consolidation_scores = {}
        self.consolidated_connections = set()
    
    def update_consolidation_scores(self, 
                                  connection_id: str,
                                  connection_strength: jnp.ndarray,
                                  activity_correlation: float):
        """Update consolidation scores based on usage and strength."""
        if connection_id not in self.consolidation_scores:
            self.consolidation_scores[connection_id] = 0.0
        
        # Increase score based on strength and correlation
        strength_score = jnp.mean(connection_strength)
        correlation_score = activity_correlation
        
        update = 0.1 * (strength_score + correlation_score)
        self.consolidation_scores[connection_id] += update
        
        # Decay score over time
        self.consolidation_scores[connection_id] *= 0.99
    
    def should_consolidate(self, connection_id: str) -> bool:
        """Check if connection should be consolidated."""
        score = self.consolidation_scores.get(connection_id, 0.0)
        return score > self.config.consolidation_threshold
    
    def consolidate_connection(self, connection_id: str):
        """Mark connection as consolidated (protected from pruning)."""
        self.consolidated_connections.add(connection_id)


class NeuroplasticityEngine:
    """
    Main neuroplasticity engine coordinating all plasticity mechanisms.
    Simulates biological neural plasticity in expert networks.
    """
    
    def __init__(self, config: Optional[PlasticityConfig] = None):
        self.config = config or PlasticityConfig()
        self.hebbian_learning = HebbianLearning(self.config)
        self.homeostatic_regulation = HomeostaticRegulation(self.config)
        self.synaptic_consolidation = SynapticConsolidation(self.config)
        
        # Track expert interactions
        self.expert_interactions = {}
        self.plasticity_history = []
    
    def update_expert_connections(self, 
                                expert_activities: Dict[str, jnp.ndarray],
                                expert_rewards: Dict[str, float]) -> Dict[str, jnp.ndarray]:
        """
        Update connections between experts based on their activities and rewards.
        
        Args:
            expert_activities: Dict mapping expert_id to activity [batch, features]
            expert_rewards: Dict mapping expert_id to reward signal
            
        Returns:
            Updated connection strengths between experts
        """
        updated_connections = {}
        expert_ids = list(expert_activities.keys())
        
        # Update pairwise connections between experts
        for i, expert_a in enumerate(expert_ids):
            for j, expert_b in enumerate(expert_ids[i+1:], i+1):
                connection_id = f"{expert_a}_to_{expert_b}"
                
                activity_a = expert_activities[expert_a]
                activity_b = expert_activities[expert_b]
                
                # Apply homeostatic regulation first
                regulated_a = self.homeostatic_regulation.regulate_activity(
                    activity_a, expert_a
                )
                regulated_b = self.homeostatic_regulation.regulate_activity(
                    activity_b, expert_b
                )
                
                # Update Hebbian connections
                connection_strength = self.hebbian_learning.update_connections(
                    regulated_a, regulated_b, connection_id
                )
                
                # Modulate by reward signals
                reward_factor = (expert_rewards.get(expert_a, 0.0) + 
                               expert_rewards.get(expert_b, 0.0)) / 2.0
                reward_modulated_strength = connection_strength * (1.0 + reward_factor)
                
                updated_connections[connection_id] = reward_modulated_strength
                
                # Update consolidation scores
                correlation = self._compute_activity_correlation(regulated_a, regulated_b)
                self.synaptic_consolidation.update_consolidation_scores(
                    connection_id, reward_modulated_strength, correlation
                )
        
        # Record plasticity event
        self.plasticity_history.append({
            'step': len(self.plasticity_history),
            'expert_count': len(expert_ids),
            'connection_count': len(updated_connections),
            'avg_connection_strength': float(jnp.mean(jnp.array([
                jnp.mean(conn) for conn in updated_connections.values()
            ])))
        })
        
        return updated_connections
    
    def _compute_activity_correlation(self, 
                                    activity_a: jnp.ndarray, 
                                    activity_b: jnp.ndarray) -> float:
        """Compute correlation between two activity patterns."""
        # Flatten activities and compute correlation
        flat_a = activity_a.flatten()
        flat_b = activity_b.flatten()
        
        # Pearson correlation coefficient
        corr_matrix = jnp.corrcoef(flat_a, flat_b)
        correlation = float(corr_matrix[0, 1])
        
        # Handle NaN case (constant activities)
        if jnp.isnan(correlation):
            correlation = 0.0
            
        return correlation
    
    def adapt_learning_rates(self, 
                           performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt learning rates based on performance feedback.
        Implements meta-plasticity: plasticity of plasticity.
        """
        adapted_rates = {}
        
        # Extract key performance indicators
        accuracy = performance_metrics.get('accuracy', 0.5)
        loss = performance_metrics.get('loss', 1.0)
        expert_utilization = performance_metrics.get('expert_utilization', 0.5)
        
        # Compute adaptation signals
        performance_signal = accuracy - 0.5  # Target 50% baseline
        stability_signal = 1.0 / (1.0 + loss)  # Higher for lower loss
        diversity_signal = expert_utilization  # Encourage expert diversity
        
        # Adapt Hebbian learning rate
        hebbian_adaptation = 0.1 * (performance_signal + diversity_signal)
        new_hebbian_rate = self.config.hebbian_rate * (1.0 + hebbian_adaptation)
        new_hebbian_rate = jnp.clip(new_hebbian_rate, 0.001, 0.1)
        
        # Adapt homeostatic rate
        homeostatic_adaptation = 0.1 * stability_signal
        new_homeostatic_rate = self.config.homeostatic_rate * (1.0 + homeostatic_adaptation)
        new_homeostatic_rate = jnp.clip(new_homeostatic_rate, 0.0001, 0.01)
        
        adapted_rates = {
            'hebbian_rate': float(new_hebbian_rate),
            'homeostatic_rate': float(new_homeostatic_rate),
            'decay_rate': self.config.decay_rate,
            'performance_signal': performance_signal,
            'stability_signal': stability_signal,
            'diversity_signal': diversity_signal
        }
        
        return adapted_rates
    
    def prune_weak_connections(self) -> List[str]:
        """Remove weak connections to maintain network efficiency."""
        pruned_connections = []
        
        for connection_id, strength_matrix in self.hebbian_learning.connection_strengths.items():
            # Skip consolidated connections
            if connection_id in self.synaptic_consolidation.consolidated_connections:
                continue
            
            avg_strength = jnp.mean(strength_matrix)
            if avg_strength < self.config.pruning_threshold:
                pruned_connections.append(connection_id)
        
        # Remove pruned connections
        for connection_id in pruned_connections:
            del self.hebbian_learning.connection_strengths[connection_id]
            if connection_id in self.synaptic_consolidation.consolidation_scores:
                del self.synaptic_consolidation.consolidation_scores[connection_id]
        
        return pruned_connections
    
    def get_plasticity_state(self) -> Dict[str, Any]:
        """Get current state of all plasticity mechanisms."""
        return {
            'connection_count': len(self.hebbian_learning.connection_strengths),
            'consolidated_count': len(self.synaptic_consolidation.consolidated_connections),
            'avg_connection_strength': float(jnp.mean(jnp.array([
                jnp.mean(conn) for conn in self.hebbian_learning.connection_strengths.values()
            ]))) if self.hebbian_learning.connection_strengths else 0.0,
            'plasticity_events': len(self.plasticity_history),
            'config': {
                'hebbian_rate': self.config.hebbian_rate,
                'decay_rate': self.config.decay_rate,
                'homeostatic_target': self.config.homeostatic_target,
                'consolidation_threshold': self.config.consolidation_threshold
            }
        }
    
    def reset_plasticity(self):
        """Reset all plasticity mechanisms to initial state."""
        self.hebbian_learning.connection_strengths.clear()
        self.hebbian_learning.activity_history.clear()
        self.homeostatic_regulation.target_activities.clear()
        self.homeostatic_regulation.scaling_factors.clear()
        self.synaptic_consolidation.consolidation_scores.clear()
        self.synaptic_consolidation.consolidated_connections.clear()
        self.plasticity_history.clear()


# Integration with AURA expert system
class PlasticExpertCore(nn.Module):
    """Expert core with neuroplasticity integration."""
    
    hidden_dim: int
    num_experts: int
    plasticity_config: PlasticityConfig = None
    
    def setup(self):
        from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore
        
        # Base retrieval core
        self.base_core = EnhancedSpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts
        )
        
        # Plasticity engine
        config = self.plasticity_config or PlasticityConfig()
        self.plasticity_engine = NeuroplasticityEngine(config)
        
        # Plastic connection weights
        self.plastic_weights = self.param('plastic_weights',
            lambda rng, shape: jax.random.normal(rng, shape) * 0.1,
            (self.num_experts, self.num_experts)
        )
    
    def __call__(self, 
                 query_embedding: jnp.ndarray,
                 expert_rewards: Optional[Dict[str, float]] = None,
                 update_plasticity: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass with plasticity updates.
        
        Returns:
            output: Processed query embedding
            plasticity_info: Information about plasticity updates
        """
        # Get base expert outputs
        base_output = self.base_core(query_embedding)
        expert_outputs = self.base_core.expert_outputs(query_embedding)
        gate_weights = self.base_core.compute_gate_weights(query_embedding)
        
        plasticity_info = {}
        
        if update_plasticity and expert_rewards is not None:
            # Create expert activity dictionary
            expert_activities = {}
            for i in range(self.num_experts):
                expert_activities[f'expert_{i}'] = expert_outputs[:, i, :]
            
            # Update plastic connections
            updated_connections = self.plasticity_engine.update_expert_connections(
                expert_activities, expert_rewards
            )
            
            # Apply plastic modulation to gate weights
            plastic_modulation = jnp.ones_like(gate_weights)
            for connection_id, strength in updated_connections.items():
                if 'expert_' in connection_id:
                    # Extract expert indices and apply modulation
                    parts = connection_id.split('_to_')
                    if len(parts) == 2:
                        idx_a = int(parts[0].split('_')[-1])
                        idx_b = int(parts[1].split('_')[-1])
                        connection_strength = jnp.mean(strength)
                        plastic_modulation = plastic_modulation.at[:, idx_b].multiply(
                            1.0 + 0.1 * connection_strength
                        )
            
            # Apply plastic modulation
            modulated_weights = gate_weights * plastic_modulation
            modulated_weights = modulated_weights / jnp.sum(modulated_weights, axis=-1, keepdims=True)
            
            # Recompute output with plastic weights
            plastic_output = jnp.einsum('bn,bnh->bh', modulated_weights, expert_outputs)
            
            plasticity_info = {
                'plastic_connections': len(updated_connections),
                'plasticity_state': self.plasticity_engine.get_plasticity_state(),
                'gate_modulation': jnp.mean(plastic_modulation)
            }
            
            return plastic_output, plasticity_info
        
        return base_output, plasticity_info


if __name__ == "__main__":
    # Example usage
    config = PlasticityConfig(
        hebbian_rate=0.01,
        homeostatic_target=0.1,
        consolidation_threshold=0.8
    )
    
    engine = NeuroplasticityEngine(config)
    
    # Simulate expert activities
    key = jax.random.key(0)
    expert_activities = {
        'expert_0': jax.random.normal(key, (8, 64)),
        'expert_1': jax.random.normal(key, (8, 64)),
        'expert_2': jax.random.normal(key, (8, 64))
    }
    
    expert_rewards = {'expert_0': 0.8, 'expert_1': 0.6, 'expert_2': 0.9}
    
    # Update connections
    connections = engine.update_expert_connections(expert_activities, expert_rewards)
    
    print("🧠 Neuroplasticity Engine Demo:")
    print(f"   Updated {len(connections)} connections")
    print(f"   Plasticity state: {engine.get_plasticity_state()}")
