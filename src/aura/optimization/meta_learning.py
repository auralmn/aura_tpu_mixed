#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Meta-Learning System for AURA
Implements MAML-style meta-learning for few-shot expert adaptation
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning system."""
    inner_learning_rate: float = 0.01
    outer_learning_rate: float = 0.001
    inner_steps: int = 5
    meta_batch_size: int = 16
    support_shots: int = 5
    query_shots: int = 10
    adaptation_layers: List[str] = None  # Which layers to adapt
    
    def __post_init__(self):
        if self.adaptation_layers is None:
            self.adaptation_layers = ["all"]  # Adapt all layers by default


class MAMLNetwork(nn.Module):
    """
    Model-Agnostic Meta-Learning network.
    Can quickly adapt to new tasks with few gradient steps.
    """
    
    hidden_dims: List[int]
    output_dim: int
    activation: str = "relu"
    
    def setup(self):
        self.layers = []
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            layer = nn.Dense(hidden_dim, name=f"hidden_{i}")
            self.layers.append(layer)
        
        self.output_layer = nn.Dense(self.output_dim, name="output")
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        current = x
        
        for layer in self.layers:
            current = layer(current)
            
            if self.activation == "relu":
                current = nn.relu(current)
            elif self.activation == "gelu":
                current = nn.gelu(current)
            elif self.activation == "tanh":
                current = nn.tanh(current)
            elif self.activation == "swish":
                current = nn.swish(current)
        
        return self.output_layer(current)


class MetaExpert(nn.Module):
    """
    Expert that can quickly adapt to new tasks using meta-learning.
    Combines a base expert with fast adaptation capabilities.
    """
    
    base_hidden_dim: int
    output_dim: int
    meta_config: MetaLearningConfig
    
    def setup(self):
        # Base expert network
        self.base_expert = MAMLNetwork(
            hidden_dims=[self.base_hidden_dim, self.base_hidden_dim // 2],
            output_dim=self.output_dim
        )
        
        # Adaptation network for computing fast weights
        self.adaptation_network = nn.Dense(self.base_hidden_dim, name="adaptation")
        
        # Context encoder for task representation
        self.context_encoder = nn.Dense(64, name="context_encoder")
        
        # Meta-learning specific parameters
        self.task_embedding = nn.Dense(32, name="task_embedding")
    
    def __call__(self, 
                 x: jnp.ndarray,
                 task_context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Forward pass through meta-expert.
        
        Args:
            x: Input data
            task_context: Optional task context for adaptation
        """
        if task_context is not None:
            # Encode task context
            task_features = self.context_encoder(task_context)
            task_embedding = self.task_embedding(task_features)
            
            # Modulate input with task embedding
            x_modulated = x + jnp.broadcast_to(task_embedding, x.shape)
            
            return self.base_expert(x_modulated)
        else:
            return self.base_expert(x)
    
    def fast_adapt(self, 
                   params: Dict[str, Any],
                   support_x: jnp.ndarray,
                   support_y: jnp.ndarray,
                   learning_rate: float,
                   num_steps: int) -> Dict[str, Any]:
        """
        Perform fast adaptation on support set.
        
        Args:
            params: Current model parameters
            support_x: Support set inputs
            support_y: Support set targets
            learning_rate: Inner loop learning rate
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted parameters
        """
        adapted_params = params
        
        for step in range(num_steps):
            # Compute gradients on support set
            def support_loss_fn(p):
                predictions = self.apply(p, support_x)
                return jnp.mean((predictions - support_y) ** 2)
            
            grads = jax.grad(support_loss_fn)(adapted_params)
            
            # Update parameters using gradient descent
            adapted_params = jax.tree_map(
                lambda p, g: p - learning_rate * g,
                adapted_params, grads
            )
        
        return adapted_params


class PrototypicalNetwork(nn.Module):
    """
    Prototypical network for few-shot learning.
    Learns representations where classes form clusters.
    """
    
    embedding_dim: int
    hidden_dims: List[int] = None
    
    def setup(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
        
        self.encoder_layers = []
        current_dim = self.embedding_dim
        
        for hidden_dim in self.hidden_dims:
            self.encoder_layers.append(nn.Dense(hidden_dim))
            current_dim = hidden_dim
        
        self.final_embedding = nn.Dense(self.embedding_dim)
    
    def embed(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode input to embedding space."""
        current = x
        
        for layer in self.encoder_layers:
            current = layer(current)
            current = nn.relu(current)
        
        return self.final_embedding(current)
    
    def compute_prototypes(self, 
                          support_embeddings: jnp.ndarray,
                          support_labels: jnp.ndarray) -> jnp.ndarray:
        """
        Compute class prototypes from support set.
        
        Args:
            support_embeddings: [N_support, embedding_dim]
            support_labels: [N_support]
            
        Returns:
            prototypes: [N_classes, embedding_dim]
        """
        unique_labels = jnp.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            class_embeddings = support_embeddings[mask]
            prototype = jnp.mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        return jnp.stack(prototypes)
    
    def __call__(self, 
                 query_x: jnp.ndarray,
                 support_x: jnp.ndarray,
                 support_y: jnp.ndarray) -> jnp.ndarray:
        """
        Classify query examples using prototypical network.
        
        Args:
            query_x: Query set inputs [N_query, input_dim]
            support_x: Support set inputs [N_support, input_dim] 
            support_y: Support set labels [N_support]
            
        Returns:
            logits: Classification logits [N_query, N_classes]
        """
        # Embed all examples
        query_embeddings = self.embed(query_x)
        support_embeddings = self.embed(support_x)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_y)
        
        # Compute distances to prototypes
        distances = jnp.linalg.norm(
            query_embeddings[:, None, :] - prototypes[None, :, :],
            axis=2
        )
        
        # Convert distances to logits (negative distance)
        logits = -distances
        
        return logits


class MetaLearningEngine:
    """
    Main meta-learning engine coordinating different meta-learning approaches.
    """
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.meta_optimizer = optax.adam(config.outer_learning_rate)
        self.training_history = []
        
    def create_meta_expert(self, 
                          input_dim: int,
                          output_dim: int,
                          expert_type: str = "maml") -> nn.Module:
        """Create a meta-learning expert of specified type."""
        
        if expert_type == "maml":
            return MetaExpert(
                base_hidden_dim=128,
                output_dim=output_dim,
                meta_config=self.config
            )
        elif expert_type == "prototypical":
            return PrototypicalNetwork(
                embedding_dim=64,
                hidden_dims=[128, 64]
            )
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
    
    def meta_train_step(self,
                       meta_expert: nn.Module,
                       params: Dict[str, Any],
                       opt_state: Any,
                       meta_batch: List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]) -> Tuple[Dict[str, Any], Any, Dict[str, float]]:
        """
        Perform one meta-training step using MAML.
        
        Args:
            meta_expert: Meta-learning expert
            params: Current parameters
            opt_state: Optimizer state
            meta_batch: List of (support_x, support_y, query_x, query_y) tuples
            
        Returns:
            Updated params, opt_state, and metrics
        """
        
        def meta_loss_fn(p):
            total_loss = 0.0
            
            for support_x, support_y, query_x, query_y in meta_batch:
                # Fast adaptation on support set
                if hasattr(meta_expert, 'fast_adapt'):
                    adapted_params = meta_expert.fast_adapt(
                        p, support_x, support_y,
                        self.config.inner_learning_rate,
                        self.config.inner_steps
                    )
                else:
                    adapted_params = p
                
                # Evaluate on query set
                query_predictions = meta_expert.apply(adapted_params, query_x)
                query_loss = jnp.mean((query_predictions - query_y) ** 2)
                total_loss += query_loss
            
            return total_loss / len(meta_batch)
        
        # Compute meta-gradients
        meta_loss, meta_grads = jax.value_and_grad(meta_loss_fn)(params)
        
        # Update meta-parameters
        updates, new_opt_state = self.meta_optimizer.update(meta_grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        metrics = {
            'meta_loss': float(meta_loss),
            'grad_norm': float(jnp.linalg.norm(
                jnp.concatenate([jnp.ravel(g) for g in jax.tree_leaves(meta_grads)])
            ))
        }
        
        return new_params, new_opt_state, metrics
    
    def few_shot_evaluate(self,
                         meta_expert: nn.Module,
                         params: Dict[str, Any],
                         support_x: jnp.ndarray,
                         support_y: jnp.ndarray,
                         query_x: jnp.ndarray,
                         query_y: jnp.ndarray) -> Dict[str, float]:
        """
        Evaluate meta-expert on a few-shot task.
        
        Returns:
            Evaluation metrics
        """
        
        if hasattr(meta_expert, 'fast_adapt'):
            # MAML-style adaptation
            adapted_params = meta_expert.fast_adapt(
                params, support_x, support_y,
                self.config.inner_learning_rate,
                self.config.inner_steps
            )
            
            # Evaluate adapted model
            query_predictions = meta_expert.apply(adapted_params, query_x)
            
        else:
            # Prototypical network or other non-parametric methods
            query_predictions = meta_expert.apply(params, query_x, support_x, support_y)
        
        # Compute metrics
        if len(query_predictions.shape) > 1 and query_predictions.shape[-1] > 1:
            # Classification
            predicted_classes = jnp.argmax(query_predictions, axis=-1)
            true_classes = jnp.argmax(query_y, axis=-1) if len(query_y.shape) > 1 else query_y
            accuracy = jnp.mean(predicted_classes == true_classes)
            
            metrics = {
                'accuracy': float(accuracy),
                'loss': float(jnp.mean((query_predictions - query_y) ** 2))
            }
        else:
            # Regression
            mse = jnp.mean((query_predictions - query_y) ** 2)
            mae = jnp.mean(jnp.abs(query_predictions - query_y))
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae)
            }
        
        return metrics
    
    def create_meta_training_state(self, 
                                  meta_expert: nn.Module,
                                  input_shape: Tuple[int, ...]) -> train_state.TrainState:
        """Create training state for meta-learning."""
        
        key = jax.random.key(0)
        dummy_input = jnp.ones((1,) + input_shape)
        
        params = meta_expert.init(key, dummy_input)
        opt_state = self.meta_optimizer.init(params)
        
        return train_state.TrainState.create(
            apply_fn=meta_expert.apply,
            params=params,
            tx=self.meta_optimizer
        )
    
    def generate_few_shot_task(self,
                              data_x: jnp.ndarray,
                              data_y: jnp.ndarray,
                              n_classes: int,
                              support_shots: int,
                              query_shots: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate a few-shot learning task from data.
        
        Returns:
            support_x, support_y, query_x, query_y
        """
        
        # Sample classes for this task
        unique_classes = jnp.unique(data_y)
        selected_classes = jax.random.choice(
            jax.random.key(0), unique_classes, (n_classes,), replace=False
        )
        
        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []
        
        for i, class_label in enumerate(selected_classes):
            # Get all examples of this class
            class_mask = data_y == class_label
            class_data = data_x[class_mask]
            
            # Sample support and query examples
            n_available = class_data.shape[0]
            total_needed = support_shots + query_shots
            
            if n_available >= total_needed:
                indices = jax.random.choice(
                    jax.random.key(int(class_label)), 
                    n_available, (total_needed,), replace=False
                )
                
                support_indices = indices[:support_shots]
                query_indices = indices[support_shots:support_shots + query_shots]
                
                support_x_list.append(class_data[support_indices])
                support_y_list.append(jnp.full((support_shots,), i))  # Relabel to 0, 1, 2, ...
                
                query_x_list.append(class_data[query_indices])
                query_y_list.append(jnp.full((query_shots,), i))
        
        # Concatenate all examples
        support_x = jnp.concatenate(support_x_list, axis=0)
        support_y = jnp.concatenate(support_y_list, axis=0)
        query_x = jnp.concatenate(query_x_list, axis=0)
        query_y = jnp.concatenate(query_y_list, axis=0)
        
        # Shuffle within support and query sets
        support_perm = jax.random.permutation(jax.random.key(1), support_x.shape[0])
        query_perm = jax.random.permutation(jax.random.key(2), query_x.shape[0])
        
        support_x = support_x[support_perm]
        support_y = support_y[support_perm]
        query_x = query_x[query_perm]
        query_y = query_y[query_perm]
        
        return support_x, support_y, query_x, query_y


class ContinualLearningExpert(nn.Module):
    """
    Expert that can learn new tasks without forgetting old ones.
    Implements elastic weight consolidation (EWC) and other continual learning techniques.
    """
    
    hidden_dim: int
    output_dim: int
    ewc_lambda: float = 1000.0
    
    def setup(self):
        self.encoder = nn.Dense(self.hidden_dim)
        self.hidden_layer = nn.Dense(self.hidden_dim)
        self.output_layer = nn.Dense(self.output_dim)
        
        # Fisher information matrix for EWC
        self.fisher_information = {}
        self.optimal_params = {}
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through continual learning expert."""
        x = self.encoder(x)
        x = nn.relu(x)
        x = self.hidden_layer(x)
        x = nn.relu(x)
        return self.output_layer(x)
    
    def compute_fisher_information(self,
                                 params: Dict[str, Any],
                                 data_x: jnp.ndarray,
                                 data_y: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute Fisher information matrix for EWC."""
        
        def log_likelihood_fn(p):
            predictions = self.apply(p, data_x)
            # Assume Gaussian likelihood for simplicity
            return -jnp.sum((predictions - data_y) ** 2)
        
        # Compute gradients of log-likelihood
        grad_fn = jax.grad(log_likelihood_fn)
        gradients = grad_fn(params)
        
        # Fisher information is expectation of squared gradients
        fisher_info = jax.tree_map(lambda g: g ** 2, gradients)
        
        return fisher_info
    
    def ewc_loss(self,
                params: Dict[str, Any],
                data_x: jnp.ndarray,
                data_y: jnp.ndarray) -> float:
        """Compute EWC regularization loss."""
        
        # Standard task loss
        predictions = self.apply(params, data_x)
        task_loss = jnp.mean((predictions - data_y) ** 2)
        
        # EWC regularization
        ewc_loss = 0.0
        if self.fisher_information and self.optimal_params:
            for param_name in params:
                if param_name in self.fisher_information and param_name in self.optimal_params:
                    param_diff = params[param_name] - self.optimal_params[param_name]
                    fisher_diag = self.fisher_information[param_name]
                    ewc_loss += jnp.sum(fisher_diag * param_diff ** 2)
        
        total_loss = task_loss + self.ewc_lambda * ewc_loss
        return total_loss


if __name__ == "__main__":
    # Example meta-learning setup
    config = MetaLearningConfig(
        inner_learning_rate=0.01,
        outer_learning_rate=0.001,
        inner_steps=5,
        support_shots=5,
        query_shots=10
    )
    
    engine = MetaLearningEngine(config)
    
    # Create meta-expert
    meta_expert = engine.create_meta_expert(
        input_dim=32,
        output_dim=3,
        expert_type="maml"
    )
    
    # Create training state
    training_state = engine.create_meta_training_state(meta_expert, (32,))
    
    # Generate sample few-shot task
    key = jax.random.key(0)
    sample_data_x = jax.random.normal(key, (1000, 32))
    sample_data_y = jax.random.randint(key, (1000,), 0, 10)  # 10 classes
    
    support_x, support_y, query_x, query_y = engine.generate_few_shot_task(
        sample_data_x, sample_data_y, n_classes=3, 
        support_shots=5, query_shots=10
    )
    
    # Evaluate few-shot performance
    metrics = engine.few_shot_evaluate(
        meta_expert, training_state.params,
        support_x, support_y, query_x, query_y
    )
    
    print("🧠 Meta-Learning System Demo:")
    print(f"   Support set: {support_x.shape}, Query set: {query_x.shape}")
    print(f"   Few-shot performance: {metrics}")
    print(f"   Configuration: {config.support_shots}-shot, {config.query_shots}-query")
