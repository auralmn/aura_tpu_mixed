#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Actual learning test for AURA bio-inspired components
"""

import os
import sys
import json
import logging
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import bio-inspired components
from aura.bio_inspired.phasor_bank import PhasorBankJAX
from aura.bio_inspired.spiking_attention import SpikingAttentionJAX
from aura.bio_inspired.thalamic_router import ThalamicGradientBroadcasterJAX
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore


class ActualBioInspiredModel(nn.Module):
    """Model with actual learning using bio-inspired components"""
    hidden_dim: int = 64
    vocab_size: int = 1000
    embed_dim: int = 32
    
    def setup(self):
        # Initialize bio-inspired components
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=10)
        self.spiking_attention = SpikingAttentionJAX(k_winners=5, decay=0.7, theta=1.0)
        self.retrieval_core = EnhancedSpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=4,
            expert_dim=16,
            phasor_harmonics=10
        )
        # Projection layers to connect components
        self.input_projection = nn.Dense(self.hidden_dim)
        self.output_projection = nn.Dense(self.vocab_size)
    
    def __call__(self, x, targets=None):
        # Project input to hidden dimension
        projected_x = self.input_projection(x)
        
        # Extract temporal features using phasor bank
        # For this test, we'll use the mean of input as scalar for phasor bank
        x_mean = jnp.mean(projected_x, axis=-1, keepdims=True)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean.squeeze(-1))
        
        # Combine original input with temporal features
        enhanced_x = jnp.concatenate([projected_x, temporal_features], axis=-1)
        
        # Process through retrieval core
        context_vector = self.retrieval_core(enhanced_x)
        
        # Apply attention mechanism
        # Convert context to token-like sequence for attention
        token_indices = jnp.argsort(jnp.abs(context_vector[0, :]))[-20:] % self.vocab_size
        attention_gains = self.spiking_attention(token_indices, self.vocab_size)
        
        # Apply attention to context (simplified)
        attended_context = context_vector * attention_gains[:context_vector.shape[-1]]
        
        # Output projection
        logits = self.output_projection(attended_context)
        
        if targets is not None:
            # Compute loss if targets provided
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss
        else:
            return logits


class ActualLearningTest:
    """Actual learning test with parameter updates"""
    
    def __init__(self, model_hidden_dim=64, vocab_size=1000, embed_dim=32):
        self.model_hidden_dim = model_hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.model = ActualBioInspiredModel(
            hidden_dim=model_hidden_dim, 
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
    def create_sample_data(self, batch_size=8):
        """Create sample training data with clear patterns"""
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 3)
        
        # Create input embeddings with clear patterns
        x = jax.random.normal(subkeys[0], (batch_size, self.embed_dim))
        
        # Create target labels with a simple pattern
        # Targets based on the sign of the first few input dimensions
        targets = jnp.sum(x[:, :5], axis=-1) > 0
        targets = targets.astype(jnp.int32) * (self.vocab_size // 2) + jnp.arange(batch_size) % (self.vocab_size // 2)
        targets = jnp.clip(targets, 0, self.vocab_size - 1)
        
        return x, targets
    
    def create_train_state(self, rng):
        """Create train state for the model"""
        # Create sample input to initialize model
        sample_x, sample_targets = self.create_sample_data(batch_size=2)
        
        # Initialize model parameters
        params = self.model.init(rng, sample_x, sample_targets)
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        
        # Create train state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
        return state
    
    @staticmethod
    @jax.jit
    def train_step(state, x, targets):
        """Single training step with JIT compilation"""
        def loss_fn(params):
            logits = state.apply_fn(params, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        
        # Apply gradients
        state = state.apply_gradients(grads=grads)
        
        return state, loss
    
    def run_learning_test(self, epochs=50):
        """Run actual learning test"""
        logger.info("Starting actual learning test for bio-inspired components")
        
        # Create random key
        key = jax.random.PRNGKey(0)
        subkey = jax.random.split(key, 1)[0]
        
        # Create train state
        state = self.create_train_state(subkey)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            # Create sample data
            x, targets = self.create_sample_data(batch_size=8)
            
            # Training step
            state, loss = self.train_step(state, x, targets)
            losses.append(loss.item())
            
            # Log progress every 10 epochs
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        # Final evaluation
        x, targets = self.create_sample_data(batch_size=8)
        final_loss = self.model.apply(state.params, x, targets)
        
        logger.info(f"Learning test completed. Initial loss: {losses[0]:.4f}, Final loss: {final_loss:.4f}")
        
        # Check if model learned something (loss decreased significantly)
        loss_decrease = losses[0] - final_loss
        if loss_decrease > 0.01:  # At least 0.01 decrease
            logger.info(f"✅ Model successfully learned - loss decreased by {loss_decrease:.4f}")
            return True
        else:
            logger.info("⚠️ Model may not have learned - loss did not decrease significantly")
            return False


def main():
    """Main function to run actual learning test"""
    # Create learning test
    learning_test = ActualLearningTest(model_hidden_dim=64, vocab_size=1000, embed_dim=32)
    
    # Run test
    success = learning_test.run_learning_test(epochs=50)
    
    if success:
        logger.info("Actual learning test PASSED")
        return True
    else:
        logger.error("Actual learning test FAILED")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
