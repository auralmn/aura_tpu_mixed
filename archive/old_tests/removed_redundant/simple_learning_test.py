#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Simple learning test for AURA bio-inspired components
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


class SimpleBioInspiredModel(nn.Module):
    """Simple model combining bio-inspired components for learning test"""
    hidden_dim: int = 64
    vocab_size: int = 1000
    
    def setup(self):
        # Initialize bio-inspired components
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=32)
        self.spiking_attention = SpikingAttentionJAX(k_winners=5, decay=0.7, theta=1.0)
        self.retrieval_core = EnhancedSpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=8,
            expert_dim=32,
            phasor_harmonics=32
        )
        self.output_layer = nn.Dense(self.vocab_size)
    
    def __call__(self, x):
        # Extract temporal features using phasor bank
        # For this test, we'll use the mean of input as scalar for phasor bank
        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean.squeeze(-1))
        
        # Combine original input with temporal features
        enhanced_x = jnp.concatenate([x, temporal_features], axis=-1)
        
        # Process through retrieval core
        context_vector = self.retrieval_core(enhanced_x)
        
        # Apply spiking attention (we need to convert to token sequence)
        # For simplicity, we'll use indices of highest values as tokens
        token_seq = jnp.argsort(context_vector[0, :])[-20:]  # Top 20 values as tokens
        attention_gains = self.spiking_attention(token_seq, self.vocab_size)
        
        # Apply attention gains to context vector
        # We'll reshape attention gains to match context vector dimensions
        attention_gains_reshaped = attention_gains[:context_vector.shape[-1]]
        attended_context = context_vector * attention_gains_reshaped
        
        # Output layer
        logits = self.output_layer(attended_context)
        return logits


class SimpleLearningTest:
    """Simple learning test to verify bio-inspired components can learn"""
    
    def __init__(self, model_hidden_dim=64, vocab_size=1000):
        self.model_hidden_dim = model_hidden_dim
        self.vocab_size = vocab_size
        self.model = SimpleBioInspiredModel(hidden_dim=model_hidden_dim, vocab_size=vocab_size)
        
    def create_sample_data(self, batch_size=8):
        """Create sample training data"""
        key = jax.random.PRNGKey(0)
        # Create input embeddings
        x = jax.random.normal(key, (batch_size, 32))  # 32-dimensional input
        
        # Create target labels (simple classification task)
        targets = jax.random.randint(key, (batch_size,), 0, self.vocab_size)
        
        return x, targets
    
    def create_train_state(self, rng):
        """Create train state for the model"""
        # Create sample input to initialize model
        sample_x = jax.random.normal(rng, (1, 32))
        
        # Initialize model parameters
        params = self.model.init(rng, sample_x)
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        
        # Create train state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
        return state
    
    def compute_loss(self, params, x, targets):
        """Compute loss for the model"""
        logits = self.model.apply(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss
    
    def train_step(self, state, x, targets):
        """Single training step"""
        # Compute gradients
        loss, grads = jax.value_and_grad(self.compute_loss)(state.params, x, targets)
        
        # Apply gradients
        state = state.apply_gradients(grads=grads)
        
        return state, loss
    
    def run_learning_test(self, epochs=10):
        """Run simple learning test"""
        logger.info("Starting simple learning test for bio-inspired components")
        
        # Create random key
        key = jax.random.PRNGKey(0)
        
        # Create train state
        state = self.create_train_state(key)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            # Create sample data
            x, targets = self.create_sample_data(batch_size=8)
            
            # Training step
            state, loss = self.train_step(state, x, targets)
            losses.append(loss.item())
            
            # Log progress
            if epoch % 2 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        # Final evaluation
        x, targets = self.create_sample_data(batch_size=8)
        final_loss = self.compute_loss(state.params, x, targets)
        
        logger.info(f"Learning test completed. Initial loss: {losses[0]:.4f}, Final loss: {final_loss:.4f}")
        
        # Check if model learned something (loss decreased)
        if final_loss < losses[0]:
            logger.info("✅ Model successfully learned - loss decreased")
            return True
        else:
            logger.info("⚠️ Model may not have learned - loss did not decrease significantly")
            return False


def main():
    """Main function to run simple learning test"""
    # Create learning test
    learning_test = SimpleLearningTest(model_hidden_dim=64, vocab_size=1000)
    
    # Run test
    success = learning_test.run_learning_test(epochs=20)
    
    if success:
        logger.info("Simple learning test PASSED")
        return True
    else:
        logger.error("Simple learning test FAILED")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
