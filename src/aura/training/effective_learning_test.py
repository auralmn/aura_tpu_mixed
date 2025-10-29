#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Effective learning test for AURA bio-inspired components
Focuses on binary classification with clear patterns
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


class EffectiveBioInspiredModel(nn.Module):
    """Model for effective learning test"""
    hidden_dim: int = 128
    vocab_size: int = 1000
    embed_dim: int = 64
    num_classes: int = 2  # Binary classification
    
    def setup(self):
        # Initialize bio-inspired components
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=32)
        self.spiking_attention = SpikingAttentionJAX(k_winners=10, decay=0.7, theta=1.0)
        self.retrieval_core = EnhancedSpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=8,
            expert_dim=64,
            phasor_harmonics=32
        )
        # Projection layers
        self.input_projection = nn.Dense(self.hidden_dim)
        self.output_projection = nn.Dense(self.num_classes)
    
    def __call__(self, x):
        # Project input to hidden dimension
        projected_x = self.input_projection(x)
        
        # Extract temporal features using phasor bank
        x_mean = jnp.mean(projected_x, axis=-1, keepdims=True)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean.squeeze(-1))
        
        # Combine original input with temporal features
        enhanced_x = jnp.concatenate([projected_x, temporal_features], axis=-1)
        
        # Process through retrieval core
        context_vector = self.retrieval_core(enhanced_x)
        
        # Simple output projection (bypassing attention for clarity in this test)
        logits = self.output_projection(context_vector)
        return logits


class EffectiveLearningTest:
    """Effective learning test with clear binary patterns"""
    
    def __init__(self, model_hidden_dim=128, vocab_size=1000, embed_dim=64, num_classes=2):
        self.model_hidden_dim = model_hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.model = EffectiveBioInspiredModel(
            hidden_dim=model_hidden_dim, 
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes
        )
        
    def create_sample_data(self, batch_size=32):
        """Create sample training data with clear binary classification pattern"""
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        
        # Create input embeddings
        x = jax.random.normal(subkeys[0], (batch_size, self.embed_dim))
        
        # Create clear binary target pattern
        # Target = 1 if mean of first half of dimensions > mean of second half, else 0
        first_half_mean = jnp.mean(x[:, :self.embed_dim//2], axis=-1)
        second_half_mean = jnp.mean(x[:, self.embed_dim//2:], axis=-1)
        targets = (first_half_mean > second_half_mean).astype(jnp.int32)
        
        return x, targets
    
    def create_train_state(self, rng):
        """Create train state for the model"""
        # Create sample input to initialize model
        sample_x, sample_targets = self.create_sample_data(batch_size=4)
        
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
    
    @staticmethod
    @jax.jit
    def compute_accuracy(state, x, targets):
        """Compute accuracy for current model state"""
        logits = state.apply_fn(state.params, x)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == targets)
        return accuracy
    
    def run_effective_learning_test(self, target_accuracy=0.95, max_epochs=1000):
        """Run effective learning test until target accuracy is reached"""
        logger.info(f"Starting effective learning test - target accuracy: {target_accuracy*100:.1f}%")
        
        # Create random key
        key = jax.random.PRNGKey(0)
        subkey = jax.random.split(key, 1)[0]
        
        # Create train state
        state = self.create_train_state(subkey)
        
        # Training loop
        epoch = 0
        best_accuracy = 0.0
        patience_counter = 0
        max_patience = 50  # Stop if no improvement for 50 epochs
        
        while epoch < max_epochs:
            # Create sample data
            x, targets = self.create_sample_data(batch_size=32)
            
            # Training step
            state, loss = self.train_step(state, x, targets)
            
            # Compute accuracy every 5 epochs
            if epoch % 5 == 0:
                # Use larger batch for accuracy evaluation
                eval_x, eval_targets = self.create_sample_data(batch_size=100)
                accuracy = self.compute_accuracy(state, eval_x, eval_targets)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}, Best = {best_accuracy:.4f}")
                
                # Check if target accuracy reached
                if accuracy >= target_accuracy:
                    logger.info(f"🎯 Target accuracy {target_accuracy*100:.1f}% reached at epoch {epoch}")
                    return True
                
                # Check for early stopping
                if patience_counter >= max_patience:
                    logger.info(f"⚠️ No improvement for {max_patience} epochs, stopping early")
                    break
            
            epoch += 1
        
        logger.info(f"Effective learning test completed. Best accuracy: {best_accuracy*100:.2f}%")
        
        if best_accuracy >= target_accuracy:
            logger.info(f"✅ Model successfully reached target accuracy of {target_accuracy*100:.1f}%")
            return True
        else:
            logger.info(f"❌ Model did not reach target accuracy. Best was {best_accuracy*100:.2f}%")
            return False


def main():
    """Main function to run effective learning test"""
    # Create learning test
    learning_test = EffectiveLearningTest(model_hidden_dim=128, vocab_size=1000, embed_dim=64, num_classes=2)
    
    # Run test until 95% accuracy
    success = learning_test.run_effective_learning_test(target_accuracy=0.95, max_epochs=1000)
    
    if success:
        logger.info("Effective learning test PASSED - 95% accuracy achieved")
        return True
    else:
        logger.error("Effective learning test FAILED - 95% accuracy not achieved")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
