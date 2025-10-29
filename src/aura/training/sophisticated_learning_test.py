#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Sophisticated learning test for AURA bio-inspired components
Uses clearer patterns and runs until at least 80% accuracy is achieved
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


class SophisticatedBioInspiredModel(nn.Module):
    """Model for sophisticated learning test"""
    hidden_dim: int = 256
    vocab_size: int = 1000
    embed_dim: int = 128
    num_classes: int = 3  # Multi-class classification
    
    def setup(self):
        # Initialize bio-inspired components
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=64)
        self.spiking_attention = SpikingAttentionJAX(k_winners=20, decay=0.8, theta=1.2)
        self.retrieval_core = EnhancedSpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=16,
            expert_dim=128,
            phasor_harmonics=64
        )
        # Projection layers
        self.input_projection = nn.Dense(self.hidden_dim)
        self.hidden_layers = [nn.Dense(self.hidden_dim) for _ in range(3)]
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
        
        # Apply attention mechanism
        # Convert context to token-like sequence for attention
        token_indices = jnp.argsort(jnp.abs(context_vector[0, :]))[-100:] % self.vocab_size
        attention_gains = self.spiking_attention(token_indices, self.vocab_size)
        
        # Apply attention to context (simplified)
        attended_context = context_vector * attention_gains[:context_vector.shape[-1]]
        
        # Additional hidden layers
        hidden = attended_context
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            hidden = nn.relu(hidden)
        
        # Output projection
        logits = self.output_projection(hidden)
        return logits


class SophisticatedLearningTest:
    """Sophisticated learning test with clearer patterns"""
    
    def __init__(self, model_hidden_dim=256, vocab_size=1000, embed_dim=128, num_classes=3):
        self.model_hidden_dim = model_hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.model = SophisticatedBioInspiredModel(
            hidden_dim=model_hidden_dim, 
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes
        )
        
    def create_sample_data(self, batch_size=64):
        """Create sample training data with clear multi-class patterns"""
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, batch_size + 1)
        
        # Create input embeddings with clear patterns for 3 classes
        x = jax.random.normal(subkeys[0], (batch_size, self.embed_dim))
        targets = jnp.zeros(batch_size, dtype=jnp.int32)
        
        # Create clearer patterns for each class
        # Class 0: First 10 dimensions have positive bias
        class0_indicator = jnp.mean(x[:, :10], axis=-1)
        # Class 1: Middle 10 dimensions have positive bias
        class1_indicator = jnp.mean(x[:, 10:20], axis=-1)
        # Class 2: Last 10 dimensions have positive bias
        class2_indicator = jnp.mean(x[:, 20:30], axis=-1)
        
        # Assign classes based on which indicator is highest
        class_indicators = jnp.stack([class0_indicator, class1_indicator, class2_indicator], axis=-1)
        targets = jnp.argmax(class_indicators, axis=-1)
        
        return x, targets
    
    def create_train_state(self, rng):
        """Create train state for the model"""
        # Create sample input to initialize model
        sample_x, sample_targets = self.create_sample_data(batch_size=8)
        
        # Initialize model parameters
        params = self.model.init(rng, sample_x)
        
        # Create optimizer with learning rate scheduling
        optimizer = optax.adam(learning_rate=3e-4)
        
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
    
    def run_sophisticated_learning_test(self, target_accuracy=0.80, max_epochs=2000):
        """Run sophisticated learning test until target accuracy is reached"""
        logger.info(f"Starting sophisticated learning test - target accuracy: {target_accuracy*100:.1f}%")
        
        # Create random key
        key = jax.random.PRNGKey(0)
        subkey = jax.random.split(key, 1)[0]
        
        # Create train state
        state = self.create_train_state(subkey)
        
        # Training loop
        epoch = 0
        best_accuracy = 0.0
        patience_counter = 0
        max_patience = 100  # Stop if no improvement for 100 epochs
        
        while epoch < max_epochs:
            # Create sample data
            x, targets = self.create_sample_data(batch_size=64)
            
            # Training step
            state, loss = self.train_step(state, x, targets)
            
            # Compute accuracy every 10 epochs
            if epoch % 10 == 0:
                # Use larger batch for accuracy evaluation
                eval_x, eval_targets = self.create_sample_data(batch_size=200)
                accuracy = self.compute_accuracy(state, eval_x, eval_targets)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                    # Log significant improvements
                    if accuracy > best_accuracy + 0.05:
                        logger.info(f"🎯 Significant improvement! Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
                else:
                    patience_counter += 1
                
                # Log every 50 epochs or when accuracy is high
                if epoch % 50 == 0 or accuracy >= 0.75:
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
        
        logger.info(f"Sophisticated learning test completed. Best accuracy: {best_accuracy*100:.2f}%")
        
        if best_accuracy >= target_accuracy:
            logger.info(f"✅ Model successfully reached target accuracy of {target_accuracy*100:.1f}%")
            return True
        else:
            logger.info(f"❌ Model did not reach target accuracy. Best was {best_accuracy*100:.2f}%")
            return False


def main():
    """Main function to run sophisticated learning test"""
    # Create learning test
    learning_test = SophisticatedLearningTest(model_hidden_dim=256, vocab_size=1000, embed_dim=128, num_classes=3)
    
    # Run test until 80% accuracy
    success = learning_test.run_sophisticated_learning_test(target_accuracy=0.80, max_epochs=2000)
    
    if success:
        logger.info("Sophisticated learning test PASSED - 80% accuracy achieved")
        return True
    else:
        logger.error("Sophisticated learning test FAILED - 80% accuracy not achieved")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
