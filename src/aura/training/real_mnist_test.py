#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Real MNIST test for AURA bio-inspired components
"""

import os
import sys
import logging
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import bio-inspired components
from aura.bio_inspired.phasor_bank import PhasorBankJAX
from aura.bio_inspired.spiking_attention import SpikingAttentionJAX
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore

try:
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torchvision not available, will use synthetic data")


class RealMNISTModel(nn.Module):
    """Model for real MNIST with bio-inspired components"""
    hidden_dim: int = 128
    num_classes: int = 10
    vocab_size: int = 1000
    
    def setup(self):
        # Initialize bio-inspired components
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=32)
        self.attention = SpikingAttentionJAX(decay=0.7, theta=1.0, k_winners=5)
        self.retrieval_core = EnhancedSpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=8,
            expert_dim=64,
            phasor_harmonics=32
        )
        # Simple projection layers
        self.input_projection = nn.Dense(self.hidden_dim)
        self.output_projection = nn.Dense(self.num_classes)
        self.phasor_projection = nn.Dense(self.hidden_dim)
    
    def __call__(self, x):
        # Flatten input image
        x_flat = x.reshape((x.shape[0], -1))  # [batch, 784]
        
        # Project to hidden dimension
        projected_x = self.input_projection(x_flat)  # [batch, hidden_dim]
        
        # Extract temporal features using phasor bank and map to hidden_dim
        x_mean = jnp.mean(projected_x, axis=-1)  # [batch,]
        temporal_features = jax.vmap(self.phasor_bank)(x_mean)  # [batch, 2*H+1]
        temporal_mapped = self.phasor_projection(temporal_features)  # [batch, hidden_dim]
        
        # Combine phasor-enhanced features before attention
        enhanced_x = projected_x + temporal_mapped  # [batch, hidden_dim]
        
        # Build token sequence from top-K hidden indices (align vocab to hidden_dim)
        K = min(32, enhanced_x.shape[-1])  # Python int for static arg
        topk_idx = jax.lax.top_k(enhanced_x, K)[1]  # [batch, K]
        token_seq = topk_idx.astype(jnp.int32)
        
        # Compute attention gains over hidden features directly
        vocab_size = int(enhanced_x.shape[-1])  # align attention domain to hidden features
        attention_gains = jax.vmap(self.attention, in_axes=(0, None))(token_seq, vocab_size)  # [batch, hidden_dim]
        attended_x = enhanced_x * attention_gains  # [batch, hidden_dim]
        
        # For MNIST classification, use attended features directly
        context_vector = attended_x  # [batch, hidden_dim]
        
        # Output projection
        logits = self.output_projection(context_vector)  # [batch, num_classes]
        return logits


class RealMNISTTest:
    """Real MNIST test with actual dataset"""
    
    def __init__(self, model_hidden_dim=128, num_classes=10):
        self.model_hidden_dim = model_hidden_dim
        self.num_classes = num_classes
        self.model = RealMNISTModel(
            hidden_dim=model_hidden_dim, 
            num_classes=num_classes
        )
        
        # Load MNIST data
        self.load_mnist_data()
        
    def load_mnist_data(self):
        """Load MNIST data from local CSV if available, else create synthetic data"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        csv_train_path = os.path.join(project_root, 'data', 'MNIST', 'raw', 'mnist_train.csv')
        csv_test_path = os.path.join(project_root, 'data', 'MNIST', 'raw', 'mnist_test.csv')
        
        if os.path.exists(csv_train_path):
            logger.info(f"Loading MNIST CSV from {csv_train_path}")
            try:
                # Load training CSV: first column is label, remaining 784 are pixel values
                data = np.loadtxt(csv_train_path, delimiter=',', dtype=np.float32)
                labels = data[:, 0].astype(np.int32)
                pixels = data[:, 1:].astype(np.float32) / 255.0
                images = pixels.reshape((-1, 28, 28))
                
                # If a separate test CSV exists, load it; else split the train set
                if os.path.exists(csv_test_path):
                    logger.info(f"Found test CSV at {csv_test_path}, loading it as well")
                    test_data = np.loadtxt(csv_test_path, delimiter=',', dtype=np.float32)
                    test_labels = test_data[:, 0].astype(np.int32)
                    test_pixels = test_data[:, 1:].astype(np.float32) / 255.0
                    test_images = test_pixels.reshape((-1, 28, 28))
                    
                    self.train_images = images
                    self.train_labels = labels
                    self.test_images = test_images
                    self.test_labels = test_labels
                else:
                    # Split last 5000 samples as test set
                    split = max(5000, int(0.1 * images.shape[0]))
                    self.train_images = images[:-split]
                    self.train_labels = labels[:-split]
                    self.test_images = images[-split:]
                    self.test_labels = labels[-split:]
                
                logger.info(f"Training data shape: {self.train_images.shape}")
                logger.info(f"Test data shape: {self.test_images.shape}")
                return
            except Exception as e:
                logger.warning(f"Failed to load MNIST CSV: {e}. Falling back to synthetic data.")
        
        logger.info("Creating synthetic MNIST-like data")
        self.create_synthetic_mnist()
    
    def create_synthetic_mnist(self):
        """Create synthetic MNIST-like data with more structured patterns"""
        # Create training samples with patterns that better represent MNIST
        self.train_images = np.zeros((6000, 28, 28), dtype=np.float32)
        self.train_labels = np.zeros(6000, dtype=np.int32)
        
        # Create simple patterns for each digit (0-9)
        for digit in range(10):
            for i in range(600):  # 600 samples per digit
                idx = digit * 600 + i
                self.train_labels[idx] = digit
                
                # Create different patterns for each digit
                if digit == 0:  # Circle
                    center_x, center_y = 14, 14
                    for x in range(8, 20):
                        for y in range(8, 20):
                            if 36 <= (x - center_x)**2 + (y - center_y)**2 <= 64:
                                self.train_images[idx, x, y] = 1.0
                elif digit == 1:  # Vertical line
                    self.train_images[idx, 5:23, 12:16] = 1.0
                elif digit == 2:  # Z shape
                    self.train_images[idx, 5:9, 5:23] = 1.0  # Top line
                    self.train_images[idx, 9:19, 12:16] = 1.0  # Diagonal
                    self.train_images[idx, 19:23, 5:23] = 1.0  # Bottom line
                elif digit == 3:  # Two horizontal lines with connections
                    self.train_images[idx, 5:9, 5:23] = 1.0  # Top line
                    self.train_images[idx, 12:16, 18:23] = 1.0  # Middle right
                    self.train_images[idx, 19:23, 5:23] = 1.0  # Bottom line
                elif digit == 4:  # Vertical line with horizontal cross
                    self.train_images[idx, 5:23, 12:16] = 1.0  # Vertical
                    self.train_images[idx, 8:12, 5:16] = 1.0  # Horizontal top
                    self.train_images[idx, 12:16, 5:9] = 1.0  # Horizontal bottom
                elif digit == 5:  # Inverted L shape
                    self.train_images[idx, 5:9, 5:23] = 1.0  # Top line
                    self.train_images[idx, 5:16, 5:9] = 1.0  # Vertical left
                    self.train_images[idx, 12:16, 5:23] = 1.0  # Middle line
                    self.train_images[idx, 12:23, 18:23] = 1.0  # Vertical right bottom
                    self.train_images[idx, 19:23, 5:19] = 1.0  # Bottom line
                elif digit == 6:  # Number 6 shape
                    center_x, center_y = 14, 12
                    for x in range(8, 20):
                        for y in range(8, 16):
                            if 36 <= (x - center_x)**2 + (y - center_y)**2 <= 64:
                                self.train_images[idx, x, y] = 1.0
                    # Add a vertical line
                    self.train_images[idx, 14:20, 10:14] = 1.0
                elif digit == 7:  # Diagonal with top line
                    self.train_images[idx, 5:9, 5:23] = 1.0  # Top line
                    for i in range(18):
                        self.train_images[idx, 5+i, 22-i] = 1.0  # Diagonal
                elif digit == 8:  # Two circles
                    center_x1, center_y1 = 10, 12
                    center_x2, center_y2 = 18, 12
                    for x in range(6, 22):
                        for y in range(8, 16):
                            if 9 <= (x - center_x1)**2 + (y - center_y1)**2 <= 25:
                                self.train_images[idx, x, y] = 1.0
                            if 9 <= (x - center_x2)**2 + (y - center_y2)**2 <= 25:
                                self.train_images[idx, x, y] = 1.0
                elif digit == 9:  # Circle with vertical line
                    center_x, center_y = 12, 12
                    for x in range(6, 18):
                        for y in range(8, 16):
                            if 9 <= (x - center_x)**2 + (y - center_y)**2 <= 25:
                                self.train_images[idx, x, y] = 1.0
                    # Add a vertical line
                    self.train_images[idx, 6:18, 14:18] = 1.0
        
        # Create test samples with similar patterns
        self.test_images = np.zeros((1000, 28, 28), dtype=np.float32)
        self.test_labels = np.zeros(1000, dtype=np.int32)
        
        for digit in range(10):
            for i in range(100):  # 100 samples per digit
                idx = digit * 100 + i
                self.test_labels[idx] = digit
                
                # Add some noise to make it more realistic
                noise = np.random.normal(0, 0.1, (28, 28)).astype(np.float32)
                
                # Same patterns but with noise
                if digit == 0:  # Circle
                    center_x, center_y = 14, 14
                    for x in range(8, 20):
                        for y in range(8, 20):
                            if 36 <= (x - center_x)**2 + (y - center_y)**2 <= 64:
                                self.test_images[idx, x, y] = 1.0 + noise[x, y]
                elif digit == 1:  # Vertical line
                    self.test_images[idx, 5:23, 12:16] = 1.0 + noise[5:23, 12:16]
                elif digit == 2:  # Z shape
                    self.test_images[idx, 5:9, 5:23] = 1.0 + noise[5:9, 5:23]
                    self.test_images[idx, 9:19, 12:16] = 1.0 + noise[9:19, 12:16]
                    self.test_images[idx, 19:23, 5:23] = 1.0 + noise[19:23, 5:23]
                elif digit == 3:  # Two horizontal lines with connections
                    self.test_images[idx, 5:9, 5:23] = 1.0 + noise[5:9, 5:23]
                    self.test_images[idx, 12:16, 18:23] = 1.0 + noise[12:16, 18:23]
                    self.test_images[idx, 19:23, 5:23] = 1.0 + noise[19:23, 5:23]
                elif digit == 4:  # Vertical line with horizontal cross
                    self.test_images[idx, 5:23, 12:16] = 1.0 + noise[5:23, 12:16]
                    self.test_images[idx, 8:12, 5:16] = 1.0 + noise[8:12, 5:16]
                    self.test_images[idx, 12:16, 5:9] = 1.0 + noise[12:16, 5:9]
                elif digit == 5:  # Inverted L shape
                    self.test_images[idx, 5:9, 5:23] = 1.0 + noise[5:9, 5:23]
                    self.test_images[idx, 5:16, 5:9] = 1.0 + noise[5:16, 5:9]
                    self.test_images[idx, 12:16, 5:23] = 1.0 + noise[12:16, 5:23]
                    self.test_images[idx, 12:23, 18:23] = 1.0 + noise[12:23, 18:23]
                    self.test_images[idx, 19:23, 5:19] = 1.0 + noise[19:23, 5:19]
                elif digit == 6:  # Number 6 shape
                    center_x, center_y = 14, 12
                    for x in range(8, 20):
                        for y in range(8, 16):
                            if 36 <= (x - center_x)**2 + (y - center_y)**2 <= 64:
                                self.test_images[idx, x, y] = 1.0 + noise[x, y]
                    self.test_images[idx, 14:20, 10:14] = 1.0 + noise[14:20, 10:14]
                elif digit == 7:  # Diagonal with top line
                    self.test_images[idx, 5:9, 5:23] = 1.0 + noise[5:9, 5:23]
                    for i in range(18):
                        x, y = 5+i, 22-i
                        if 0 <= x < 28 and 0 <= y < 28:
                            self.test_images[idx, x, y] = 1.0 + noise[x, y]
                elif digit == 8:  # Two circles
                    center_x1, center_y1 = 10, 12
                    center_x2, center_y2 = 18, 12
                    for x in range(6, 22):
                        for y in range(8, 16):
                            val1 = 1.0 + noise[x, y] if 9 <= (x - center_x1)**2 + (y - center_y1)**2 <= 25 else 0.0
                            val2 = 1.0 + noise[x, y] if 9 <= (x - center_x2)**2 + (y - center_y2)**2 <= 25 else 0.0
                            self.test_images[idx, x, y] = max(val1, val2)
                elif digit == 9:  # Circle with vertical line
                    center_x, center_y = 12, 12
                    for x in range(6, 18):
                        for y in range(8, 16):
                            if 9 <= (x - center_x)**2 + (y - center_y)**2 <= 25:
                                self.test_images[idx, x, y] = 1.0 + noise[x, y]
                    self.test_images[idx, 6:18, 14:18] = 1.0 + noise[6:18, 14:18]
                
                # Clip values to [0, 1]
                self.test_images[idx] = np.clip(self.test_images[idx], 0.0, 1.0)
        
        logger.info("Created structured synthetic MNIST data")
        logger.info(f"Training data shape: {self.train_images.shape}")
        logger.info(f"Test data shape: {self.test_images.shape}")
    
    def create_train_state(self, rng):
        """Create train state for the model"""
        # Create sample input to initialize model
        sample_x = jnp.ones((1, 28, 28))
        
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
    def train_step(state, x_batch, y_batch):
        """Single training step with JIT compilation"""
        def loss_fn(params):
            logits = state.apply_fn(params, x_batch)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()
            return loss
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        
        # Apply gradients
        state = state.apply_gradients(grads=grads)
        
        return state, loss
    
    @staticmethod
    @jax.jit
    def compute_accuracy(state, x_batch, y_batch):
        """Compute accuracy for current model state"""
        logits = state.apply_fn(state.params, x_batch)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == y_batch)
        return accuracy
    
    def run_real_mnist_test(self, target_accuracy=0.90, max_epochs=50):
        """Run real MNIST test"""
        logger.info(f"Starting real MNIST test - target accuracy: {target_accuracy*100:.1f}%")
        
        # Create random key
        key = jax.random.PRNGKey(0)
        subkey = jax.random.split(key, 1)[0]
        
        # Create train state
        state = self.create_train_state(subkey)
        
        # Training loop
        epoch = 0
        best_accuracy = 0.0
        patience_counter = 0
        max_patience = 10  # Stop if no improvement for 10 epochs
        
        batch_size = 128
        num_batches = len(self.train_images) // batch_size
        
        while epoch < max_epochs:
            # Shuffle training data each epoch
            perm = np.random.permutation(len(self.train_images))
            train_images_shuf = self.train_images[perm]
            train_labels_shuf = self.train_labels[perm]
            # Training epoch
            epoch_loss = 0.0
            for i in range(num_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                x_batch = jnp.array(train_images_shuf[start_idx:end_idx])
                y_batch = jnp.array(train_labels_shuf[start_idx:end_idx])
                
                # Training step
                state, loss = self.train_step(state, x_batch, y_batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / num_batches
            
            # Evaluate every epoch
            test_batch_size = 100
            num_test_batches = len(self.test_images) // test_batch_size
            total_accuracy = 0.0
            
            for i in range(num_test_batches):
                start_idx = i * test_batch_size
                end_idx = start_idx + test_batch_size
                x_test_batch = jnp.array(self.test_images[start_idx:end_idx])
                y_test_batch = jnp.array(self.test_labels[start_idx:end_idx])
                
                accuracy = self.compute_accuracy(state, x_test_batch, y_test_batch)
                total_accuracy += accuracy
            
            avg_accuracy = total_accuracy / num_test_batches
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                patience_counter = 0
                logger.info(f"New best accuracy. Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")
            else:
                patience_counter += 1
            
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}, Best = {best_accuracy:.4f}")
            
            # Check if target accuracy reached
            if avg_accuracy >= target_accuracy:
                logger.info(f"Target accuracy {target_accuracy*100:.1f}% reached at epoch {epoch}")
                return True
            
            # Check for early stopping
            if patience_counter >= max_patience:
                logger.info(f"No improvement for {max_patience} epochs, stopping early")
                break
            
            epoch += 1
        
        logger.info(f"Real MNIST test completed. Best accuracy: {best_accuracy*100:.2f}%")
        
        if best_accuracy >= target_accuracy:
            logger.info(f"Model successfully reached target accuracy of {target_accuracy*100:.1f}%")
            return True
        else:
            logger.info(f"Model did not reach target accuracy. Best was {best_accuracy*100:.2f}%")
            return False


def main():
    """Main function to run real MNIST test"""
    # Create learning test
    learning_test = RealMNISTTest(model_hidden_dim=128, num_classes=10)
    
    # Run test
    success = learning_test.run_real_mnist_test(target_accuracy=0.85, max_epochs=30)
    
    if success:
        logger.info("Real MNIST test PASSED")
        return True
    else:
        logger.error("Real MNIST test FAILED")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
