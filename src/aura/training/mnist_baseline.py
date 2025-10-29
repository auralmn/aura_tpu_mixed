#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Baseline MNIST classifier using CSV data and a simple MLP (Flax/JAX).
Targets >95% accuracy quickly to validate the data pipeline.
"""

import os
import sys
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

# Add src to path for imports if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineMNISTModel(nn.Module):
    hidden_dim: int = 512
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


def load_csv_mnist():
    """Load MNIST train/test from CSV files located under data/MNIST/raw"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    csv_train_path = os.path.join(project_root, 'data', 'MNIST', 'raw', 'mnist_train.csv')
    csv_test_path = os.path.join(project_root, 'data', 'MNIST', 'raw', 'mnist_test.csv')

    if not os.path.exists(csv_train_path):
        raise FileNotFoundError(f"MNIST train CSV not found at {csv_train_path}")

    logger.info(f"Loading MNIST train CSV: {csv_train_path}")
    train_data = np.loadtxt(csv_train_path, delimiter=',', dtype=np.float32)
    y_train = train_data[:, 0].astype(np.int32)
    X_train = (train_data[:, 1:] / 255.0).reshape((-1, 28, 28)).astype(np.float32)

    if os.path.exists(csv_test_path):
        logger.info(f"Loading MNIST test CSV: {csv_test_path}")
        test_data = np.loadtxt(csv_test_path, delimiter=',', dtype=np.float32)
        y_test = test_data[:, 0].astype(np.int32)
        X_test = (test_data[:, 1:] / 255.0).reshape((-1, 28, 28)).astype(np.float32)
    else:
        # Split from train if test not provided
        split = max(5000, int(0.1 * X_train.shape[0]))
        X_test, y_test = X_train[-split:], y_train[-split:]
        X_train, y_train = X_train[:-split], y_train[:-split]

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def create_train_state(rng, model):
    params = model.init(rng, jnp.ones((1, 28, 28)))
    tx = optax.adam(1e-3)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, x, y):
    logits = state.apply_fn(state.params, x)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y)
    return acc


def run_training(target_acc=0.95, max_epochs=15, batch_size=256, hidden_dim=512):
    X_train, y_train, X_test, y_test = load_csv_mnist()

    model = BaselineMNISTModel(hidden_dim=hidden_dim, num_classes=10)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    for epoch in range(max_epochs):
        t0 = time.time()
        # Shuffle each epoch
        perm = np.random.permutation(n_train)
        X_train_shuf = X_train[perm]
        y_train_shuf = y_train[perm]

        # Mini-batch SGD
        num_batches = n_train // batch_size
        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = jnp.array(X_train_shuf[start:end])
            y_batch = jnp.array(y_train_shuf[start:end])
            state, loss = train_step(state, x_batch, y_batch)
            epoch_loss += float(loss)

        # Eval
        test_bs = 512
        num_test_batches = n_test // test_bs
        acc_sum = 0.0
        for i in range(num_test_batches):
            s = i * test_bs
            e = s + test_bs
            acc = float(eval_step(state, jnp.array(X_test[s:e]), jnp.array(y_test[s:e])))
            acc_sum += acc
        test_acc = acc_sum / max(1, num_test_batches)

        logger.info(f"Epoch {epoch}: loss={(epoch_loss/max(1,num_batches)):.4f}, acc={test_acc:.4f}, time={(time.time()-t0):.1f}s")
        if test_acc >= target_acc:
            logger.info(f"Target accuracy {target_acc*100:.1f}% reached at epoch {epoch}")
            return True

    logger.info(f"Training completed. Best observed accuracy may be below target {target_acc*100:.1f}%")
    # Final full evaluation (optional)
    return False


def main():
    success = run_training(target_acc=0.95, max_epochs=20, batch_size=256, hidden_dim=512)
    if not success:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
