#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import linen as nn


class MLPExpert(nn.Module):
    """Simple MLP expert that maps [batch, in_dim] -> [batch, hidden_dim].
    Dense layers infer input dimension at init-time; no explicit in_dim needed.
    """
    hidden_dim: int
    bottleneck: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.bottleneck)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        return x


class Conv1DExpert(nn.Module):
    """Conv1D expert over feature dimension.
    Treats input [batch, length] as a 1D signal with channels=1, applies small conv stack,
    pools, then projects to hidden_dim.
    """
    hidden_dim: int
    channels: int = 32
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch, length]
        x = x[..., None]  # [batch, length, 1]
        x = nn.Conv(features=self.channels, kernel_size=(self.kernel_size,))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.channels, kernel_size=(3,))(x)
        x = nn.gelu(x)
        # Global average pool over length
        x = jnp.mean(x, axis=1)  # [batch, channels]
        x = nn.Dense(self.hidden_dim)(x)
        return x


class RationalExpert(nn.Module):
    """Rational expert modeling gated interactions to link topics quickly."""
    hidden_dim: int
    bottleneck: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h1 = nn.Dense(self.bottleneck)(x)
        h1 = nn.gelu(h1)
        h2 = nn.Dense(self.bottleneck)(x)
        h2 = nn.sigmoid(h2)
        inter = h1 * h2
        out = nn.Dense(self.hidden_dim)(inter)
        return out


class CodeExpert(nn.Module):
    """Code expert using multi-branch composition patterns."""
    hidden_dim: int
    branch_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b1 = nn.Dense(self.branch_dim)(x)
        b2 = nn.Dense(self.branch_dim)(x)
        b3 = nn.Dense(self.branch_dim)(x)
        h = nn.gelu(b1) + nn.silu(b2) + nn.relu(b3)
        out = nn.Dense(self.hidden_dim)(h)
        return out


class SelfImproveExpert(nn.Module):
    """Self-improvement expert with residual refinement."""
    hidden_dim: int
    bottleneck: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        base = nn.Dense(self.hidden_dim)(x)
        h = nn.Dense(self.bottleneck)(x)
        h = nn.gelu(h)
        delta = nn.Dense(self.hidden_dim)(h)
        return base + 0.5 * delta
