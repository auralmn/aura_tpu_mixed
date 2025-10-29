#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Harmonic Phasor Bank for Temporal Feature Extraction
TPU-compatible implementation using JAX
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class PhasorBankJAX(nn.Module):
    """
    H-harmonic leaky resonator bank for temporal feature extraction.
    Compatible with TPU through vectorized operations.
    """
    delta0: float
    H: int = 192  # Number of harmonics (results in 384 features)
    rho: float = 0.985  # Decay factor
    kappa: float = 1.0  # Input gain
    
    def setup(self):
        self.omega = 2 * jnp.pi / jnp.maximum(self.delta0, 1e-4)
    
    def __call__(self, u_t: float) -> jnp.ndarray:
        """
        Process a single input value through phasor bank.
        
        Args:
            u_t: Input signal (scalar)
            
        Returns:
            Temporal features [2*H+1,]
        """
        # Vectorized implementation for all harmonics
        k = jnp.arange(1, self.H + 1)  # [H,]
        omega = 2 * jnp.pi / jnp.maximum(self.delta0, 1e-4)
        k_omega = k * omega
        
        # Initialize state variables
        cx = jnp.zeros(self.H)
        cy = jnp.zeros(self.H)
        
        # Compute rotation matrices for all harmonics at once
        cosw = jnp.cos(k_omega)  # [H,]
        sinw = jnp.sin(k_omega)  # [H,]
        
        # Apply rotation and decay (vectorized)
        xr = cosw * cx - sinw * cy  # [H,]
        yr = sinw * cx + cosw * cy  # [H,]
        
        # Update state with input
        x_new = self.rho * xr + self.kappa * u_t  # Broadcast u_t to [H,]
        y_new = self.rho * yr
        
        # Flatten to feature vector [1 + 2*H elements]
        feats = jnp.concatenate([
            jnp.array([1.0]),  # Bias term
            jnp.stack([x_new, y_new], axis=-1).reshape(-1)
        ])
        
        return feats


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test
    key = jax.random.PRNGKey(0)
    input_data = 0.5  # scalar input
    
    # Initialize phasor bank
    phasor_bank = PhasorBankJAX(delta0=7.0, H=10)  # Small H for testing
    
    # Initialize parameters
    params = phasor_bank.init(key, input_data)
    
    # Process input
    features = phasor_bank.apply(params, input_data)
    
    print(f"Input value: {input_data}")
    print(f"Output shape: {features.shape}")
    print(f"First 10 features: {features[:10]}")
    print("Phasor bank implementation test completed successfully")
