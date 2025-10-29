#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Spiking Attention Mechanism for Adaptive Learning Rate Modulation
TPU-compatible implementation using JAX
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class SpikingAttentionJAX(nn.Module):
    """
    k-WTA spiking attention for adaptive learning rate modulation.
    TPU-compatible through vectorized operations.
    """
    decay: float = 0.7
    theta: float = 1.0
    k_winners: int = 5
    gain_up: float = 1.5
    gain_down: float = 0.6
    
    def __call__(self, token_seq: jnp.ndarray, vocab_size: int) -> jnp.ndarray:
        """
        Compute attention gains for token sequence.
        
        Args:
            token_seq: Token sequence [seq_len,]
            vocab_size: Vocabulary size
            
        Returns:
            Attention gains [vocab_size,]
        """
        # Initialize state
        v = jnp.zeros(vocab_size)
        spikes = jnp.zeros(vocab_size)
        
        # Process sequence through LIF dynamics using scan
        def lif_step(state, token_id):
            v, spikes = state
            # Ensure token_id is within bounds
            token_id_int = jnp.clip(token_id, 0, vocab_size - 1).astype(jnp.int32)
            
            # Update membrane potential
            v_new = self.decay * v.at[token_id_int].get() + 1.0
            
            # Check for spike
            spiked = v_new >= self.theta
            
            # Soft reset
            v_reset = jnp.where(spiked, v_new - self.theta, v_new)
            
            # Update potentials
            v = v.at[token_id_int].set(v_reset)
            
            # Update spike counts
            spikes_new = spikes.at[token_id_int].add(spiked.astype(jnp.int32))
            
            return (v, spikes_new), None
        
        # Scan through token sequence
        (v_final, spikes_final), _ = jax.lax.scan(lif_step, (v, spikes), token_seq)
        
        # Determine top-k winners
        k = int(min(self.k_winners, int(vocab_size)))
        top_k_vals, top_k_indices = jax.lax.top_k(v_final, k)
        winners = jnp.zeros(vocab_size).at[top_k_indices].set(1)
        
        # Compute learning rate gains
        gains = jnp.where(
            winners, 
            self.gain_up, 
            jnp.where(spikes_final > 0, self.gain_down, 1.0)
        )
        
        return gains


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test
    key = jax.random.PRNGKey(0)
    token_sequence = jax.random.randint(key, (20,), 0, 1000)  # 20 tokens from vocab of 1000
    
    # Initialize spiking attention
    spiking_attention = SpikingAttentionJAX()
    
    # Initialize parameters
    params = spiking_attention.init(key, token_sequence, 1000)
    
    # Compute attention gains
    gains = spiking_attention.apply(params, token_sequence, 1000)
    
    print(f"Token sequence shape: {token_sequence.shape}")
    print(f"Gains shape: {gains.shape}")
    print(f"Top gains: {jnp.sort(gains)[-10:]}")
    print("Spiking attention implementation test completed successfully")
