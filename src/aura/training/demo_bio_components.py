#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Demo script showing how all bio-inspired components work together
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

# Import bio-inspired components
from aura.bio_inspired.phasor_bank import PhasorBankJAX
from aura.bio_inspired.spiking_attention import SpikingAttentionJAX
from aura.bio_inspired.thalamic_router import ThalamicGradientBroadcasterJAX


def demo_bio_inspired_components():
    """Demonstrate all bio-inspired components working together"""
    print("Bio-Inspired Neural Components Demo")
    print("=" * 40)
    
    # Create random key for initialization
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)
    
    # 1. Phasor Bank - Temporal Feature Extraction
    print("\n1. Phasor Bank - Temporal Feature Extraction")
    print("-" * 40)
    
    phasor_bank = PhasorBankJAX(delta0=7.0, H=10)
    input_signal = 0.5  # Scalar input
    params = phasor_bank.init(subkeys[0], input_signal)
    temporal_features = phasor_bank.apply(params, input_signal)
    
    print(f"Input signal: {input_signal}")
    print(f"Temporal features shape: {temporal_features.shape}")
    print(f"First 5 temporal features: {temporal_features[:5]}")
    
    # 2. Spiking Attention - Adaptive Learning Modulation
    print("\n2. Spiking Attention - Adaptive Learning Modulation")
    print("-" * 40)
    
    spiking_attention = SpikingAttentionJAX(k_winners=5, decay=0.7, theta=1.0)
    # For demo, we'll use integer token sequence as expected by spiking attention
    attention_input = jax.random.randint(subkeys[1], (20,), 0, 1000)  # 20 tokens from vocab of 1000
    attention_params = spiking_attention.init(subkeys[1], attention_input, 1000)
    attention_gains = spiking_attention.apply(attention_params, attention_input, 1000)
    # Compute average gain as a measure of learning rate modulation
    learning_rate_mod = jnp.mean(attention_gains)
    
    print(f"Attention input shape: {attention_input.shape}")
    print(f"Attention gains shape: {attention_gains.shape}")
    print(f"Top 5 attention gains: {jnp.sort(attention_gains)[-5:]}")
    print(f"Average learning rate modulation: {learning_rate_mod:.4f}")
    
    # 3. Thalamic Gradient Broadcaster - Learning Signal Routing
    print("\n3. Thalamic Gradient Broadcaster - Learning Signal Routing")
    print("-" * 40)
    
    gradient_broadcaster = ThalamicGradientBroadcasterJAX(total_neurons=1536)
    
    # Create dummy gradients and attention weights
    dummy_gradients = jax.random.normal(subkeys[2], (1536,))
    dummy_attention = jax.random.uniform(subkeys[2], (1536,))
    zone_activations = {
        'retrieval': jax.random.uniform(subkeys[2], (512,)),
        'language': jax.random.uniform(subkeys[2], (512,)),
        'decoder': jax.random.uniform(subkeys[2], (512,))
    }
    
    broadcaster_params = gradient_broadcaster.init(
        subkeys[2], dummy_gradients, dummy_attention, zone_activations
    )
    
    zone_gradients = gradient_broadcaster.apply(
        broadcaster_params, dummy_gradients, dummy_attention, zone_activations
    )
    
    print(f"Input gradients shape: {dummy_gradients.shape}")
    print(f"Zone gradients shapes:")
    for zone_name, grads in zone_gradients.items():
        print(f"  {zone_name}: {grads.shape}")
    
    print("\n" + "=" * 40)
    print("All bio-inspired components demonstrated successfully!")
    print("These components can be integrated into the full AURA training pipeline.")


if __name__ == "__main__":
    demo_bio_inspired_components()
