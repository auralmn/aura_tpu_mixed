#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Simple test for bio-inspired components
"""

import os
import sys
import logging
import jax
import jax.numpy as jnp
from flax import linen as nn

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import bio-inspired components
from aura.bio_inspired.phasor_bank import PhasorBankJAX
from aura.bio_inspired.spiking_attention import SpikingAttentionJAX
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore


def test_phasor_bank():
    """Test phasor bank component"""
    logger.info("Testing PhasorBankJAX...")
    
    # Create a simple test
    key = jax.random.PRNGKey(0)
    input_data = 0.5  # scalar input
    
    # Initialize phasor bank
    phasor_bank = PhasorBankJAX(delta0=7.0, H=10)  # Small H for testing
    
    # Initialize parameters
    params = phasor_bank.init(key, input_data)
    
    # Process input
    features = phasor_bank.apply(params, input_data)
    
    logger.info(f"Input value: {input_data}")
    logger.info(f"Output shape: {features.shape}")
    logger.info(f"First 10 features: {features[:10]}")
    logger.info("Phasor bank test completed successfully")
    
    return True


def test_spiking_attention():
    """Test spiking attention component"""
    logger.info("Testing SpikingAttentionJAX...")
    
    # Create a simple test
    key = jax.random.PRNGKey(0)
    token_sequence = jax.random.randint(key, (20,), 0, 1000)  # 20 tokens from vocab of 1000
    
    # Initialize spiking attention
    spiking_attention = SpikingAttentionJAX()
    
    # Initialize parameters
    params = spiking_attention.init(key, token_sequence, 1000)
    
    # Compute attention gains
    gains = spiking_attention.apply(params, token_sequence, 1000)
    
    logger.info(f"Token sequence shape: {token_sequence.shape}")
    logger.info(f"Gains shape: {gains.shape}")
    logger.info(f"Top gains: {jnp.sort(gains)[-10:]}")
    logger.info("Spiking attention test completed successfully")
    
    return True


def test_enhanced_spiking_retrieval():
    """Test enhanced spiking retrieval component"""
    logger.info("Testing EnhancedSpikingRetrievalCore...")
    
    # Create a simple test
    key = jax.random.PRNGKey(0)
    query_embedding = jax.random.normal(key, (1, 64))  # [batch, embed_dim]
    
    # Initialize enhanced spiking retrieval core
    retrieval_core = EnhancedSpikingRetrievalCore(
        hidden_dim=64,
        num_experts=4,
        expert_dim=32,
        phasor_harmonics=10
    )
    
    # Initialize parameters
    params = retrieval_core.init(key, query_embedding)
    
    # Process query
    context_vector = retrieval_core.apply(params, query_embedding)
    
    logger.info(f"Query embedding shape: {query_embedding.shape}")
    logger.info(f"Context vector shape: {context_vector.shape}")
    logger.info("Enhanced spiking retrieval test completed successfully")
    
    return True


def main():
    """Main function to test all bio-inspired components"""
    logger.info("Starting bio-inspired components test")
    
    try:
        # Test each component
        test_phasor_bank()
        test_spiking_attention()
        test_enhanced_spiking_retrieval()
        
        logger.info("All bio-inspired components tested successfully")
        return True
    except Exception as e:
        logger.error(f"Bio-inspired components test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
