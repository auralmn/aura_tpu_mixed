#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Test script for enhanced spiking retrieval core
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

# Import the enhanced retrieval core
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore


def test_enhanced_retrieval_core():
    """Test the enhanced spiking retrieval core"""
    # Create a simple test
    key = jax.random.PRNGKey(0)
    query_data = jax.random.normal(key, (4, 32))  # [batch, embed_dim]
    
    # Initialize enhanced retrieval core
    retrieval_core = EnhancedSpikingRetrievalCore(
        hidden_dim=64,
        num_experts=8,
        expert_dim=32
    )
    
    # Initialize parameters
    params = retrieval_core.init(key, query_data)
    
    # Process query
    context_vector = retrieval_core.apply(params, query_data)
    
    print(f"Query shape: {query_data.shape}")
    print(f"Context vector shape: {context_vector.shape}")
    print("Enhanced spiking retrieval core implementation test completed successfully")


if __name__ == "__main__":
    test_enhanced_retrieval_core()
