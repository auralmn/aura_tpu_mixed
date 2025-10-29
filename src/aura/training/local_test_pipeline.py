#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight local test pipeline for AURA consciousness system
Simulates training phases without full computational overhead
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

# AURA imports
from aura.self_teaching_llm.spiking_retrieval_core import SpikingRetrievalCore
from aura.self_teaching_llm.spiking_language_core import SpikingLanguageCore
from aura.self_teaching_llm.token_decoder import TokenDecoder
from aura.self_teaching_llm.self_teaching_adapter import SelfTeachingAdapter
from aura.consciousness.aura_consciousness_system import AURAConsciousnessSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalTestConfig:
    """Lightweight configuration for local testing"""
    
    def __init__(self):
        # Reduced parameters for local testing
        self.test_epochs = 2
        self.test_batch_size = 4
        self.embed_dim = 32
        self.hidden_dim = 64
        self.vocab_size = 1000
        self.num_experts = 8


class AURALocalTestPipeline:
    """Lightweight local test pipeline for AURA system components"""
    
    def __init__(self, config: LocalTestConfig):
        self.config = config
        
        # Initialize consciousness system
        self.consciousness = AURAConsciousnessSystem()
        self.consciousness.start_processing()
        
        # Initialize self-teaching adapter with reduced dimensions
        self.adapter = SelfTeachingAdapter(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            vocab_size=self.config.vocab_size,
            num_experts=self.config.num_experts
        )
        
        # Test metrics
        self.metrics = {
            'component_initialization': {},
            'consciousness_integration': {},
            'self_teaching_functionality': {}
        }
    
    def test_component_initialization(self) -> Dict[str, Any]:
        """Test initialization of core components"""
        logger.info("Testing Component Initialization")
        
        start_time = time.time()
        
        # Test spiking retrieval core
        retrieval_core = SpikingRetrievalCore(
            hidden_dim=self.config.hidden_dim,
            num_experts=self.config.num_experts,
            expert_dim=self.config.embed_dim
        )
        
        # Test spiking language core
        language_core = SpikingLanguageCore(
            hidden_dim=self.config.hidden_dim
        )
        
        # Test token decoder
        token_decoder = TokenDecoder(
            hidden_dim=self.config.hidden_dim,
            vocab_size=self.config.vocab_size
        )
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        query_embedding = jax.random.normal(key, (2, self.config.embed_dim))
        rate_vector = jax.random.uniform(key, (2, self.config.hidden_dim))
        
        # Test retrieval core
        retrieval_params = retrieval_core.init(key, query_embedding)
        context_vector = retrieval_core.apply(retrieval_params, query_embedding)
        
        # Test language core
        lang_params = language_core.init(key, rate_vector, 
                                       language_core.initialize_state(2))
        initial_state = language_core.initialize_state(2)
        output_rate, next_state = language_core.apply(lang_params, 
                                                    context_vector, 
                                                    initial_state)
        
        # Test token decoder
        decoder_params = token_decoder.init(key, rate_vector)
        token_probs = token_decoder.apply(decoder_params, output_rate)
        
        end_time = time.time()
        
        # Record metrics
        self.metrics['component_initialization'] = {
            'duration': end_time - start_time,
            'retrieval_core_shape': context_vector.shape,
            'language_core_shape': output_rate.shape,
            'decoder_shape': token_probs.shape,
            'components_initialized': ['retrieval_core', 'language_core', 'token_decoder']
        }
        
        logger.info(f"Component initialization test completed in {end_time - start_time:.2f} seconds")
        return self.metrics['component_initialization']
    
    def test_consciousness_integration(self) -> Dict[str, Any]:
        """Test consciousness system integration"""
        logger.info("Testing Consciousness Integration")
        
        start_time = time.time()
        
        # Add knowledge fragments to consciousness system
        knowledge_fragments = [
            ("The quick brown fox jumps over the lazy dog", 
             jax.random.normal(jax.random.PRNGKey(0), (self.config.hidden_dim,))),
            ("Machine learning models process data patterns efficiently", 
             jax.random.normal(jax.random.PRNGKey(1), (self.config.hidden_dim,))),
            ("Neural networks learn through backpropagation and attention", 
             jax.random.normal(jax.random.PRNGKey(2), (self.config.hidden_dim,))),
            ("Consciousness emerges from complex neural dynamics and feedback", 
             jax.random.normal(jax.random.PRNGKey(3), (self.config.hidden_dim,)))
        ]
        
        for text, embedding in knowledge_fragments:
            self.consciousness.add_knowledge(text, embedding)
        
        # Run consciousness processing for a few cycles
        for epoch in range(self.config.test_epochs):
            # Get consciousness status
            status = self.consciousness.get_consciousness_status()
            
            # Simulate processing
            time.sleep(0.1)  # Simulate processing time
            
            logger.info(f"Consciousness Epoch {epoch}: Level = {status['consciousness_level']:.3f}")
        
        end_time = time.time()
        
        # Record metrics
        final_status = self.consciousness.get_consciousness_status()
        self.metrics['consciousness_integration'] = {
            'duration': end_time - start_time,
            'final_consciousness_level': final_status['consciousness_level'],
            'workspace_size': len(final_status['workspace']['contents']),
            'knowledge_buffer_size': final_status['knowledge_buffer_size'],
            'integration_successful': True
        }
        
        logger.info(f"Consciousness integration test completed in {end_time - start_time:.2f} seconds")
        return self.metrics['consciousness_integration']
    
    def test_self_teaching_functionality(self) -> Dict[str, Any]:
        """Test self-teaching adapter functionality"""
        logger.info("Testing Self-Teaching Functionality")
        
        start_time = time.time()
        
        # Create test prompt embeddings
        key = jax.random.PRNGKey(42)
        prompt_embeddings = jax.random.normal(key, (2, self.config.embed_dim))
        
        # Set adapter key for initialization
        self.adapter.key = key
        
        # Test generation without consciousness
        tokens_no_context, rates_no_context = self.adapter.generate_with_consciousness(
            prompt_embeddings, max_len=10, temperature=0.7)
        
        # Test generation with consciousness
        tokens_with_context, rates_with_context = self.adapter.generate_with_consciousness(
            prompt_embeddings, consciousness_system=self.consciousness, 
            max_len=10, temperature=0.7)
        
        end_time = time.time()
        
        # Record metrics
        self.metrics['self_teaching_functionality'] = {
            'duration': end_time - start_time,
            'generation_without_context_shape': tokens_no_context.shape,
            'generation_with_context_shape': tokens_with_context.shape,
            'rates_without_context_count': len(rates_no_context),
            'rates_with_context_count': len(rates_with_context),
            'functionality_tested': ['generation', 'consciousness_integration']
        }
        
        logger.info(f"Self-teaching functionality test completed in {end_time - start_time:.2f} seconds")
        return self.metrics['self_teaching_functionality']
    
    def run_local_test_pipeline(self) -> Dict[str, Any]:
        """Run the complete lightweight local test pipeline"""
        logger.info("Starting lightweight local AURA test pipeline")
        
        start_time = time.time()
        
        # Test component initialization
        self.test_component_initialization()
        
        # Test consciousness integration
        self.test_consciousness_integration()
        
        # Test self-teaching functionality
        self.test_self_teaching_functionality()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Compile final results
        results = {
            'total_duration': total_duration,
            'metrics': self.metrics,
            'tests_completed': ['component_initialization', 'consciousness_integration', 'self_teaching_functionality']
        }
        
        logger.info(f"Local test pipeline completed in {total_duration:.2f} seconds")
        return results
    
    def export_test_results(self, results: Dict[str, Any], output_path: str):
        """Export test results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Test results exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export test results: {e}")


def main():
    """Main local test pipeline execution"""
    parser = argparse.ArgumentParser(description="AURA Local Test Pipeline")
    parser.add_argument("--output", type=str, default="local_test_results.json",
                       help="Output file for test results")
    
    args = parser.parse_args()
    
    # Initialize test configuration
    config = LocalTestConfig()
    
    # Initialize test pipeline
    pipeline = AURALocalTestPipeline(config)
    
    # Run test pipeline
    results = pipeline.run_local_test_pipeline()
    
    # Export results
    pipeline.export_test_results(results, args.output)
    
    logger.info("Local test pipeline execution completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
