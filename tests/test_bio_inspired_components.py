#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for all bio-inspired components.
Consolidates testing from multiple redundant learning test files.
"""

import unittest
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aura.bio_inspired.phasor_bank import PhasorBankJAX
from aura.bio_inspired.spiking_attention import SpikingAttentionJAX
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore
from aura.bio_inspired.thalamic_router import ThalamicGradientBroadcasterJAX
from aura.bio_inspired.experts import MLPExpert, Conv1DExpert, RationalExpert
from aura.bio_inspired.merit_board import MeritBoard
from aura.bio_inspired.personality_engine import PersonalityEngineJAX
from aura.bio_inspired.expert_registry import build_core_kwargs_for_zone


class TestPhasorBank(unittest.TestCase):
    """Test PhasorBankJAX functionality."""
    
    def setUp(self):
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=10)
        self.key = jax.random.key(0)
        self.params = self.phasor_bank.init(self.key, 1.0)
    
    def test_phasor_output_shape(self):
        """Test that phasor bank produces correct output shape."""
        input_val = 1.5
        output = self.phasor_bank.apply(self.params, input_val)
        expected_shape = (21,)  # 2 * H + 1
        self.assertEqual(output.shape, expected_shape)
    
    def test_phasor_deterministic(self):
        """Test that phasor bank is deterministic."""
        input_val = 2.0
        output1 = self.phasor_bank.apply(self.params, input_val)
        output2 = self.phasor_bank.apply(self.params, input_val)
        np.testing.assert_array_almost_equal(output1, output2)
    
    def test_phasor_different_inputs(self):
        """Test that different inputs produce different outputs."""
        output1 = self.phasor_bank.apply(self.params, 1.0)
        output2 = self.phasor_bank.apply(self.params, 2.0)
        self.assertFalse(jnp.allclose(output1, output2))


class TestSpikingAttention(unittest.TestCase):
    """Test SpikingAttentionJAX functionality."""
    
    def setUp(self):
        self.attention = SpikingAttentionJAX(decay=0.7, k_winners=5)
        self.key = jax.random.key(0)
    
    def test_attention_output_shape(self):
        """Test that spiking attention produces correct output shape."""
        token_seq = jnp.array([1, 5, 10, 3, 8])
        vocab_size = 100
        output = self.attention(token_seq, vocab_size)
        self.assertEqual(output.shape, (vocab_size,))
    
    def test_attention_bounds(self):
        """Test that attention outputs are within expected bounds."""
        token_seq = jnp.array([1, 5, 10])
        vocab_size = 50
        output = self.attention(token_seq, vocab_size)
        self.assertTrue(jnp.all(output >= 0))
        # Check that some values are modified (not all zeros)
        self.assertTrue(jnp.any(output > 0))


class TestExperts(unittest.TestCase):
    """Test various expert types."""
    
    def setUp(self):
        self.key = jax.random.key(0)
        self.batch_size = 4
        self.input_dim = 32
        self.hidden_dim = 64
        self.test_input = jax.random.normal(self.key, (self.batch_size, self.input_dim))
    
    def test_mlp_expert(self):
        """Test MLPExpert functionality."""
        expert = MLPExpert(hidden_dim=self.hidden_dim, bottleneck=32)
        params = expert.init(self.key, self.test_input)
        output = expert.apply(params, self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
    
    def test_conv1d_expert(self):
        """Test Conv1DExpert functionality."""
        expert = Conv1DExpert(hidden_dim=self.hidden_dim)
        params = expert.init(self.key, self.test_input)
        output = expert.apply(params, self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
    
    def test_rational_expert(self):
        """Test RationalExpert functionality."""
        expert = RationalExpert(hidden_dim=self.hidden_dim)
        params = expert.init(self.key, self.test_input)
        output = expert.apply(params, self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))


class TestEnhancedSpikingRetrieval(unittest.TestCase):
    """Test EnhancedSpikingRetrievalCore functionality."""
    
    def setUp(self):
        self.key = jax.random.key(0)
        self.batch_size = 4
        self.embed_dim = 32
        self.hidden_dim = 64
        self.num_experts = 6
        self.test_input = jax.random.normal(self.key, (self.batch_size, self.embed_dim))
        
        self.core = EnhancedSpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            expert_types=("mlp", "conv1d", "rational"),
            use_bio_gating=True
        )
        self.params = self.core.init(self.key, self.test_input)
    
    def test_core_output_shape(self):
        """Test that retrieval core produces correct output shape."""
        output = self.core.apply(self.params, self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
    
    def test_gate_weights(self):
        """Test gate weight computation."""
        gate_weights = self.core.apply(
            self.params, self.test_input, method=self.core.compute_gate_weights
        )
        self.assertEqual(gate_weights.shape, (self.batch_size, self.num_experts))
        # Check that weights sum to 1 (softmax property)
        weight_sums = jnp.sum(gate_weights, axis=1)
        np.testing.assert_array_almost_equal(weight_sums, jnp.ones(self.batch_size), decimal=5)
    
    def test_expert_outputs(self):
        """Test expert output computation."""
        expert_outputs = self.core.apply(
            self.params, self.test_input, method=self.core.expert_outputs
        )
        self.assertEqual(expert_outputs.shape, (self.batch_size, self.num_experts, self.hidden_dim))
    
    def test_distill_loss(self):
        """Test distillation loss computation."""
        teacher_idx = 0
        student_idx = 1
        loss = self.core.apply(
            self.params, self.test_input, teacher_idx, student_idx,
            method=self.core.distill_loss
        )
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)
    
    def test_active_experts_masking(self):
        """Test that active experts parameter works correctly."""
        active_experts = 3
        output = self.core.apply(self.params, self.test_input, active_experts=active_experts)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        
        # Test that gate weights are masked correctly
        gate_weights = self.core.apply(
            self.params, self.test_input, active_experts,
            method=self.core.compute_gate_weights
        )
        # Inactive experts should have near-zero weights due to -1e9 masking
        inactive_weights = gate_weights[:, active_experts:]
        self.assertTrue(jnp.all(inactive_weights < 1e-6))


class TestMeritBoard(unittest.TestCase):
    """Test MeritBoard functionality."""
    
    def setUp(self):
        self.num_experts = 4
        self.merit_board = MeritBoard(
            num_experts=self.num_experts,
            momentum=0.9,
            scale=0.1
        )
    
    def test_merit_initialization(self):
        """Test that merit board initializes correctly."""
        self.assertEqual(len(self.merit_board.merit), self.num_experts)
        self.assertTrue(all(m == 0.0 for m in self.merit_board.merit))
    
    def test_merit_update(self):
        """Test merit update functionality."""
        weights = jnp.array([0.0, 1.0, 0.0, 0.0])  # Expert 1 gets all weight
        reward = 0.8
        initial_merit = self.merit_board.merit[1]
        self.merit_board.update(weights, reward)
        updated_merit = self.merit_board.merit[1]
        self.assertNotEqual(initial_merit, updated_merit)
    
    def test_get_exploration_bias(self):
        """Test exploration bias computation."""
        bias = self.merit_board.bias()
        self.assertEqual(len(bias), self.num_experts)
        self.assertTrue(all(isinstance(b, (int, float, np.float32, np.float64)) for b in bias))


class TestPersonalityEngine(unittest.TestCase):
    """Test PersonalityEngineJAX functionality."""
    
    def setUp(self):
        self.key = jax.random.key(0)
        self.batch_size = 4
        self.num_experts = 6
        self.personality_traits = jnp.array([0.7, 0.5, 0.8, 0.6, 0.4])  # O, C, E, A, N
        
        self.engine = PersonalityEngineJAX(
            num_experts=self.num_experts,
            hidden_dim=64
        )
        self.params = self.engine.init(self.key, self.personality_traits, jnp.ones((64,)))
    
    def test_personality_modulation(self):
        """Test personality-based modulation."""
        stimulus = jax.random.normal(self.key, (64,))
        output = self.engine.apply(self.params, self.personality_traits, stimulus)
        
        # Check output shape - should be some modulation output
        self.assertIsInstance(output, jnp.ndarray)
        self.assertEqual(len(output.shape), 1)  # Should be 1D output


class TestExpertRegistry(unittest.TestCase):
    """Test expert registry functionality."""
    
    def test_zone_configurations(self):
        """Test that zone configurations are properly set up."""
        zones = ['hippocampus', 'amygdala', 'thalamus', 'hypothalamus', 'language']
        hidden_dim = 128
        
        for zone in zones:
            kwargs = build_core_kwargs_for_zone(zone, hidden_dim, freeze_experts=False)
            self.assertIn('hidden_dim', kwargs)
            self.assertIn('num_experts', kwargs)
            self.assertIn('expert_types', kwargs)
            self.assertEqual(kwargs['hidden_dim'], hidden_dim)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def setUp(self):
        self.key = jax.random.key(0)
        self.batch_size = 2
        self.embed_dim = 32
        self.hidden_dim = 64
    
    def test_phasor_attention_integration(self):
        """Test integration of phasor bank and spiking attention."""
        # Create synthetic input
        query_embedding = jax.random.normal(self.key, (self.batch_size, self.embed_dim))
        
        # Process through phasor bank
        phasor_bank = PhasorBankJAX(delta0=7.0, H=16)
        phasor_params = phasor_bank.init(self.key, 1.0)
        
        query_mean = jnp.mean(query_embedding, axis=-1)
        temporal_features = jax.vmap(phasor_bank.apply, in_axes=(None, 0))(phasor_params, query_mean)
        
        # Process through attention
        attention = SpikingAttentionJAX(k_winners=5)
        K = min(8, self.embed_dim)
        topk_idx = jax.lax.top_k(jnp.abs(query_embedding), K)[1]
        attention_gains = jax.vmap(attention, in_axes=(0, None))(topk_idx, self.embed_dim)
        
        # Check shapes
        self.assertEqual(temporal_features.shape, (self.batch_size, 33))  # 2 * H + 1
        self.assertEqual(attention_gains.shape, (self.batch_size, self.embed_dim))
    
    def test_full_bio_pipeline(self):
        """Test full bio-inspired pipeline with learning."""
        class TestModel(nn.Module):
            def setup(self):
                self.core = EnhancedSpikingRetrievalCore(
                    hidden_dim=32,
                    num_experts=4,
                    expert_types=("mlp", "mlp"),
                    use_bio_gating=True
                )
                self.classifier = nn.Dense(2)  # Binary classification
            
            def __call__(self, x):
                features = self.core(x)
                return self.classifier(features)
        
        # Initialize model
        model = TestModel()
        key1, key2 = jax.random.split(self.key)
        test_input = jax.random.normal(key1, (self.batch_size, self.embed_dim))
        params = model.init(key2, test_input)
        
        # Test forward pass
        output = model.apply(params, test_input)
        self.assertEqual(output.shape, (self.batch_size, 2))
        
        # Test that we can compute gradients
        def loss_fn(params, x, y):
            logits = model.apply(params, x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        
        target = jnp.array([0, 1])
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, test_input, target)
        
        # Check that gradients exist and are not all zeros
        self.assertIsNotNone(grads)
        # Flatten all gradients to check if any are non-zero
        flat_grads = jax.tree_util.tree_leaves(grads)
        has_nonzero_grad = any(jnp.any(jnp.abs(g) > 1e-8) for g in flat_grads)
        self.assertTrue(has_nonzero_grad)


if __name__ == '__main__':
    # Set JAX to use CPU for testing
    jax.config.update('jax_platform_name', 'cpu')
    
    unittest.main()
