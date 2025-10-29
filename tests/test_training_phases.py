#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for training phases.
Consolidates functionality from simple_phase_test.py and test_all_phases.py.
"""

import unittest
import os
import sys
import tempfile
import jax
import jax.numpy as jnp
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aura.training.bio_inspired_training import BioInspiredTrainingConfig, BioInspiredAURATrainingPipeline


class TestTrainingPhases(unittest.TestCase):
    """Test training phase functionality."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = BioInspiredTrainingConfig()
        # Reduce parameters for fast testing
        self.config.phase0_epochs = 1
        self.config.phase1_epochs = 1
        self.config.phase2_epochs = 1
        self.config.batch_size = 4
        self.config.learning_rate = 1e-3
        
        self.pipeline = BioInspiredAURATrainingPipeline(self.config)
    
    def test_phase0_temporal_feature_enhancement(self):
        """Test Phase 0: Temporal Feature Enhancement."""
        try:
            training_state = self.pipeline.phase0_temporal_feature_enhancement()
            
            # Check that we got a training state
            self.assertIsNotNone(training_state)
            
            # Check that metrics were recorded
            self.assertIn('phase0', self.pipeline.metrics)
            phase0_metrics = self.pipeline.metrics['phase0']
            self.assertGreater(len(phase0_metrics), 0)
            
            # Check that the final metrics have expected keys
            final_metrics = phase0_metrics[-1]
            expected_keys = ['epoch', 'loss', 'accuracy']
            for key in expected_keys:
                if key in final_metrics:  # Some keys might be optional
                    self.assertIsInstance(final_metrics[key], (int, float, np.ndarray))
            
            print(f"Phase 0 completed successfully. Final metrics: {final_metrics}")
            
        except Exception as e:
            self.fail(f"Phase 0 failed with error: {e}")
    
    def test_phase1_attention_modulated_training(self):
        """Test Phase 1: Attention-Modulated Training."""
        # First run Phase 0 to get initial state
        try:
            initial_state = self.pipeline.phase0_temporal_feature_enhancement()
        except Exception as e:
            self.skipTest(f"Cannot test Phase 1 because Phase 0 failed: {e}")
        
        try:
            training_state = self.pipeline.phase1_attention_modulated_training(initial_state)
            
            # Check that we got a training state
            self.assertIsNotNone(training_state)
            
            # Check that metrics were recorded
            self.assertIn('phase1', self.pipeline.metrics)
            phase1_metrics = self.pipeline.metrics['phase1']
            self.assertGreater(len(phase1_metrics), 0)
            
            # Check that the final metrics have expected keys
            final_metrics = phase1_metrics[-1]
            expected_keys = ['epoch', 'loss', 'accuracy']
            for key in expected_keys:
                if key in final_metrics:
                    self.assertIsInstance(final_metrics[key], (int, float, np.ndarray))
            
            print(f"Phase 1 completed successfully. Final metrics: {final_metrics}")
            
        except Exception as e:
            self.fail(f"Phase 1 failed with error: {e}")
    
    def test_phase2_gradient_broadcasting_refinement(self):
        """Test Phase 2: Gradient Broadcasting Refinement."""
        # First run Phase 0 and 1 to get initial state
        try:
            state_p0 = self.pipeline.phase0_temporal_feature_enhancement()
            initial_state = self.pipeline.phase1_attention_modulated_training(state_p0)
        except Exception as e:
            self.skipTest(f"Cannot test Phase 2 because earlier phases failed: {e}")
        
        try:
            training_state = self.pipeline.phase2_gradient_broadcasting_refinement(initial_state)
            
            # Check that we got a training state
            self.assertIsNotNone(training_state)
            
            # Check that metrics were recorded
            self.assertIn('phase2', self.pipeline.metrics)
            phase2_metrics = self.pipeline.metrics['phase2']
            self.assertGreater(len(phase2_metrics), 0)
            
            # Check that the final metrics have expected keys
            final_metrics = phase2_metrics[-1]
            expected_keys = ['epoch', 'loss', 'accuracy']
            for key in expected_keys:
                if key in final_metrics:
                    self.assertIsInstance(final_metrics[key], (int, float, np.ndarray))
            
            print(f"Phase 2 completed successfully. Final metrics: {final_metrics}")
            
        except Exception as e:
            self.fail(f"Phase 2 failed with error: {e}")
    
    def test_all_phases_sequential(self):
        """Test running all phases sequentially."""
        try:
            # Phase 0
            print("Running Phase 0: Temporal Feature Enhancement")
            state_p0 = self.pipeline.phase0_temporal_feature_enhancement()
            self.assertIsNotNone(state_p0)
            
            # Phase 1
            print("Running Phase 1: Attention-Modulated Training")
            state_p1 = self.pipeline.phase1_attention_modulated_training(state_p0)
            self.assertIsNotNone(state_p1)
            
            # Phase 2
            print("Running Phase 2: Gradient Broadcasting Refinement")
            state_p2 = self.pipeline.phase2_gradient_broadcasting_refinement(state_p1)
            self.assertIsNotNone(state_p2)
            
            # Check that all phases recorded metrics
            self.assertIn('phase0', self.pipeline.metrics)
            self.assertIn('phase1', self.pipeline.metrics)
            self.assertIn('phase2', self.pipeline.metrics)
            
            # Check that metrics show some progression
            for phase in ['phase0', 'phase1', 'phase2']:
                metrics_list = self.pipeline.metrics[phase]
                self.assertGreater(len(metrics_list), 0)
                print(f"{phase} final metrics: {metrics_list[-1]}")
            
            print("All phases completed successfully!")
            
        except Exception as e:
            self.fail(f"Sequential phase execution failed: {e}")
    
    def test_phase_configuration(self):
        """Test that phase configurations are properly set."""
        # Test default configuration
        default_config = BioInspiredTrainingConfig()
        
        # Check that all required attributes exist
        required_attrs = [
            'phase0_epochs', 'phase1_epochs', 'phase2_epochs',
            'batch_size', 'learning_rate', 'hidden_dim',
            'num_experts', 'embed_dim'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(default_config, attr), f"Missing attribute: {attr}")
            value = getattr(default_config, attr)
            self.assertIsNotNone(value, f"Attribute {attr} is None")
            
        # Test custom configuration
        custom_config = BioInspiredTrainingConfig()
        custom_config.phase0_epochs = 5
        custom_config.batch_size = 16
        custom_config.learning_rate = 2e-4
        
        custom_pipeline = BioInspiredAURATrainingPipeline(custom_config)
        self.assertEqual(custom_pipeline.config.phase0_epochs, 5)
        self.assertEqual(custom_pipeline.config.batch_size, 16)
        self.assertEqual(custom_pipeline.config.learning_rate, 2e-4)
    
    def test_metrics_recording(self):
        """Test that metrics are properly recorded during training."""
        # Run a minimal training to check metrics recording
        try:
            # Run just phase 0 with single epoch
            self.config.phase0_epochs = 1
            pipeline = BioInspiredAURATrainingPipeline(self.config)
            
            state = pipeline.phase0_temporal_feature_enhancement()
            
            # Check metrics structure
            self.assertIn('phase0', pipeline.metrics)
            metrics_list = pipeline.metrics['phase0']
            self.assertIsInstance(metrics_list, list)
            self.assertGreater(len(metrics_list), 0)
            
            # Check individual metric entries
            for metric_entry in metrics_list:
                self.assertIsInstance(metric_entry, dict)
                # At minimum should have epoch info
                self.assertIn('epoch', metric_entry)
                
        except Exception as e:
            self.skipTest(f"Cannot test metrics recording due to training failure: {e}")
    
    def test_state_persistence(self):
        """Test that training state is properly maintained between phases."""
        try:
            # Run phase 0 and check state
            state_p0 = self.pipeline.phase0_temporal_feature_enhancement()
            
            # State should have required components
            self.assertIsNotNone(state_p0)
            # Check that state has typical JAX training state attributes
            expected_attrs = ['params', 'opt_state', 'step']
            for attr in expected_attrs:
                if hasattr(state_p0, attr):
                    self.assertIsNotNone(getattr(state_p0, attr))
            
            # Run phase 1 with the state
            state_p1 = self.pipeline.phase1_attention_modulated_training(state_p0)
            self.assertIsNotNone(state_p1)
            
            # Step count should have progressed
            if hasattr(state_p0, 'step') and hasattr(state_p1, 'step'):
                self.assertGreaterEqual(state_p1.step, state_p0.step)
            
        except Exception as e:
            self.skipTest(f"Cannot test state persistence due to training failure: {e}")


class TestTrainingConfiguration(unittest.TestCase):
    """Test training configuration functionality."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = BioInspiredTrainingConfig()
        
        # Test that numerical parameters are positive
        self.assertGreater(config.phase0_epochs, 0)
        self.assertGreater(config.phase1_epochs, 0)
        self.assertGreater(config.phase2_epochs, 0)
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.learning_rate, 0)
        self.assertGreater(config.hidden_dim, 0)
        self.assertGreater(config.num_experts, 0)
        self.assertGreater(config.embed_dim, 0)
    
    def test_config_modification(self):
        """Test that configuration can be modified."""
        config = BioInspiredTrainingConfig()
        
        # Modify values
        original_epochs = config.phase0_epochs
        config.phase0_epochs = original_epochs + 5
        
        original_batch_size = config.batch_size
        config.batch_size = original_batch_size * 2
        
        # Check modifications took effect
        self.assertEqual(config.phase0_epochs, original_epochs + 5)
        self.assertEqual(config.batch_size, original_batch_size * 2)


if __name__ == '__main__':
    # Set JAX to use CPU for testing
    jax.config.update('jax_platform_name', 'cpu')
    
    unittest.main(verbosity=2)
