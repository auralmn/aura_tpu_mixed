#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Simple test to run one epoch of each training phase without checkpoint saving
"""

import os
import sys
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aura.training.bio_inspired_training import BioInspiredTrainingConfig, BioInspiredAURATrainingPipeline


def test_single_epoch_no_checkpoint():
    """Test single epoch of each phase without checkpoint saving"""
    # Create training configuration
    config = BioInspiredTrainingConfig()
    
    # Reduce epochs to 1 for quick testing
    config.phase0_epochs = 1
    config.phase1_epochs = 1
    config.phase2_epochs = 1
    
    # Initialize training pipeline
    pipeline = BioInspiredAURATrainingPipeline(config)
    
    print("Testing Phase 0: Temporal Feature Enhancement (1 epoch)")
    try:
        training_state = pipeline.phase0_temporal_feature_enhancement()
        print("Phase 0 completed successfully")
        print(f"Phase 0 metrics: {pipeline.metrics['phase0']}")
    except Exception as e:
        print(f"Phase 0 failed: {e}")
        return False
    
    print("\nTesting Phase 1: Attention-Modulated Training (1 epoch)")
    try:
        training_state = pipeline.phase1_attention_modulated_training(training_state)
        print("Phase 1 completed successfully")
        print(f"Phase 1 metrics: {pipeline.metrics['phase1']}")
    except Exception as e:
        print(f"Phase 1 failed: {e}")
        return False
    
    print("\nTesting Phase 2: Gradient Broadcasting Refinement (1 epoch)")
    try:
        training_state = pipeline.phase2_gradient_broadcasting_refinement(training_state)
        print("Phase 2 completed successfully")
        print(f"Phase 2 metrics: {pipeline.metrics['phase2']}")
    except Exception as e:
        print(f"Phase 2 failed: {e}")
        return False
    
    print("\nAll phases completed successfully!")
    return True


if __name__ == "__main__":
    success = test_single_epoch_no_checkpoint()
    if not success:
        sys.exit(1)
