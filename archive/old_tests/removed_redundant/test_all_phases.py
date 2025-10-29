#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Test script to run all phases of the bio-inspired training pipeline
"""

import os
import sys
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aura.training.bio_inspired_training import BioInspiredTrainingConfig, BioInspiredAURATrainingPipeline


def test_all_phases():
    """Test all phases of the bio-inspired training pipeline"""
    # Create training configuration
    config = BioInspiredTrainingConfig()
    
    # Initialize training pipeline
    pipeline = BioInspiredAURATrainingPipeline(config)
    
    print("Testing Phase 0: Temporal Feature Enhancement")
    try:
        training_state = pipeline.phase0_temporal_feature_enhancement()
        print("Phase 0 completed successfully")
    except Exception as e:
        print(f"Phase 0 failed: {e}")
        return False
    
    print("\nTesting Phase 1: Attention-Modulated Training")
    try:
        training_state = pipeline.phase1_attention_modulated_training(training_state)
        print("Phase 1 completed successfully")
    except Exception as e:
        print(f"Phase 1 failed: {e}")
        return False
    
    print("\nTesting Phase 2: Gradient Broadcasting Refinement")
    try:
        training_state = pipeline.phase2_gradient_broadcasting_refinement(training_state)
        print("Phase 2 completed successfully")
    except Exception as e:
        print(f"Phase 2 failed: {e}")
        return False
    
    # Export training results
    results = {
        'phase': 'all',
        'metrics': pipeline.metrics
    }
    
    output_file = 'bio_all_phases_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll phases completed successfully. Results saved to {output_file}")
    return True


if __name__ == "__main__":
    success = test_all_phases()
    if not success:
        sys.exit(1)
