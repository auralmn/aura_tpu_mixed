#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
CPU/MPS training script for AURA bio-inspired components
"""

import os
import sys
import json
import logging
import argparse
import stat

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CPU/MPS optimized configuration
from aura.training.cpu_mps_training_config import CPUMPSBioInspiredTrainingConfig
from aura.training.bio_inspired_training import BioInspiredAURATrainingPipeline


def train_phase(phase: int, output_file: str, config: CPUMPSBioInspiredTrainingConfig):
    """Train a specific phase with CPU/MPS optimization"""
    logger.info(f"Starting CPU/MPS training for phase {phase}")
    
    # Ensure output directory exists with proper permissions
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o777, exist_ok=True)
        os.chmod(output_dir, 0o777)
    
    # Initialize training pipeline
    pipeline = BioInspiredAURATrainingPipeline(config)
    
    # Execute the appropriate training phase
    if phase == 0:
        training_state = pipeline.phase0_temporal_feature_enhancement()
    elif phase == 1:
        # For phase 1, we need a training state from phase 0
        training_state = pipeline.setup_training_state(config.phase1_learning_rate)
        training_state = pipeline.phase1_attention_modulated_training(training_state)
    elif phase == 2:
        # For phase 2, we need a training state (can be initialized)
        training_state = pipeline.setup_training_state(config.phase2_learning_rate)
        training_state = pipeline.phase2_gradient_broadcasting_refinement(training_state)
    else:
        raise ValueError(f"Invalid phase: {phase}")
    
    # Export training results
    results = {
        'phase': phase,
        'metrics': pipeline.metrics[f'phase{phase}'],
        'config': {
            'embed_dim': config.embed_dim,
            'hidden_dim': config.hidden_dim,
            'vocab_size': config.vocab_size,
            'num_experts': config.num_experts,
            'phasor_harmonics': config.phasor_harmonics,
            'epochs': getattr(config, f'phase{phase}_epochs'),
            'batch_size': getattr(config, f'phase{phase}_batch_size'),
            'learning_rate': getattr(config, f'phase{phase}_learning_rate')
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Phase {phase} training completed. Results saved to {output_file}")
    return True


def train_all_phases(output_dir: str, config: CPUMPSBioInspiredTrainingConfig):
    """Train all phases sequentially"""
    logger.info("Starting CPU/MPS training for all phases")
    
    # Ensure output directory exists with proper permissions
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o777, exist_ok=True)
        os.chmod(output_dir, 0o777)
    
    # Initialize training pipeline
    pipeline = BioInspiredAURATrainingPipeline(config)
    
    # Phase 0
    logger.info("Starting Phase 0: Temporal Feature Enhancement")
    training_state = pipeline.phase0_temporal_feature_enhancement()
    
    # Phase 1
    logger.info("Starting Phase 1: Attention-Modulated Training")
    training_state = pipeline.phase1_attention_modulated_training(training_state)
    
    # Phase 2
    logger.info("Starting Phase 2: Gradient Broadcasting Refinement")
    training_state = pipeline.phase2_gradient_broadcasting_refinement(training_state)
    
    # Export all training results
    results = {
        'phases': ['temporal_feature_enhancement', 'attention_modulated_training', 'gradient_broadcasting_refinement'],
        'metrics': pipeline.metrics,
        'config': {
            'embed_dim': config.embed_dim,
            'hidden_dim': config.hidden_dim,
            'vocab_size': config.vocab_size,
            'num_experts': config.num_experts,
            'phasor_harmonics': config.phasor_harmonics,
            'phase0_epochs': config.phase0_epochs,
            'phase1_epochs': config.phase1_epochs,
            'phase2_epochs': config.phase2_epochs,
            'phase0_batch_size': config.phase0_batch_size,
            'phase1_batch_size': config.phase1_batch_size,
            'phase2_batch_size': config.phase2_batch_size,
            'phase0_learning_rate': config.phase0_learning_rate,
            'phase1_learning_rate': config.phase1_learning_rate,
            'phase2_learning_rate': config.phase2_learning_rate
        }
    }
    
    output_file = os.path.join(output_dir, "cpu_mps_training_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All phases training completed. Results saved to {output_file}")
    return True


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="CPU/MPS training for AURA bio-inspired components")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2], 
                        help="Training phase to execute (0, 1, or 2)")
    parser.add_argument("--output", type=str, default="cpu_mps_training_results.json",
                        help="Output file for training results")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for training results")
    parser.add_argument("--all-phases", action="store_true",
                        help="Train all phases sequentially")
    
    args = parser.parse_args()
    
    # Create CPU/MPS optimized configuration
    config = CPUMPSBioInspiredTrainingConfig()
    
    if args.all_phases:
        success = train_all_phases(args.output_dir, config)
    elif args.phase is not None:
        output_file = args.output
        if not output_file.endswith('.json'):
            output_file = f"phase{args.phase}_{output_file}"
            if not output_file.endswith('.json'):
                output_file += ".json"
        success = train_phase(args.phase, output_file, config)
    else:
        parser.print_help()
        return False
    
    if success:
        logger.info("CPU/MPS training completed successfully")
        return True
    else:
        logger.error("CPU/MPS training failed")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
