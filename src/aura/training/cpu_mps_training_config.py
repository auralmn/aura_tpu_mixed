#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
CPU/MPS optimized training configuration for AURA bio-inspired components
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aura.training.bio_inspired_training import BioInspiredTrainingConfig


class CPUMPSBioInspiredTrainingConfig(BioInspiredTrainingConfig):
    """
    CPU/MPS optimized training configuration for bio-inspired components
    """
    
    def __init__(self):
        super().__init__()
        
        # Reduce model dimensions for CPU/MPS training
        self.embed_dim = 256      # Reduced from 768
        self.hidden_dim = 128    # Reduced from 512
        self.vocab_size = 10000  # Reduced from 32000
        self.num_experts = 8     # Reduced from 16
        
        # Reduce phasor harmonics for CPU/MPS
        self.phasor_harmonics = 64  # Reduced from 192 (for 128-dimensional temporal features)
        
        # Adjust attention parameters for smaller model
        self.attention_k_winners = 5
        self.attention_decay = 0.7
        self.attention_threshold = 1.0
        
        # CPU/MPS optimized training parameters
        self.phase0_epochs = 5      # Reduced from 150
        self.phase1_epochs = 5      # Reduced from 150
        self.phase2_epochs = 5      # Reduced from 150
        
        self.phase0_batch_size = 8   # Reduced from 32
        self.phase1_batch_size = 8   # Reduced from 64
        self.phase2_batch_size = 8   # Reduced from 128
        
        self.phase0_learning_rate = 1e-3
        self.phase1_learning_rate = 1e-3
        self.phase2_learning_rate = 1e-3
        
        # Adjust training parameters for CPU/MPS
        self.warmup_steps = 100
        self.decay_steps = 1000
        self.grad_clip_norm = 1.0
        
        # Local checkpoint directory
        self.local_checkpoint_dir = os.path.abspath("./bio_checkpoints_cpu")
        
        # Data directory for CPU/MPS training
        self.data_dir = os.path.abspath("./data_production")


def main():
    """Test the CPU/MPS training configuration"""
    import json
    
    config = CPUMPSBioInspiredTrainingConfig()
    
    # Print configuration summary
    print("CPU/MPS Bio-Inspired Training Configuration")
    print("=" * 50)
    print(f"Embedding dimension: {config.embed_dim}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Number of experts: {config.num_experts}")
    print(f"Phasor harmonics: {config.phasor_harmonics}")
    print(f"Phase 0 epochs: {config.phase0_epochs}")
    print(f"Phase 1 epochs: {config.phase1_epochs}")
    print(f"Phase 2 epochs: {config.phase2_epochs}")
    print(f"Phase 0 batch size: {config.phase0_batch_size}")
    print(f"Phase 1 batch size: {config.phase1_batch_size}")
    print(f"Phase 2 batch size: {config.phase2_batch_size}")
    print(f"Local checkpoint directory: {config.local_checkpoint_dir}")
    print(f"Data directory: {config.data_dir}")
    
    # Save configuration to file
    config_dict = {
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "vocab_size": config.vocab_size,
        "num_experts": config.num_experts,
        "phasor_harmonics": config.phasor_harmonics,
        "attention_k_winners": config.attention_k_winners,
        "attention_decay": config.attention_decay,
        "attention_threshold": config.attention_threshold,
        "phase0_epochs": config.phase0_epochs,
        "phase1_epochs": config.phase1_epochs,
        "phase2_epochs": config.phase2_epochs,
        "phase0_batch_size": config.phase0_batch_size,
        "phase1_batch_size": config.phase1_batch_size,
        "phase2_batch_size": config.phase2_batch_size,
        "phase0_learning_rate": config.phase0_learning_rate,
        "phase1_learning_rate": config.phase1_learning_rate,
        "phase2_learning_rate": config.phase2_learning_rate,
        "warmup_steps": config.warmup_steps,
        "decay_steps": config.decay_steps,
        "grad_clip_norm": config.grad_clip_norm,
        "local_checkpoint_dir": config.local_checkpoint_dir,
        "data_dir": config.data_dir
    }
    
    with open("cpu_mps_training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print("\nConfiguration saved to cpu_mps_training_config.json")


if __name__ == "__main__":
    main()
