#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Create a minimal dataset from Hugging Face GoEmotions for AURA training
"""

import os
import sys
import json
from typing import List, Dict
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_minimal_goemotions_dataset(output_dir: str = "data", sample_size: int = 100):
    """
    Create a minimal dataset from Hugging Face GoEmotions for testing training pipeline
    
    Args:
        output_dir: Directory to save the dataset files
        sample_size: Number of samples to include in the minimal dataset
    """
    try:
        from datasets import load_dataset
        logger.info("Loading GoEmotions dataset from Hugging Face...")
        
        # Load a small subset of the GoEmotions dataset
        dataset = load_dataset("go_emotions", split="train[:{}]".format(sample_size))
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create phase 0 dataset (core initialization)
        phase0_data = []
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
                
            sample = {
                "prompt": item["text"],
                "response": f"This is a response to: {item['text']}",
                "embedding": [0.1 * (i % 10)] * 768  # Dummy embedding
            }
            phase0_data.append(sample)
        
        # Save phase 0 dataset
        phase0_path = os.path.join(output_dir, "bio_phase0_temporal.jsonl")
        with open(phase0_path, "w") as f:
            for item in phase0_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Phase 0 dataset saved to {phase0_path} with {len(phase0_data)} samples")
        
        # Create phase 1 dataset (consciousness integration)
        phase1_data = []
        emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
                   "confusion", "curiosity", "desire", "disappointment", "disapproval", 
                   "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
                   "joy", "love", "nervousness", "optimism", "pride", "realization", 
                   "relief", "remorse", "sadness", "surprise"]
        
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
                
            # Get primary emotion for this sample
            emotion_labels = [emotions[j] for j, label in enumerate(item["labels"]) if label == 1]
            primary_emotion = emotion_labels[0] if emotion_labels else "neutral"
            
            sample = {
                "prompt": item["text"],
                "response": f"This is a {primary_emotion} response to: {item['text']}",
                "consciousness_context": f"emotion: {primary_emotion}",
                "token_sequence": [i % 1000, (i + 1) % 1000, (i + 2) % 1000]  # Dummy token sequence
            }
            phase1_data.append(sample)
        
        # Save phase 1 dataset
        phase1_path = os.path.join(output_dir, "bio_phase1_attention.jsonl")
        with open(phase1_path, "w") as f:
            for item in phase1_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Phase 1 dataset saved to {phase1_path} with {len(phase1_data)} samples")
        
        # Create phase 2 dataset (self-teaching refinement)
        phase2_data = []
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
                
            # Get primary emotion for this sample
            emotion_labels = [emotions[j] for j, label in enumerate(item["labels"]) if label == 1]
            primary_emotion = emotion_labels[0] if emotion_labels else "neutral"
            
            sample = {
                "prompt": item["text"],
                "response": f"This is a {primary_emotion} response to: {item['text']}",
                "feedback": f"Good {primary_emotion} expression with appropriate tone",
                "consciousness_context": f"emotion: {primary_emotion}"
            }
            phase2_data.append(sample)
        
        # Save phase 2 dataset
        phase2_path = os.path.join(output_dir, "bio_phase2_gradient.jsonl")
        with open(phase2_path, "w") as f:
            for item in phase2_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Phase 2 dataset saved to {phase2_path} with {len(phase2_data)} samples")
        
        # Create metadata file
        metadata = {
            "dataset_source": "Hugging Face GoEmotions",
            "sample_size": sample_size,
            "phases": {
                "phase0": {
                    "file": "bio_phase0_temporal.jsonl",
                    "samples": len(phase0_data),
                    "description": "Core initialization training data"
                },
                "phase1": {
                    "file": "bio_phase1_attention.jsonl",
                    "samples": len(phase1_data),
                    "description": "Consciousness integration training data"
                },
                "phase2": {
                    "file": "bio_phase2_gradient.jsonl",
                    "samples": len(phase2_data),
                    "description": "Self-teaching refinement training data"
                }
            }
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset metadata saved to {metadata_path}")
        logger.info("Minimal GoEmotions dataset creation completed successfully")
        
    except ImportError:
        logger.error("datasets library not found. Please install with 'pip install datasets'")
        # Create dummy datasets if datasets library is not available
        create_dummy_datasets(output_dir, sample_size)
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        # Create dummy datasets as fallback
        create_dummy_datasets(output_dir, sample_size)


def create_dummy_datasets(output_dir: str = "data", sample_size: int = 100):
    """
    Create dummy datasets for testing when Hugging Face datasets is not available
    
    Args:
        output_dir: Directory to save the dataset files
        sample_size: Number of samples to include in the dummy dataset
    """
    logger.info("Creating dummy datasets as fallback...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy phase 0 dataset
    phase0_data = []
    for i in range(sample_size):
        sample = {
            "prompt": f"Dummy prompt {i} for core initialization",
            "response": f"Dummy response {i} for core initialization",
            "embedding": [0.1 * (i % 10)] * 768  # Dummy embedding
        }
        phase0_data.append(sample)
    
    # Save phase 0 dataset
    phase0_path = os.path.join(output_dir, "bio_phase0_temporal.jsonl")
    with open(phase0_path, "w") as f:
        for item in phase0_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Dummy phase 0 dataset saved to {phase0_path} with {len(phase0_data)} samples")
    
    # Create dummy phase 1 dataset
    phase1_data = []
    for i in range(sample_size):
        sample = {
            "prompt": f"Dummy prompt {i} for consciousness integration",
            "response": f"Dummy response {i} for consciousness integration",
            "consciousness_context": f"dummy_context_{i % 10}",
            "token_sequence": [i % 1000, (i + 1) % 1000, (i + 2) % 1000]
        }
        phase1_data.append(sample)
    
    # Save phase 1 dataset
    phase1_path = os.path.join(output_dir, "bio_phase1_attention.jsonl")
    with open(phase1_path, "w") as f:
        for item in phase1_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Dummy phase 1 dataset saved to {phase1_path} with {len(phase1_data)} samples")
    
    # Create dummy phase 2 dataset
    phase2_data = []
    for i in range(sample_size):
        sample = {
            "prompt": f"Dummy prompt {i} for self-teaching refinement",
            "response": f"Dummy response {i} for self-teaching refinement",
            "feedback": f"Good response quality for sample {i}",
            "consciousness_context": f"dummy_context_{i % 10}"
        }
        phase2_data.append(sample)
    
    # Save phase 2 dataset
    phase2_path = os.path.join(output_dir, "bio_phase2_gradient.jsonl")
    with open(phase2_path, "w") as f:
        for item in phase2_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Dummy phase 2 dataset saved to {phase2_path} with {len(phase2_data)} samples")
    
    # Create metadata file
    metadata = {
        "dataset_source": "Dummy data",
        "sample_size": sample_size,
        "phases": {
            "phase0": {
                "file": "bio_phase0_temporal.jsonl",
                "samples": len(phase0_data),
                "description": "Core initialization training data"
            },
            "phase1": {
                "file": "bio_phase1_attention.jsonl",
                "samples": len(phase1_data),
                "description": "Consciousness integration training data"
            },
            "phase2": {
                "file": "bio_phase2_gradient.jsonl",
                "samples": len(phase2_data),
                "description": "Self-teaching refinement training data"
            }
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset metadata saved to {metadata_path}")
    logger.info("Dummy dataset creation completed successfully")


def main():
    """Main function to create minimal dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create minimal GoEmotions dataset for AURA training")
    parser.add_argument("--output-dir", type=str, default="data", 
                       help="Directory to save the dataset files")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of samples to include in the minimal dataset")
    
    args = parser.parse_args()
    
    create_minimal_goemotions_dataset(args.output_dir, args.sample_size)


if __name__ == "__main__":
    main()
