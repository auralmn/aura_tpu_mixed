#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Create a production-like dataset for CPU/MPS training with realistic embeddings and token sequences
"""

import os
import sys
import json
import logging
from typing import List, Dict
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_production_dataset(output_dir: str = "data_production", sample_size: int = 50):
    """
    Create a production-like dataset for CPU/MPS training with realistic embeddings and token sequences
    
    Args:
        output_dir: Directory to save the dataset files
        sample_size: Number of samples to include in the dataset (smaller for CPU/MPS)
    """
    try:
        from datasets import load_dataset
        logger.info("Loading GoEmotions dataset from Hugging Face...")
        
        # Load a small subset of the GoEmotions dataset
        dataset = load_dataset("go_emotions", split=f"train[:{sample_size}]")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Emotion categories from GoEmotions
        emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
                   "confusion", "curiosity", "desire", "disappointment", "disapproval", 
                   "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
                   "joy", "love", "nervousness", "optimism", "pride", "realization", 
                   "relief", "remorse", "sadness", "surprise", "neutral"]
        
        # Create phase 0 dataset (core initialization with temporal features)
        phase0_data = []
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
                
            # Get primary emotion for this sample
            emotion_labels = [emotions[j] for j, label in enumerate(item["labels"]) if label == 1]
            primary_emotion = emotion_labels[0] if emotion_labels else "neutral"
            
            sample = {
                "prompt": item["text"],
                "response": f"This is a {primary_emotion} response to: {item['text']}",
                "embedding": np.random.normal(0, 0.1, 768).tolist()  # More realistic embeddings
            }
            phase0_data.append(sample)
        
        # Save phase 0 dataset
        phase0_path = os.path.join(output_dir, "bio_phase0_temporal.jsonl")
        with open(phase0_path, "w") as f:
            for item in phase0_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Phase 0 dataset saved to {phase0_path} with {len(phase0_data)} samples")
        
        # Create phase 1 dataset (consciousness integration with attention)
        phase1_data = []
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
                
            # Get primary emotion for this sample
            emotion_labels = [emotions[j] for j, label in enumerate(item["labels"]) if label == 1]
            primary_emotion = emotion_labels[0] if emotion_labels else "neutral"
            
            # Generate more realistic token sequences (simulating tokenized text)
            token_sequence = np.random.randint(0, 32000, 20).tolist()  # 20 tokens from vocab
            
            sample = {
                "prompt": item["text"],
                "response": f"This is a {primary_emotion} response to: {item['text']}",
                "consciousness_context": f"emotion: {primary_emotion}",
                "token_sequence": token_sequence
            }
            phase1_data.append(sample)
        
        # Save phase 1 dataset
        phase1_path = os.path.join(output_dir, "bio_phase1_attention.jsonl")
        with open(phase1_path, "w") as f:
            for item in phase1_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Phase 1 dataset saved to {phase1_path} with {len(phase1_data)} samples")
        
        # Create phase 2 dataset (self-teaching refinement with gradient broadcasting)
        phase2_data = []
        feedback_templates = [
            "Good {emotion} expression with appropriate tone",
            "Well expressed {emotion} sentiment",
            "Clear {emotion} communication",
            "Effective {emotion} response",
            "Appropriate {emotion} expression for context"
        ]
        
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
                
            # Get primary emotion for this sample
            emotion_labels = [emotions[j] for j, label in enumerate(item["labels"]) if label == 1]
            primary_emotion = emotion_labels[0] if emotion_labels else "neutral"
            
            # Select a feedback template
            feedback = np.random.choice(feedback_templates).format(emotion=primary_emotion)
            
            sample = {
                "prompt": item["text"],
                "response": f"This is a {primary_emotion} response to: {item['text']}",
                "feedback": feedback,
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
            "embedding_dim": 768,
            "vocab_size": 32000,
            "phases": {
                "phase0": {
                    "file": "bio_phase0_temporal.jsonl",
                    "samples": len(phase0_data),
                    "description": "Core initialization training data with temporal features"
                },
                "phase1": {
                    "file": "bio_phase1_attention.jsonl",
                    "samples": len(phase1_data),
                    "description": "Consciousness integration training data with attention"
                },
                "phase2": {
                    "file": "bio_phase2_gradient.jsonl",
                    "samples": len(phase2_data),
                    "description": "Self-teaching refinement training data with gradient broadcasting"
                }
            }
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset metadata saved to {metadata_path}")
        logger.info("Production-like GoEmotions dataset creation completed successfully")
        
    except ImportError:
        logger.error("datasets library not found. Please install with 'pip install datasets'")
        # Create dummy datasets if datasets library is not available
        create_dummy_production_datasets(output_dir, sample_size)
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        # Create dummy datasets as fallback
        create_dummy_production_datasets(output_dir, sample_size)


def create_dummy_production_datasets(output_dir: str = "data_production", sample_size: int = 50):
    """
    Create production-like dummy datasets for CPU/MPS training
    
    Args:
        output_dir: Directory to save the dataset files
        sample_size: Number of samples to include in the dummy dataset
    """
    logger.info("Creating production-like dummy datasets as fallback...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Emotion categories
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
               "confusion", "curiosity", "desire", "disappointment", "disapproval", 
               "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
               "joy", "love", "nervousness", "optimism", "pride", "realization", 
               "relief", "remorse", "sadness", "surprise", "neutral"]
    
    # Create dummy phase 0 dataset
    phase0_data = []
    for i in range(sample_size):
        sample = {
            "prompt": f"Sample prompt {i} for temporal feature enhancement",
            "response": f"Sample response {i} for temporal feature enhancement",
            "embedding": np.random.normal(0, 0.1, 768).tolist()  # More realistic embeddings
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
        primary_emotion = np.random.choice(emotions)
        token_sequence = np.random.randint(0, 32000, 20).tolist()  # 20 tokens from vocab
        
        sample = {
            "prompt": f"Sample prompt {i} for attention modulation",
            "response": f"Sample response {i} expressing {primary_emotion}",
            "consciousness_context": f"emotion: {primary_emotion}",
            "token_sequence": token_sequence
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
    feedback_templates = [
        "Good {emotion} expression with appropriate tone",
        "Well expressed {emotion} sentiment",
        "Clear {emotion} communication",
        "Effective {emotion} response",
        "Appropriate {emotion} expression for context"
    ]
    
    for i in range(sample_size):
        primary_emotion = np.random.choice(emotions)
        feedback = np.random.choice(feedback_templates).format(emotion=primary_emotion)
        
        sample = {
            "prompt": f"Sample prompt {i} for gradient broadcasting",
            "response": f"Sample response {i} expressing {primary_emotion}",
            "feedback": feedback,
            "consciousness_context": f"emotion: {primary_emotion}"
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
        "embedding_dim": 768,
        "vocab_size": 32000,
        "phases": {
            "phase0": {
                "file": "bio_phase0_temporal.jsonl",
                "samples": len(phase0_data),
                "description": "Core initialization training data with temporal features"
            },
            "phase1": {
                "file": "bio_phase1_attention.jsonl",
                "samples": len(phase1_data),
                "description": "Consciousness integration training data with attention"
            },
            "phase2": {
                "file": "bio_phase2_gradient.jsonl",
                "samples": len(phase2_data),
                "description": "Self-teaching refinement training data with gradient broadcasting"
            }
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset metadata saved to {metadata_path}")
    logger.info("Production-like dummy dataset creation completed successfully")


def main():
    """Main function to create production-like dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create production-like GoEmotions dataset for CPU/MPS training")
    parser.add_argument("--output-dir", type=str, default="data_production", 
                       help="Directory to save the dataset files")
    parser.add_argument("--sample-size", type=int, default=50,
                       help="Number of samples to include in the dataset (smaller for CPU/MPS)")
    
    args = parser.parse_args()
    
    create_production_dataset(args.output_dir, args.sample_size)


if __name__ == "__main__":
    main()
