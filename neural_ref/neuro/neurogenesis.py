# SPDX-License-Identifier: Apache-2.0
"""
AURA Neurogenesis Training System for Hierarchical MoE
- Dynamic expert creation based on content patterns
- Hebbian learning for synaptic strength adaptation
- Neural pruning for ineffective experts
- Content-aware expert specialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
from collections import defaultdict
from .expert import Expert
from .hierarchical_moe import HierarchicalMoELayer

import logging
logger = logging.getLogger(__name__)

@dataclass
class NeurogenesisConfig:
    """Configuration for neurogenesis system"""
    # Creation thresholds
    creation_threshold: float = 0.75     # Routing confidence below this triggers creation
    utilization_threshold: float = 0.1  # Usage below this triggers pruning

    # Learning rates
    hebbian_lr: float = 0.01            # Hebbian learning rate
    synaptic_decay: float = 0.999       # Synaptic strength decay

    # Population control
    max_experts_per_category: int = 16  # Maximum experts per category
    min_experts_per_category: int = 2   # Minimum experts per category

    # Content analysis
    content_history_window: int = 1000  # Tokens to analyze for patterns
    pattern_similarity_threshold: float = 0.65  # Similarity for expert assignment

class ContentAnalyzer:
    """Analyzes content patterns to guide expert specialization"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pattern_memory = defaultdict(list)  # Store content patterns

    def extract_content_features(self, token_ids: torch.Tensor, text: Optional[str] = None) -> Dict[str, float]:
        """Extract content features for expert specialization"""
        features = {}

        if token_ids is not None:
            token_ids_flat = token_ids.flatten().cpu().numpy()

            # Token frequency distribution
            unique_tokens, counts = np.unique(token_ids_flat, return_counts=True)
            features['vocab_diversity'] = len(unique_tokens) / len(token_ids_flat)
            features['repetition_ratio'] = 1.0 - (len(unique_tokens) / len(token_ids_flat))

            # Token ID statistics
            features['avg_token_id'] = float(np.mean(token_ids_flat))
            features['token_id_std'] = float(np.std(token_ids_flat))

        if text is not None:
            # Linguistic features
            features['avg_word_length'] = np.mean([len(word) for word in text.split()])
            features['punctuation_density'] = len(re.findall(r'[.!?,:;]', text)) / max(len(text), 1)
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)

            # Content type indicators
            features['has_numbers'] = 1.0 if re.search(r'\d', text) else 0.0
            features['has_code'] = 1.0 if any(kw in text.lower() for kw in ['def ', 'class ', 'import ', '{', '}']) else 0.0
            features['has_math'] = 1.0 if any(sym in text for sym in ['=', '+', '-', '*', '/', '%']) else 0.0

        return features

    def extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features for enhanced content analysis"""
        features = {}
        
        if not text:
            return features
            
        # Sentence complexity analysis
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            features['sentence_complexity'] = avg_sentence_length
        else:
            features['sentence_complexity'] = 0.0
            
        # Technical vocabulary analysis
        technical_terms = [
            'algorithm', 'analysis', 'architecture', 'computation', 'configuration',
            'implementation', 'optimization', 'parameter', 'protocol', 'specification',
            'synthesis', 'validation', 'verification', 'framework', 'methodology'
        ]
        
        text_lower = text.lower()
        technical_count = sum(1 for term in technical_terms if term in text_lower)
        technical_vocabulary_ratio = technical_count / max(len(text.split()), 1)
        features['technical_terms'] = technical_vocabulary_ratio
        
        # Narrative structure indicators
        narrative_indicators = [
            'however', 'therefore', 'consequently', 'furthermore', 'moreover',
            'initially', 'subsequently', 'finally', 'meanwhile', 'nevertheless',
            'indeed', 'specifically', 'particularly', 'especially', 'notably'
        ]
        
        narrative_count = sum(1 for indicator in narrative_indicators if indicator in text_lower)
        story_structure_signals = narrative_count / max(len(text.split()), 1)
        features['narrative_indicators'] = story_structure_signals
        
        return features

    def categorize_content(self, features: Dict[str, float]) -> int:
        """Categorize content into one of 8 general categories"""
        # Simple rule-based categorization (can be enhanced with ML)
        if features.get('has_code', 0.0) > 0.5:
            return 0  # Programming
        elif features.get('has_math', 0.0) > 0.5:
            return 1  # Mathematics
        elif features.get('punctuation_density', 0.0) > 0.05:
            return 2  # Formal writing
        elif features.get('vocab_diversity', 0.5) > 0.8:
            return 3  # Creative writing
        elif features.get('has_numbers', 0.0) > 0.5:
            return 4  # Data/Statistics
        elif features.get('repetition_ratio', 0.0) > 0.3:
            return 5  # Structured content
        elif features.get('avg_word_length', 5.0) > 7.0:
            return 6  # Technical writing
        else:
            return 7  # General content

    def find_similar_expert(self, features: Dict[str, float], category: int, 
                          existing_patterns: List[Dict[str, float]], 
                          threshold: float = 0.7) -> Optional[int]:
        """Find existing expert with similar content patterns"""
        if not existing_patterns:
            return None

        feature_keys = set(features.keys())

        for i, pattern in enumerate(existing_patterns):
            pattern_keys = set(pattern.keys())
            common_keys = feature_keys.intersection(pattern_keys)

            if not common_keys:
                continue

            # Compute cosine similarity
            features_vec = np.array([features[k] for k in common_keys])
            pattern_vec = np.array([pattern[k] for k in common_keys])

            if np.linalg.norm(features_vec) == 0 or np.linalg.norm(pattern_vec) == 0:
                continue

            similarity = np.dot(features_vec, pattern_vec) / (
                np.linalg.norm(features_vec) * np.linalg.norm(pattern_vec)
            )

            if similarity > threshold:
                return i

        return None

class NeurogenesisTrainer:
    """Handles dynamic expert creation, adaptation, and pruning"""

    def __init__(self, config: NeurogenesisConfig):
        self.config = config
        self.content_analyzer = ContentAnalyzer()

        # Track expert performance and usage
        self.expert_performance = defaultdict(list)  # category -> [specialist_performance]
        self.expert_patterns = defaultdict(list)     # category -> [content_patterns]
        self.expert_utilization = defaultdict(lambda: defaultdict(float))  # category -> specialist -> usage
        self.synaptic_strengths = defaultdict(lambda: defaultdict(float))  # Hebbian connections

        # Neurogenesis statistics
        self.creation_events = 0
        self.pruning_events = 0

    def should_create_expert(self, category: int, routing_confidence: float, 
                           content_features: Dict[str, float]) -> bool:
        """Determine if new expert should be created"""
        # Low routing confidence indicates no good existing expert
        if routing_confidence < self.config.creation_threshold:
            # Check if we have room for more experts
            current_count = len(self.expert_patterns[category])
            if current_count < self.config.max_experts_per_category:
                # Check if content pattern is novel
                similar_expert = self.content_analyzer.find_similar_expert(
                    content_features, category, self.expert_patterns[category],
                    self.config.pattern_similarity_threshold
                )
                return similar_expert is None
        return False

    def create_expert(self, moe_layer, category: int, content_features: Dict[str, float],
                     d_model: int, d_ff: int, dropout: float = 0.1) -> int:
        """Create new expert through neurogenesis"""
        # Ensure category exists
        if len(moe_layer.experts) <= category:
            while len(moe_layer.experts) <= category:
                moe_layer.experts.append(nn.ModuleList())

        # Create new specialist expert
        new_expert = Expert(d_model, d_ff, dropout)

        # Initialize with content-aware weights
        self._initialize_expert_weights(new_expert, content_features)

        # Move to same device as existing experts
        if len(moe_layer.experts) > 0 and len(moe_layer.experts[0]) > 0:
            device = next(moe_layer.experts[0][0].parameters()).device
            new_expert = new_expert.to(device)

        # Add to the category
        moe_layer.experts[category].append(new_expert)
        specialist_idx = len(moe_layer.experts[category]) - 1

        # Record the content pattern
        self.expert_patterns[category].append(content_features.copy())

        # Initialize performance tracking
        if len(self.expert_performance[category]) <= specialist_idx:
            self.expert_performance[category].append([])
        self.expert_utilization[category][specialist_idx] = 0.0

        self.creation_events += 1

        return specialist_idx

    def _initialize_expert_weights(self, expert: nn.Module, content_features: Dict[str, float]):
        """Initialize expert weights based on content features"""
        # Content-aware initialization
        vocab_bias = content_features.get('avg_token_id', 0.0) / 32000.0  # Normalize
        complexity_factor = content_features.get('vocab_diversity', 0.5)

        # Adjust initialization based on content complexity
        init_std = 0.02 * (1.0 + complexity_factor)

        with torch.no_grad():
            # Initialize with slight bias based on content
            expert.w1.weight.normal_(mean=vocab_bias * 0.01, std=init_std)
            expert.w2.weight.normal_(mean=0.0, std=init_std)

    def adapt_expert(self, expert: nn.Module, input_tokens: torch.Tensor, 
                    performance: float, category: int, specialist: int):
        """Adapt existing expert using Hebbian learning"""
        # Update utilization
        self.expert_utilization[category][specialist] += 1.0

        # Record performance
        if len(self.expert_performance[category]) <= specialist:
            self.expert_performance[category].extend([[] for _ in range(specialist + 1 - len(self.expert_performance[category]))])
        self.expert_performance[category][specialist].append(performance)

        # Hebbian adaptation based on performance
        if performance > 0.5:  # Good performance
            with torch.no_grad():
                # Strengthen connections that were active
                input_mean = torch.mean(input_tokens, dim=0)

                # Hebbian rule: strengthen weights proportional to input and performance
                hebbian_update = self.config.hebbian_lr * performance * input_mean

                # Update first layer weights (input connections)
                if input_mean.numel() == expert.w1.weight.size(1):
                    expert.w1.weight += hebbian_update.unsqueeze(0).expand_as(expert.w1.weight) * 0.1

                # Decay to prevent unbounded growth
                expert.w1.weight *= self.config.synaptic_decay
                expert.w2.weight *= self.config.synaptic_decay

    def should_prune_expert(self, category: int, specialist: int) -> bool:
        """Determine if expert should be pruned"""
        # Don't prune below minimum
        if len(self.expert_patterns[category]) <= self.config.min_experts_per_category:
            return False

        # Check utilization
        utilization = self.expert_utilization[category].get(specialist, 0.0)
        if utilization < self.config.utilization_threshold:
            return True

        # Check performance history
        if (category in self.expert_performance and 
            specialist < len(self.expert_performance[category]) and
            len(self.expert_performance[category][specialist]) > 10):
            recent_performance = np.mean(self.expert_performance[category][specialist][-10:])
            if recent_performance < 0.3:  # Consistently poor performance
                return True

        return False

    def prune_expert(self, moe_layer, category: int, specialist: int):
        """Remove underperforming expert"""
        if (category < len(moe_layer.experts) and 
            specialist < len(moe_layer.experts[category])):
            # Remove the expert
            del moe_layer.experts[category][specialist]

            # Clean up tracking data
            if specialist < len(self.expert_patterns[category]):
                del self.expert_patterns[category][specialist]
            if (category in self.expert_performance and 
                specialist < len(self.expert_performance[category])):
                del self.expert_performance[category][specialist]
            if specialist in self.expert_utilization[category]:
                del self.expert_utilization[category][specialist]

            self.pruning_events += 1

    def train_step(self, moe_layer, hidden_states: torch.Tensor, 
                  token_ids: Optional[torch.Tensor] = None,
                  text: Optional[str] = None,
                  routing_confidence: Optional[torch.Tensor] = None,
                  expert_outputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single training step with neurogenesis"""
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        # Extract content features
        content_features = self.content_analyzer.extract_content_features(token_ids, text)
        category = self.content_analyzer.categorize_content(content_features)

        # Initialize routing outputs
        cat_indices = torch.full((batch_size, seq_len, 2), category, dtype=torch.long, device=device)

        # Simulate low routing confidence to trigger expert creation
        # In practice, this would come from the actual router
        simulated_confidence = np.random.uniform(0.1, 0.4) if routing_confidence is None else routing_confidence

        # Check if we need to create new expert
        if self.should_create_expert(category, simulated_confidence, content_features) or routing_confidence is None:
            new_specialist = self.create_expert(
                moe_layer, category, content_features, 
                d_model, d_model * 4  # d_ff = 4 * d_model
            )
            print(f"Created new expert: category {category}, specialist {new_specialist}")

        # Determine specialist routing
        num_specialists = (len(moe_layer.experts[category]) 
                          if category < len(moe_layer.experts) 
                          else 1)

        # Simple routing to specialists (can be enhanced with learned routing)
        sub_indices = torch.randint(0, max(num_specialists, 1), (batch_size, seq_len, 2), device=device)
        weights = torch.softmax(torch.randn(batch_size, seq_len, 2, device=device), dim=-1)

        # Adapt existing experts if we have performance feedback
        if expert_outputs is not None:
            # Compute performance as inverse of prediction error
            target = hidden_states  # Auto-encoding task
            performance = 1.0 / (1.0 + torch.mean((expert_outputs - target) ** 2, dim=-1))

            for specialist in range(num_specialists):
                if (category < len(moe_layer.experts) and 
                    specialist < len(moe_layer.experts[category])):
                    avg_performance = torch.mean(performance).item()
                    self.adapt_expert(
                        moe_layer.experts[category][specialist],
                        hidden_states.mean(dim=0).mean(dim=0),  # Average input
                        avg_performance,
                        category,
                        specialist
                    )

        # Periodic pruning check
        if np.random.random() < 0.01:  # 1% chance per training step
            for specialist in range(num_specialists - 1, -1, -1):  # Reverse order for safe deletion
                if self.should_prune_expert(category, specialist):
                    print(f"Pruning expert: category {category}, specialist {specialist}")
                    self.prune_expert(moe_layer, category, specialist)

        return cat_indices, sub_indices, weights

    def get_neurogenesis_stats(self) -> Dict[str, any]:
        """Get neurogenesis training statistics"""
        stats = {
            'creation_events': self.creation_events,
            'pruning_events': self.pruning_events,
            'total_experts': sum(len(patterns) for patterns in self.expert_patterns.values()),
            'experts_per_category': {cat: len(patterns) for cat, patterns in self.expert_patterns.items()},
            'avg_utilization_per_category': {},
            'avg_performance_per_category': {}
        }

        # Calculate average metrics per category
        for category in self.expert_utilization.keys():
            if self.expert_utilization[category]:
                avg_util = np.mean(list(self.expert_utilization[category].values()))
                stats['avg_utilization_per_category'][category] = avg_util

            if category in self.expert_performance and self.expert_performance[category]:
                category_perfs = []
                for specialist_perfs in self.expert_performance[category]:
                    if specialist_perfs:
                        category_perfs.extend(specialist_perfs)
                if category_perfs:
                    stats['avg_performance_per_category'][category] = np.mean(category_perfs)

        return stats

class NeurogenesisMoELayer(HierarchicalMoELayer):
    """MoE Layer with neurogenesis capabilities"""

    def __init__(self, d_model: int, d_ff: int, num_categories: int = 8, 
                 initial_specialists: int = 2, top_k: int = 2, dropout: float = 0.1,
                 neurogenesis_config: Optional[NeurogenesisConfig] = None):
        # Initialize with minimal experts
        super().__init__(d_model, d_ff, num_categories, initial_specialists, top_k, dropout)

        self.neurogenesis_config = neurogenesis_config or NeurogenesisConfig()
        self.trainer = NeurogenesisTrainer(self.neurogenesis_config)

    def forward(self, hidden, cat_indices=None, sub_indices=None, weights=None, 
               mask=None, token_ids=None, text=None, training_mode=False):
        """Forward with optional neurogenesis training"""

        if training_mode:
            # Generate routing through neurogenesis
            cat_indices, sub_indices, weights = self.trainer.train_step(
                self, hidden, token_ids, text
            )

        # Standard forward pass
        return super().forward(hidden, cat_indices, sub_indices, weights, mask)

    def get_neurogenesis_stats(self):
        """Get neurogenesis statistics"""
        return self.trainer.get_neurogenesis_stats()
