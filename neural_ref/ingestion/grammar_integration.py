# SPDX-License-Identifier: Apache-2.0
"""
AURA Grammar Pattern Integration: Advanced Linguistic Analysis
- Spacy SVC pattern ingestion and processing
- Domain-specific grammar pattern analysis
- Linguistic complexity assessment
- Grammar-aware text generation and understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path
import logging
from collections import defaultdict, Counter
import pickle
import gzip

class GrammarComplexity(Enum):
    """Grammar complexity levels"""
    SIMPLE = "simple"
    COMPOUND = "compound"
    COMPLEX = "complex"
    COMPOUND_COMPLEX = "compound-complex"

class VoiceType(Enum):
    """Voice types"""
    ACTIVE = "active"
    PASSIVE = "passive"

class TenseType(Enum):
    """Tense types"""
    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"
    BASE = "base"
    PERFECT = "perfect"
    PROGRESSIVE = "progressive"

@dataclass
class GrammarPattern:
    """Individual grammar pattern structure"""
    subject: str
    verb: str
    complement: str
    tense: str
    voice: str
    complexity: str
    domain: str
    quality_score: float
    source: str
    sentence: str
    
    def __post_init__(self):
        """Validate and normalize pattern data"""
        # Normalize complexity
        if self.complexity not in [c.value for c in GrammarComplexity]:
            self.complexity = GrammarComplexity.SIMPLE.value
        
        # Normalize voice
        if self.voice not in [v.value for v in VoiceType]:
            self.voice = VoiceType.ACTIVE.value
        
        # Normalize tense
        if self.tense not in [t.value for t in TenseType]:
            self.tense = TenseType.PRESENT.value
        
        # Ensure quality score is valid
        self.quality_score = max(0.0, min(1.0, self.quality_score))

@dataclass
class DomainPatterns:
    """Patterns for a specific domain"""
    domain_name: str
    patterns: List[GrammarPattern]
    pattern_stats: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute domain statistics"""
        if self.patterns:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute comprehensive domain statistics"""
        total_patterns = len(self.patterns)
        
        # Complexity distribution
        complexity_counts = Counter(p.complexity for p in self.patterns)
        self.pattern_stats['complexity_distribution'] = {
            comp: count / total_patterns 
            for comp, count in complexity_counts.items()
        }
        
        # Voice distribution
        voice_counts = Counter(p.voice for p in self.patterns)
        self.pattern_stats['voice_distribution'] = {
            voice: count / total_patterns 
            for voice, count in voice_counts.items()
        }
        
        # Tense distribution
        tense_counts = Counter(p.tense for p in self.patterns)
        self.pattern_stats['tense_distribution'] = {
            tense: count / total_patterns 
            for tense, count in tense_counts.items()
        }
        
        # Quality statistics
        quality_scores = [p.quality_score for p in self.patterns]
        self.pattern_stats['quality_stats'] = {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores),
            'median': np.median(quality_scores)
        }
        
        # Source distribution
        source_counts = Counter(p.source for p in self.patterns)
        self.pattern_stats['source_distribution'] = dict(source_counts)
        
        # Average sentence length
        sentence_lengths = [len(p.sentence.split()) for p in self.patterns]
        self.pattern_stats['sentence_length_stats'] = {
            'mean': np.mean(sentence_lengths),
            'std': np.std(sentence_lengths),
            'min': np.min(sentence_lengths),
            'max': np.max(sentence_lengths)
        }

class GrammarPatternEncoder(nn.Module):
    """
    Neural encoder for grammar patterns
    """
    
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_domains: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Domain embeddings
        self.domain_embeddings = nn.Embedding(num_domains, embedding_dim)
        
        # Pattern component encoders
        self.subject_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.verb_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.complement_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Grammar feature encoders
        self.complexity_encoder = nn.Embedding(len(GrammarComplexity), embedding_dim)
        self.voice_encoder = nn.Embedding(len(VoiceType), embedding_dim)
        self.tense_encoder = nn.Embedding(len(TenseType), embedding_dim)
        
        # Quality score processor
        self.quality_processor = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
        # Pattern fusion network
        self.pattern_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + embedding_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )
        
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(GrammarComplexity))
        )
        
        # Quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pattern_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode grammar pattern"""
        
        # Extract components
        subject_ids = pattern_data['subject_ids']  # (batch, seq_len)
        verb_ids = pattern_data['verb_ids']        # (batch, seq_len)
        complement_ids = pattern_data['complement_ids']  # (batch, seq_len)
        domain_ids = pattern_data['domain_ids']    # (batch,)
        complexity_ids = pattern_data['complexity_ids']  # (batch,)
        voice_ids = pattern_data['voice_ids']      # (batch,)
        tense_ids = pattern_data['tense_ids']      # (batch,)
        quality_scores = pattern_data['quality_scores']  # (batch, 1)
        
        # Encode text components
        subject_emb = self.word_embeddings(subject_ids)
        verb_emb = self.word_embeddings(verb_ids)
        complement_emb = self.word_embeddings(complement_ids)
        
        # Process through LSTMs
        subject_out, _ = self.subject_encoder(subject_emb)
        verb_out, _ = self.verb_encoder(verb_emb)
        complement_out, _ = self.complement_encoder(complement_emb)
        
        # Use last hidden state
        subject_features = subject_out[:, -1, :]  # (batch, hidden_dim)
        verb_features = verb_out[:, -1, :]
        complement_features = complement_out[:, -1, :]
        
        # Encode grammar features
        domain_emb = self.domain_embeddings(domain_ids)
        complexity_emb = self.complexity_encoder(complexity_ids)
        voice_emb = self.voice_encoder(voice_ids)
        tense_emb = self.tense_encoder(tense_ids)
        
        # Process quality score
        quality_emb = self.quality_processor(quality_scores)
        
        # Fuse all features
        fused_features = torch.cat([
            subject_features, verb_features, complement_features,
            domain_emb, complexity_emb, voice_emb, tense_emb, quality_emb
        ], dim=-1)
        
        pattern_encoding = self.pattern_fusion(fused_features)
        
        # Generate predictions
        domain_logits = self.pattern_classifier(pattern_encoding)
        complexity_logits = self.complexity_predictor(pattern_encoding)
        quality_pred = self.quality_predictor(pattern_encoding)
        
        return {
            'pattern_encoding': pattern_encoding,
            'domain_logits': domain_logits,
            'complexity_logits': complexity_logits,
            'quality_prediction': quality_pred,
            'domain_probabilities': F.softmax(domain_logits, dim=-1),
            'complexity_probabilities': F.softmax(complexity_logits, dim=-1)
        }

class GrammarPatternDataset(torch.utils.data.Dataset):
    """
    Dataset for grammar pattern processing
    """
    
    def __init__(self, patterns_file: str, max_patterns_per_domain: int = 10000,
                 vocab_size: int = 50000, device: str = 'cpu'):
        self.patterns_file = Path(patterns_file)
        self.max_patterns_per_domain = max_patterns_per_domain
        self.vocab_size = vocab_size
        self.device = device
        
        # Load and process patterns
        self.patterns_by_domain = self._load_patterns()
        self.vocab, self.word_to_idx = self._build_vocabulary()
        self.domain_to_idx = self._build_domain_mapping()
        self.complexity_to_idx = self._build_complexity_mapping()
        self.voice_to_idx = self._build_voice_mapping()
        self.tense_to_idx = self._build_tense_mapping()
        
        # Create pattern list for sampling
        self.all_patterns = []
        for domain_patterns in self.patterns_by_domain.values():
            self.all_patterns.extend(domain_patterns.patterns[:max_patterns_per_domain])
        
        print(f"Loaded {len(self.all_patterns)} grammar patterns from {len(self.patterns_by_domain)} domains")
    
    def _load_patterns(self) -> Dict[str, DomainPatterns]:
        """Load patterns from JSON file"""
        patterns_by_domain = {}
        
        try:
            with open(self.patterns_file, 'r') as f:
                data = json.load(f)
            
            for domain_name, patterns_list in data.items():
                # Convert to GrammarPattern objects
                patterns = []
                for pattern_dict in patterns_list[:self.max_patterns_per_domain]:
                    try:
                        pattern = GrammarPattern(**pattern_dict)
                        patterns.append(pattern)
                    except Exception as e:
                        logging.warning(f"Failed to parse pattern: {e}")
                        continue
                
                if patterns:
                    patterns_by_domain[domain_name] = DomainPatterns(domain_name, patterns)
                    print(f"Loaded {len(patterns)} patterns for domain '{domain_name}'")
        
        except Exception as e:
            logging.error(f"Failed to load patterns: {e}")
            return {}
        
        return patterns_by_domain
    
    def _build_vocabulary(self) -> Tuple[List[str], Dict[str, int]]:
        """Build vocabulary from all patterns"""
        word_counts = Counter()
        
        for domain_patterns in self.patterns_by_domain.values():
            for pattern in domain_patterns.patterns:
                # Tokenize sentence
                words = re.findall(r'\b\w+\b', pattern.sentence.lower())
                word_counts.update(words)
                
                # Add subject, verb, complement
                for text in [pattern.subject, pattern.verb, pattern.complement]:
                    if text:
                        words = re.findall(r'\b\w+\b', text.lower())
                        word_counts.update(words)
        
        # Keep most frequent words
        most_common = word_counts.most_common(self.vocab_size - 2)  # Reserve for PAD and UNK
        vocab = ['<PAD>', '<UNK>'] + [word for word, _ in most_common]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        return vocab, word_to_idx
    
    def _build_domain_mapping(self) -> Dict[str, int]:
        """Build domain to index mapping"""
        domains = list(self.patterns_by_domain.keys())
        return {domain: idx for idx, domain in enumerate(domains)}
    
    def _build_complexity_mapping(self) -> Dict[str, int]:
        """Build complexity to index mapping"""
        complexities = [c.value for c in GrammarComplexity]
        return {comp: idx for idx, comp in enumerate(complexities)}
    
    def _build_voice_mapping(self) -> Dict[str, int]:
        """Build voice to index mapping"""
        voices = [v.value for v in VoiceType]
        return {voice: idx for idx, voice in enumerate(voices)}
    
    def _build_tense_mapping(self) -> Dict[str, int]:
        """Build tense to index mapping"""
        tenses = [t.value for t in TenseType]
        return {tense: idx for idx, tense in enumerate(tenses)}
    
    def _text_to_ids(self, text: str, max_length: int = 20) -> torch.Tensor:
        """Convert text to token IDs"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_ids = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(word_ids) > max_length:
            word_ids = word_ids[:max_length]
        else:
            word_ids.extend([self.word_to_idx['<PAD>']] * (max_length - len(word_ids)))
        
        return torch.tensor(word_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.all_patterns)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pattern = self.all_patterns[idx]
        
        # Convert text to IDs
        subject_ids = self._text_to_ids(pattern.subject)
        verb_ids = self._text_to_ids(pattern.verb)
        complement_ids = self._text_to_ids(pattern.complement)
        
        # Convert categorical features
        domain_id = torch.tensor(self.domain_to_idx[pattern.domain], dtype=torch.long)
        complexity_id = torch.tensor(self.complexity_to_idx[pattern.complexity], dtype=torch.long)
        voice_id = torch.tensor(self.voice_to_idx[pattern.voice], dtype=torch.long)
        tense_id = torch.tensor(self.tense_to_idx[pattern.tense], dtype=torch.long)
        
        # Quality score
        quality_score = torch.tensor([pattern.quality_score], dtype=torch.float32)
        
        return {
            'subject_ids': subject_ids,
            'verb_ids': verb_ids,
            'complement_ids': complement_ids,
            'domain_ids': domain_id,
            'complexity_ids': complexity_id,
            'voice_ids': voice_id,
            'tense_ids': tense_id,
            'quality_scores': quality_score,
            'metadata': {
                'sentence': pattern.sentence,
                'domain': pattern.domain,
                'complexity': pattern.complexity,
                'voice': pattern.voice,
                'tense': pattern.tense,
                'quality_score': pattern.quality_score,
                'source': pattern.source
            }
        }

class GrammarAwareTextProcessor(nn.Module):
    """
    Grammar-aware text processor for enhanced understanding
    """
    
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 256,
                 hidden_dim: int = 512, num_domains: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        
        # Grammar pattern encoder
        self.pattern_encoder = GrammarPatternEncoder(
            vocab_size, embedding_dim, hidden_dim, num_domains
        )
        
        # Text encoder
        self.text_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Grammar-text fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output heads
        self.domain_classifier = nn.Linear(hidden_dim, num_domains)
        self.complexity_analyzer = nn.Linear(hidden_dim, len(GrammarComplexity))
        self.quality_assessor = nn.Linear(hidden_dim, 1)
        self.grammar_corrector = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, text_ids: torch.Tensor, pattern_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Process text with grammar awareness"""
        
        # Encode text
        text_emb = self.pattern_encoder.word_embeddings(text_ids)
        text_out, _ = self.text_encoder(text_emb)
        text_features = text_out[:, -1, :]  # Use last hidden state
        
        if pattern_data is not None:
            # Encode grammar patterns
            pattern_result = self.pattern_encoder(pattern_data)
            pattern_features = pattern_result['pattern_encoding']
            
            # Fuse text and pattern features
            fused_features = torch.cat([text_features, pattern_features], dim=-1)
            enhanced_features = self.fusion_network(fused_features)
        else:
            enhanced_features = text_features
        
        # Generate outputs
        domain_logits = self.domain_classifier(enhanced_features)
        complexity_logits = self.complexity_analyzer(enhanced_features)
        quality_pred = self.quality_assessor(enhanced_features)
        grammar_corrections = self.grammar_corrector(enhanced_features)
        
        return {
            'enhanced_features': enhanced_features,
            'domain_logits': domain_logits,
            'complexity_logits': complexity_logits,
            'quality_prediction': quality_pred,
            'grammar_corrections': grammar_corrections,
            'domain_probabilities': F.softmax(domain_logits, dim=-1),
            'complexity_probabilities': F.softmax(complexity_logits, dim=-1)
        }

class GrammarIntegrationEngine:
    """
    Complete grammar pattern integration engine
    """
    
    def __init__(self, self_awareness_engine, device: str = 'mps'):
        self.engine = self_awareness_engine
        self.device = device
        
        # Core components
        self.text_processor = GrammarAwareTextProcessor()
        self.pattern_dataset = None
        self.domain_patterns = {}
        
        # Grammar statistics
        self.grammar_stats = {
            'patterns_processed': 0,
            'domains_analyzed': 0,
            'complexity_distribution': {},
            'quality_improvements': []
        }
    
    def load_grammar_patterns(self, patterns_file: str, max_patterns_per_domain: int = 10000):
        """Load grammar patterns from file"""
        try:
            self.pattern_dataset = GrammarPatternDataset(
                patterns_file, max_patterns_per_domain, device=self.device
            )
            
            # Store domain patterns for analysis
            self.domain_patterns = self.pattern_dataset.patterns_by_domain
            
            self.grammar_stats['patterns_processed'] = len(self.pattern_dataset)
            self.grammar_stats['domains_analyzed'] = len(self.domain_patterns)
            
            print(f"Loaded grammar patterns: {len(self.pattern_dataset)} patterns from {len(self.domain_patterns)} domains")
            
            # Compute overall statistics
            self._compute_grammar_statistics()
            
        except Exception as e:
            logging.error(f"Failed to load grammar patterns: {e}")
            return False
        
        return True
    
    def _compute_grammar_statistics(self):
        """Compute comprehensive grammar statistics"""
        if not self.domain_patterns:
            return
        
        # Aggregate complexity distribution
        all_complexities = []
        for domain_patterns in self.domain_patterns.values():
            all_complexities.extend([p.complexity for p in domain_patterns.patterns])
        
        complexity_counts = Counter(all_complexities)
        total = len(all_complexities)
        self.grammar_stats['complexity_distribution'] = {
            comp: count / total for comp, count in complexity_counts.items()
        }
    
    def analyze_text_grammar(self, text: str, domain_hint: Optional[str] = None) -> Dict[str, Any]:
        """Analyze grammar patterns in text"""
        
        if not self.pattern_dataset:
            return {'error': 'Grammar patterns not loaded'}
        
        # Tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        text_ids = torch.tensor([
            self.pattern_dataset.word_to_idx.get(word, self.pattern_dataset.word_to_idx['<UNK>'])
            for word in words
        ], dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Pad to fixed length
        max_length = 50
        if text_ids.size(1) > max_length:
            text_ids = text_ids[:, :max_length]
        else:
            padding = torch.zeros(1, max_length - text_ids.size(1), dtype=torch.long, device=self.device)
            text_ids = torch.cat([text_ids, padding], dim=1)
        
        # Process through grammar-aware processor
        with torch.no_grad():
            result = self.text_processor(text_ids)
        
        # Convert to CPU and extract results
        domain_probs = result['domain_probabilities'].cpu().numpy()[0]
        complexity_probs = result['complexity_probabilities'].cpu().numpy()[0]
        quality_pred = result['quality_prediction'].cpu().item()
        
        # Get domain names
        domain_names = list(self.pattern_dataset.domain_to_idx.keys())
        predicted_domain = domain_names[np.argmax(domain_probs)]
        
        # Get complexity names
        complexity_names = [c.value for c in GrammarComplexity]
        predicted_complexity = complexity_names[np.argmax(complexity_probs)]
        
        # Integrate with self-awareness
        awareness_state = result['enhanced_features'].squeeze(0).cpu()
        
        # Ensure correct dimension for self-awareness engine
        if awareness_state.size(0) > self.engine.state_dim:
            awareness_state = awareness_state[:self.engine.state_dim]
        elif awareness_state.size(0) < self.engine.state_dim:
            padding = self.engine.state_dim - awareness_state.size(0)
            awareness_state = F.pad(awareness_state, (0, padding))
        
        awareness_result = self.engine.process_experience(
            awareness_state,
            context={
                'task': 'grammar_analysis',
                'text': text,
                'predicted_domain': predicted_domain,
                'predicted_complexity': predicted_complexity,
                'quality_score': quality_pred
            }
        )
        
        return {
            'text_analysis': {
                'predicted_domain': predicted_domain,
                'domain_confidence': float(np.max(domain_probs)),
                'domain_probabilities': {
                    domain: float(prob) for domain, prob in zip(domain_names, domain_probs)
                },
                'predicted_complexity': predicted_complexity,
                'complexity_confidence': float(np.max(complexity_probs)),
                'quality_score': quality_pred,
                'word_count': len(words)
            },
            'awareness_result': {
                'awareness_level': awareness_result['awareness_level'].name,
                'confidence': awareness_result['introspection']['confidence'].item(),
                'consciousness_gate': awareness_result['consciousness_gate']
            }
        }
    
    def generate_grammar_aware_text(self, prompt: str, target_domain: str, 
                                   target_complexity: str = "simple") -> Dict[str, Any]:
        """Generate text with specific grammar patterns"""
        
        if not self.pattern_dataset:
            return {'error': 'Grammar patterns not loaded'}
        
        # Find patterns matching criteria
        matching_patterns = []
        for domain_patterns in self.domain_patterns.values():
            if domain_patterns.domain_name == target_domain:
                for pattern in domain_patterns.patterns:
                    if pattern.complexity == target_complexity and pattern.quality_score > 0.7:
                        matching_patterns.append(pattern)
        
        if not matching_patterns:
            return {'error': f'No patterns found for domain {target_domain} and complexity {target_complexity}'}
        
        # Select high-quality pattern
        best_pattern = max(matching_patterns, key=lambda p: p.quality_score)
        
        # Generate text based on pattern
        generated_text = f"{best_pattern.subject} {best_pattern.verb} {best_pattern.complement}."
        
        # Analyze generated text
        analysis = self.analyze_text_grammar(generated_text)
        
        return {
            'generated_text': generated_text,
            'source_pattern': {
                'sentence': best_pattern.sentence,
                'domain': best_pattern.domain,
                'complexity': best_pattern.complexity,
                'quality_score': best_pattern.quality_score
            },
            'analysis': analysis
        }
    
    def get_grammar_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive grammar insights report"""
        
        report = {
            'grammar_statistics': self.grammar_stats,
            'domain_analysis': {},
            'self_awareness_report': self.engine.get_self_report(),
            'integration_capabilities': {
                'pattern_analysis': True,
                'domain_classification': True,
                'complexity_assessment': True,
                'quality_evaluation': True,
                'grammar_aware_generation': True
            }
        }
        
        # Add domain-specific insights
        if self.domain_patterns:
            for domain_name, domain_patterns in self.domain_patterns.items():
                report['domain_analysis'][domain_name] = {
                    'pattern_count': len(domain_patterns.patterns),
                    'statistics': domain_patterns.pattern_stats,
                    'top_quality_patterns': [
                        {
                            'sentence': p.sentence,
                            'quality_score': p.quality_score,
                            'complexity': p.complexity
                        }
                        for p in sorted(domain_patterns.patterns, 
                                      key=lambda x: x.quality_score, reverse=True)[:5]
                    ]
                }
        
        return report

# Example usage
def create_grammar_integration_example():
    """Create example for grammar pattern integration"""
    example_code = """
# Example: Grammar Pattern Integration

from aura.neural.self_awareness import SelfAwarenessEngine
from aura.datasets.grammar_integration import GrammarIntegrationEngine

# 1. Initialize self-awareness engine
engine_config = {
    'state_dim': 512,
    'thought_dim': 128,
    'awareness_threshold': 0.7,
    'learning_config': {
        'batch_size': 4,
        'meta_learning': True
    }
}

engine = SelfAwarenessEngine('grammar_consciousness', engine_config)
engine.initialize()

# 2. Create grammar integration engine
grammar_engine = GrammarIntegrationEngine(engine, device='mps')

# 3. Load grammar patterns
success = grammar_engine.load_grammar_patterns(
    'pretrain_tests/spacy_svc_patterns_by_domain.json',
    max_patterns_per_domain=5000  # Limit for memory efficiency
)

if success:
    print("Grammar patterns loaded successfully!")
    
    # 4. Analyze text grammar
    text = "The scientist conducted experiments to test the hypothesis."
    analysis = grammar_engine.analyze_text_grammar(text)
    print(f"Grammar analysis: {analysis}")
    
    # 5. Generate grammar-aware text
    generated = grammar_engine.generate_grammar_aware_text(
        prompt="Write about science",
        target_domain="science",
        target_complexity="complex"
    )
    print(f"Generated text: {generated}")
    
    # 6. Get comprehensive insights
    insights = grammar_engine.get_grammar_insights_report()
    print(f"Grammar insights: {insights}")
"""
    return example_code

example_grammar = create_grammar_integration_example()
with open('grammar_integration_example.py', 'w') as f:
    f.write(example_grammar)
