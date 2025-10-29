# SPDX-License-Identifier: Apache-2.0
"""
AURA Instruction Dataset Integration: Advanced Instruction Following and Generation
- 55K instruction dataset processing and analysis
- Instruction complexity and domain classification
- Response quality assessment and improvement
- Instruction-aware learning and adaptation
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
from transformers import AutoTokenizer, AutoModel
import spacy

class InstructionComplexity(Enum):
    """Instruction complexity levels"""
    SIMPLE = "simple"           # Basic factual questions
    MODERATE = "moderate"       # Multi-step reasoning
    COMPLEX = "complex"         # Creative/analytical tasks
    EXPERT = "expert"          # Domain-specific expertise

class InstructionDomain(Enum):
    """Instruction domains"""
    GENERAL = "general"
    SCIENCE = "science"
    HEALTH = "health"
    TECHNOLOGY = "technology"
    EDUCATION = "education"
    CREATIVE = "creative"
    PROBLEM_SOLVING = "problem_solving"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    ADVICE = "advice"

class ResponseQuality(Enum):
    """Response quality levels"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class InstructionSample:
    """Individual instruction sample structure"""
    prompt: str
    response: str
    text: str  # Formatted conversation
    
    # Derived features
    prompt_length: int = 0
    response_length: int = 0
    complexity: InstructionComplexity = InstructionComplexity.SIMPLE
    domain: InstructionDomain = InstructionDomain.GENERAL
    quality_score: float = 0.0
    
    def __post_init__(self):
        """Compute derived features"""
        self.prompt_length = len(self.prompt.split())
        self.response_length = len(self.response.split())
        
        # Analyze complexity
        self.complexity = self._analyze_complexity()
        
        # Analyze domain
        self.domain = self._analyze_domain()
        
        # Assess quality
        self.quality_score = self._assess_quality()
    
    def _analyze_complexity(self) -> InstructionComplexity:
        """Analyze instruction complexity"""
        prompt_lower = self.prompt.lower()
        
        # Expert level indicators
        expert_keywords = ['analyze', 'evaluate', 'critique', 'design', 'develop', 'create', 'synthesize']
        if any(keyword in prompt_lower for keyword in expert_keywords):
            return InstructionComplexity.EXPERT
        
        # Complex level indicators
        complex_keywords = ['explain', 'describe', 'compare', 'contrast', 'discuss', 'why', 'how']
        if any(keyword in prompt_lower for keyword in complex_keywords):
            return InstructionComplexity.COMPLEX
        
        # Moderate level indicators
        moderate_keywords = ['list', 'give', 'provide', 'tell', 'what are']
        if any(keyword in prompt_lower for keyword in moderate_keywords):
            return InstructionComplexity.MODERATE
        
        return InstructionComplexity.SIMPLE
    
    def _analyze_domain(self) -> InstructionDomain:
        """Analyze instruction domain"""
        prompt_lower = self.prompt.lower()
        
        # Science domain
        science_keywords = ['atom', 'molecule', 'chemical', 'physics', 'biology', 'experiment', 'research']
        if any(keyword in prompt_lower for keyword in science_keywords):
            return InstructionDomain.SCIENCE
        
        # Health domain
        health_keywords = ['health', 'disease', 'medicine', 'exercise', 'diet', 'nutrition', 'medical']
        if any(keyword in prompt_lower for keyword in health_keywords):
            return InstructionDomain.HEALTH
        
        # Technology domain
        tech_keywords = ['computer', 'software', 'programming', 'algorithm', 'data', 'digital', 'tech']
        if any(keyword in prompt_lower for keyword in tech_keywords):
            return InstructionDomain.TECHNOLOGY
        
        # Education domain
        edu_keywords = ['learn', 'teach', 'study', 'education', 'school', 'student', 'academic']
        if any(keyword in prompt_lower for keyword in edu_keywords):
            return InstructionDomain.EDUCATION
        
        # Creative domain
        creative_keywords = ['write', 'story', 'poem', 'creative', 'imagine', 'art', 'design']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return InstructionDomain.CREATIVE
        
        # Problem solving domain
        problem_keywords = ['solve', 'problem', 'solution', 'fix', 'troubleshoot', 'debug']
        if any(keyword in prompt_lower for keyword in problem_keywords):
            return InstructionDomain.PROBLEM_SOLVING
        
        # Explanation domain
        explanation_keywords = ['explain', 'describe', 'how does', 'what is', 'define']
        if any(keyword in prompt_lower for keyword in explanation_keywords):
            return InstructionDomain.EXPLANATION
        
        # Analysis domain
        analysis_keywords = ['analyze', 'evaluate', 'assess', 'compare', 'contrast']
        if any(keyword in prompt_lower for keyword in analysis_keywords):
            return InstructionDomain.ANALYSIS
        
        # Advice domain
        advice_keywords = ['advice', 'recommend', 'suggest', 'tips', 'help', 'guidance']
        if any(keyword in prompt_lower for keyword in advice_keywords):
            return InstructionDomain.ADVICE
        
        return InstructionDomain.GENERAL
    
    def _assess_quality(self) -> float:
        """Assess response quality based on various factors"""
        quality_score = 0.0
        
        # Length factor (longer responses often more comprehensive)
        length_score = min(1.0, self.response_length / 100.0)
        quality_score += length_score * 0.2
        
        # Structure factor (numbered lists, paragraphs)
        structure_indicators = ['1.', '2.', '3.', '\n\n', '•', '-']
        structure_count = sum(1 for indicator in structure_indicators if indicator in self.response)
        structure_score = min(1.0, structure_count / 5.0)
        quality_score += structure_score * 0.2
        
        # Detail factor (specific examples, explanations)
        detail_indicators = ['for example', 'specifically', 'in particular', 'such as', 'including']
        detail_count = sum(1 for indicator in detail_indicators if indicator.lower() in self.response.lower())
        detail_score = min(1.0, detail_count / 3.0)
        quality_score += detail_score * 0.2
        
        # Completeness factor (addresses the question)
        prompt_words = set(self.prompt.lower().split())
        response_words = set(self.response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        completeness_score = min(1.0, overlap / max(1, len(prompt_words)))
        quality_score += completeness_score * 0.2
        
        # Clarity factor (clear language, proper grammar)
        clarity_indicators = ['clearly', 'specifically', 'precisely', 'exactly']
        clarity_count = sum(1 for indicator in clarity_indicators if indicator in self.response.lower())
        clarity_score = min(1.0, clarity_count / 2.0)
        quality_score += clarity_score * 0.2
        
        return min(1.0, quality_score)

class InstructionDataset(torch.utils.data.Dataset):
    """
    Dataset for instruction following and generation
    """
    
    def __init__(self, dataset_path: str, max_samples: int = 10000, 
                 tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.dataset_path = Path(dataset_path)
        self.max_samples = max_samples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoder = AutoModel.from_pretrained(tokenizer_name)
        
        # Load and process instruction data
        self.samples = self._load_instruction_data()
        
        # Build vocabulary and mappings
        self.vocab, self.word_to_idx = self._build_vocabulary()
        self.domain_to_idx = self._build_domain_mapping()
        self.complexity_to_idx = self._build_complexity_mapping()
        
        # Compute dataset statistics
        self.dataset_stats = self._compute_dataset_statistics()
        
        print(f"Loaded {len(self.samples)} instruction samples")
        print(f"Dataset statistics: {self.dataset_stats}")
    
    def _load_instruction_data(self) -> List[InstructionSample]:
        """Load instruction data from JSONL file"""
        samples = []
        
        try:
            with open(self.dataset_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= self.max_samples:
                        break
                    
                    if line.strip():
                        try:
                            data = json.loads(line)
                            sample = InstructionSample(
                                prompt=data['prompt'],
                                response=data['response'],
                                text=data['text']
                            )
                            samples.append(sample)
                        except Exception as e:
                            logging.warning(f"Failed to parse line {i}: {e}")
                            continue
        
        except Exception as e:
            logging.error(f"Failed to load instruction data: {e}")
            return []
        
        return samples
    
    def _build_vocabulary(self) -> Tuple[List[str], Dict[str, int]]:
        """Build vocabulary from all samples"""
        word_counts = Counter()
        
        for sample in self.samples:
            # Tokenize prompt and response
            prompt_words = re.findall(r'\b\w+\b', sample.prompt.lower())
            response_words = re.findall(r'\b\w+\b', sample.response.lower())
            word_counts.update(prompt_words + response_words)
        
        # Keep most frequent words
        most_common = word_counts.most_common(50000 - 2)  # Reserve for PAD and UNK
        vocab = ['<PAD>', '<UNK>'] + [word for word, _ in most_common]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        return vocab, word_to_idx
    
    def _build_domain_mapping(self) -> Dict[str, int]:
        """Build domain to index mapping"""
        domains = [d.value for d in InstructionDomain]
        return {domain: idx for idx, domain in enumerate(domains)}
    
    def _build_complexity_mapping(self) -> Dict[str, int]:
        """Build complexity to index mapping"""
        complexities = [c.value for c in InstructionComplexity]
        return {complexity: idx for idx, complexity in enumerate(complexities)}
    
    def _compute_dataset_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive dataset statistics"""
        if not self.samples:
            return {}
        
        # Basic statistics
        total_samples = len(self.samples)
        prompt_lengths = [s.prompt_length for s in self.samples]
        response_lengths = [s.response_length for s in self.samples]
        quality_scores = [s.quality_score for s in self.samples]
        
        # Domain distribution
        domain_counts = Counter(s.domain.value for s in self.samples)
        domain_distribution = {domain: count / total_samples for domain, count in domain_counts.items()}
        
        # Complexity distribution
        complexity_counts = Counter(s.complexity.value for s in self.samples)
        complexity_distribution = {complexity: count / total_samples for complexity, count in complexity_counts.items()}
        
        return {
            'total_samples': total_samples,
            'prompt_length_stats': {
                'mean': np.mean(prompt_lengths),
                'std': np.std(prompt_lengths),
                'min': np.min(prompt_lengths),
                'max': np.max(prompt_lengths)
            },
            'response_length_stats': {
                'mean': np.mean(response_lengths),
                'std': np.std(response_lengths),
                'min': np.min(response_lengths),
                'max': np.max(response_lengths)
            },
            'quality_stats': {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'domain_distribution': domain_distribution,
            'complexity_distribution': complexity_distribution
        }
    
    def _text_to_ids(self, text: str, max_length: int = 128) -> torch.Tensor:
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
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert text to IDs
        prompt_ids = self._text_to_ids(sample.prompt, max_length=64)
        response_ids = self._text_to_ids(sample.response, max_length=128)
        
        # Convert categorical features
        domain_id = torch.tensor(self.domain_to_idx[sample.domain.value], dtype=torch.long)
        complexity_id = torch.tensor(self.complexity_to_idx[sample.complexity.value], dtype=torch.long)
        
        # Quality score
        quality_score = torch.tensor([sample.quality_score], dtype=torch.float32)
        
        return {
            'prompt_ids': prompt_ids,
            'response_ids': response_ids,
            'domain_ids': domain_id,
            'complexity_ids': complexity_id,
            'quality_scores': quality_score,
            'metadata': {
                'prompt': sample.prompt,
                'response': sample.response,
                'domain': sample.domain.value,
                'complexity': sample.complexity.value,
                'quality_score': sample.quality_score,
                'prompt_length': sample.prompt_length,
                'response_length': sample.response_length
            }
        }

class InstructionEncoder(nn.Module):
    """
    Neural encoder for instruction understanding and generation
    """
    
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_domains: int = 10, num_complexities: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.num_complexities = num_complexities
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Domain and complexity embeddings
        self.domain_embeddings = nn.Embedding(num_domains, embedding_dim)
        self.complexity_embeddings = nn.Embedding(num_complexities, embedding_dim)
        
        # Instruction encoders
        self.prompt_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.response_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Cross-attention between prompt and response
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Quality assessment
        self.quality_assessor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )
        
        # Complexity classifier
        self.complexity_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_complexities)
        )
        
        # Response generator
        self.response_generator = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.response_decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                domain_ids: torch.Tensor, complexity_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode instruction and response"""
        
        # Encode prompt and response
        prompt_emb = self.word_embeddings(prompt_ids)
        response_emb = self.word_embeddings(response_ids)
        
        prompt_out, _ = self.prompt_encoder(prompt_emb)
        response_out, _ = self.response_encoder(response_emb)
        
        # Use last hidden state
        prompt_features = prompt_out[:, -1, :]  # (batch, hidden_dim)
        response_features = response_out[:, -1, :]
        
        # Cross-attention between prompt and response
        attended_response, attention_weights = self.cross_attention(
            response_out, prompt_out, prompt_out
        )
        attended_response = attended_response[:, -1, :]  # Use last attended state
        
        # Combine features
        combined_features = torch.cat([prompt_features, attended_response], dim=-1)
        
        # Generate predictions
        quality_pred = self.quality_assessor(combined_features)
        domain_logits = self.domain_classifier(combined_features)
        complexity_logits = self.complexity_classifier(combined_features)
        
        # Generate response (for training)
        if self.training:
            response_gen_out, _ = self.response_generator(
                torch.cat([prompt_emb, response_emb], dim=-1)
            )
            response_logits = self.response_decoder(response_gen_out)
        else:
            response_logits = None
        
        return {
            'prompt_features': prompt_features,
            'response_features': response_features,
            'combined_features': combined_features,
            'quality_prediction': quality_pred,
            'domain_logits': domain_logits,
            'complexity_logits': complexity_logits,
            'response_logits': response_logits,
            'attention_weights': attention_weights,
            'domain_probabilities': F.softmax(domain_logits, dim=-1),
            'complexity_probabilities': F.softmax(complexity_logits, dim=-1)
        }

class InstructionFollowingEngine:
    """
    Complete instruction following and generation engine
    """
    
    def __init__(self, self_awareness_engine, device: str = 'mps'):
        self.engine = self_awareness_engine
        self.device = device
        
        # Core components
        self.instruction_encoder = InstructionEncoder()
        self.instruction_dataset = None
        
        # Instruction statistics
        self.instruction_stats = {
            'samples_processed': 0,
            'domains_analyzed': 0,
            'complexity_distribution': {},
            'quality_improvements': []
        }
    
    def load_instruction_dataset(self, dataset_path: str, max_samples: int = 10000):
        """Load instruction dataset"""
        try:
            self.instruction_dataset = InstructionDataset(
                dataset_path, max_samples
            )
            
            self.instruction_stats['samples_processed'] = len(self.instruction_dataset)
            self.instruction_stats['domains_analyzed'] = len(self.instruction_dataset.dataset_stats.get('domain_distribution', {}))
            self.instruction_stats['complexity_distribution'] = self.instruction_dataset.dataset_stats.get('complexity_distribution', {})
            
            print(f"Loaded instruction dataset: {len(self.instruction_dataset)} samples")
            
        except Exception as e:
            logging.error(f"Failed to load instruction dataset: {e}")
            return False
        
        return True
    
    def analyze_instruction(self, prompt: str, response: str = None) -> Dict[str, Any]:
        """Analyze instruction complexity and domain"""
        
        if not self.instruction_dataset:
            return {'error': 'Instruction dataset not loaded'}
        
        # Create sample for analysis
        sample = InstructionSample(
            prompt=prompt,
            response=response or "",
            text=f"<human>: {prompt}\n<bot>: {response or ''}"
        )
        
        # Convert to tensor format
        prompt_ids = self.instruction_dataset._text_to_ids(prompt, max_length=64)
        response_ids = self.instruction_dataset._text_to_ids(response or "", max_length=128)
        domain_id = torch.tensor(self.instruction_dataset.domain_to_idx[sample.domain.value], dtype=torch.long)
        complexity_id = torch.tensor(self.instruction_dataset.complexity_to_idx[sample.complexity.value], dtype=torch.long)
        
        # Add batch dimension
        prompt_ids = prompt_ids.unsqueeze(0).to(self.device)
        response_ids = response_ids.unsqueeze(0).to(self.device)
        domain_id = domain_id.unsqueeze(0).to(self.device)
        complexity_id = complexity_id.unsqueeze(0).to(self.device)
        
        # Process through encoder
        with torch.no_grad():
            result = self.instruction_encoder(prompt_ids, response_ids, domain_id, complexity_id)
        
        # Extract results
        domain_probs = result['domain_probabilities'].cpu().numpy()[0]
        complexity_probs = result['complexity_probabilities'].cpu().numpy()[0]
        quality_pred = result['quality_prediction'].cpu().item()
        
        # Get domain and complexity names
        domain_names = list(self.instruction_dataset.domain_to_idx.keys())
        complexity_names = list(self.instruction_dataset.complexity_to_idx.keys())
        
        predicted_domain = domain_names[np.argmax(domain_probs)]
        predicted_complexity = complexity_names[np.argmax(complexity_probs)]
        
        # Integrate with self-awareness
        awareness_state = result['combined_features'].squeeze(0).cpu()
        
        # Ensure correct dimension for self-awareness engine
        if awareness_state.size(0) > self.engine.state_dim:
            awareness_state = awareness_state[:self.engine.state_dim]
        elif awareness_state.size(0) < self.engine.state_dim:
            padding = self.engine.state_dim - awareness_state.size(0)
            awareness_state = F.pad(awareness_state, (0, padding))
        
        awareness_result = self.engine.process_experience(
            awareness_state,
            context={
                'task': 'instruction_analysis',
                'prompt': prompt,
                'response': response,
                'predicted_domain': predicted_domain,
                'predicted_complexity': predicted_complexity,
                'quality_score': quality_pred
            }
        )
        
        return {
            'instruction_analysis': {
                'predicted_domain': predicted_domain,
                'domain_confidence': float(np.max(domain_probs)),
                'domain_probabilities': {
                    domain: float(prob) for domain, prob in zip(domain_names, domain_probs)
                },
                'predicted_complexity': predicted_complexity,
                'complexity_confidence': float(np.max(complexity_probs)),
                'complexity_probabilities': {
                    complexity: float(prob) for complexity, prob in zip(complexity_names, complexity_probs)
                },
                'quality_score': quality_pred,
                'prompt_length': sample.prompt_length,
                'response_length': sample.response_length
            },
            'awareness_result': {
                'awareness_level': awareness_result['awareness_level'].name,
                'confidence': awareness_result['introspection']['confidence'].item(),
                'consciousness_gate': awareness_result['consciousness_gate']
            }
        }
    
    def generate_response(self, prompt: str, target_domain: str = None, 
                         target_complexity: str = None) -> Dict[str, Any]:
        """Generate response for instruction"""
        
        if not self.instruction_dataset:
            return {'error': 'Instruction dataset not loaded'}
        
        # Find similar instructions in dataset
        similar_samples = self._find_similar_instructions(prompt, target_domain, target_complexity)
        
        if not similar_samples:
            return {'error': 'No similar instructions found'}
        
        # Select best sample as template
        best_sample = max(similar_samples, key=lambda s: s.quality_score)
        
        # Generate response based on template
        generated_response = self._generate_response_from_template(prompt, best_sample)
        
        # Analyze generated response
        analysis = self.analyze_instruction(prompt, generated_response)
        
        return {
            'generated_response': generated_response,
            'template_sample': {
                'prompt': best_sample.prompt,
                'response': best_sample.response,
                'domain': best_sample.domain.value,
                'complexity': best_sample.complexity.value,
                'quality_score': best_sample.quality_score
            },
            'analysis': analysis
        }
    
    def _find_similar_instructions(self, prompt: str, target_domain: str = None, 
                                 target_complexity: str = None) -> List[InstructionSample]:
        """Find similar instructions in dataset"""
        similar_samples = []
        
        for sample in self.instruction_dataset.samples:
            # Filter by domain if specified
            if target_domain and sample.domain.value != target_domain:
                continue
            
            # Filter by complexity if specified
            if target_complexity and sample.complexity.value != target_complexity:
                continue
            
            # Simple similarity based on word overlap
            prompt_words = set(prompt.lower().split())
            sample_words = set(sample.prompt.lower().split())
            overlap = len(prompt_words.intersection(sample_words))
            similarity = overlap / max(1, len(prompt_words))
            
            if similarity > 0.1:  # Minimum similarity threshold
                similar_samples.append(sample)
        
        return similar_samples[:10]  # Return top 10 similar samples
    
    def _generate_response_from_template(self, prompt: str, template_sample: InstructionSample) -> str:
        """Generate response using template sample"""
        # Simple template-based generation
        # In practice, this would use a more sophisticated generation model
        
        template_response = template_sample.response
        
        # Basic adaptation based on prompt keywords
        prompt_lower = prompt.lower()
        
        if 'list' in prompt_lower or 'give' in prompt_lower:
            # Format as numbered list
            lines = template_response.split('\n')
            numbered_lines = []
            for i, line in enumerate(lines, 1):
                if line.strip():
                    numbered_lines.append(f"{i}. {line.strip()}")
            return '\n'.join(numbered_lines)
        
        elif 'explain' in prompt_lower or 'describe' in prompt_lower:
            # Add explanation structure
            return f"Let me explain this step by step:\n\n{template_response}"
        
        else:
            # Return template as-is
            return template_response
    
    def get_instruction_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive instruction insights report"""
        
        report = {
            'instruction_statistics': self.instruction_stats,
            'dataset_analysis': {},
            'self_awareness_report': self.engine.get_self_report(),
            'capabilities': {
                'instruction_analysis': True,
                'domain_classification': True,
                'complexity_assessment': True,
                'quality_evaluation': True,
                'response_generation': True,
                'template_matching': True
            }
        }
        
        # Add dataset analysis if available
        if self.instruction_dataset:
            report['dataset_analysis'] = self.instruction_dataset.dataset_stats
        
        return report

# Example usage
def create_instruction_integration_example():
    """Create example for instruction dataset integration"""
    example_code = """
# Example: Instruction Dataset Integration

from aura.neural.self_awareness import SelfAwarenessEngine
from aura.datasets.instruction_integration import InstructionFollowingEngine

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

engine = SelfAwarenessEngine('instruction_consciousness', engine_config)
engine.initialize()

# 2. Create instruction following engine
instruction_engine = InstructionFollowingEngine(engine, device='mps')

# 3. Load instruction dataset
success = instruction_engine.load_instruction_dataset(
    'pretrain_tests/instruct_55k_clean.jsonl',
    max_samples=10000  # Limit for memory efficiency
)

if success:
    print("Instruction dataset loaded successfully!")
    
    # 4. Analyze instruction
    prompt = "Explain the structure of an atom"
    analysis = instruction_engine.analyze_instruction(prompt)
    print(f"Instruction analysis: {analysis}")
    
    # 5. Generate response
    generated = instruction_engine.generate_response(
        prompt="Give three tips for staying healthy",
        target_domain="health",
        target_complexity="moderate"
    )
    print(f"Generated response: {generated}")
    
    # 6. Get comprehensive insights
    insights = instruction_engine.get_instruction_insights_report()
    print(f"Instruction insights: {insights}")
"""
    return example_code

example_instruction = create_instruction_integration_example()
with open('instruction_integration_example.py', 'w') as f:
    f.write(example_instruction)
