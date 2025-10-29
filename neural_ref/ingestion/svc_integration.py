# SPDX-License-Identifier: Apache-2.0
"""
AURA SVC (Subject-Verb-Complement) Dataset Integration
Grammatical structure learning and syntactic analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import logging

@dataclass
class SVCData:
    """Single SVC grammatical structure data point"""
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

class SVCDataset(Dataset):
    """
    Dataset for Subject-Verb-Complement grammatical structure learning
    Handles syntactic analysis and grammatical pattern recognition
    """
    
    def __init__(self, dataset_path: str, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoder = AutoModel.from_pretrained(tokenizer_name)
        
        # Load and parse data
        self.data = self._load_svc_data()
        
        # Build grammatical feature encodings
        self.tense_to_id = self._build_tense_encoding()
        self.voice_to_id = self._build_voice_encoding()
        self.complexity_to_id = self._build_complexity_encoding()
        self.domain_to_id = self._build_domain_encoding()
        
        # Compute grammatical statistics
        self.grammatical_stats = self._compute_grammatical_stats()
    
    def _load_svc_data(self) -> List[SVCData]:
        """Load and parse SVC data from JSON file"""
        data = []
        
        try:
            with open(self.dataset_path, 'r') as f:
                json_data = json.load(f)
                
                # Handle both nested and flat structure
                if isinstance(json_data, dict) and 'general' in json_data:
                    svc_items = json_data['general']
                elif isinstance(json_data, list):
                    svc_items = json_data
                else:
                    svc_items = [json_data]
                
                for item in svc_items:
                    svc_data = SVCData(
                        subject=item['subject'],
                        verb=item['verb'],
                        complement=item['complement'],
                        tense=item['tense'],
                        voice=item['voice'],
                        complexity=item['complexity'],
                        domain=item['domain'],
                        quality_score=item['quality_score'],
                        source=item['source'],
                        sentence=item['sentence']
                    )
                    data.append(svc_data)
        
        except Exception as e:
            logging.error(f"Error loading SVC data: {e}")
            return []
        
        return data
    
    def _build_tense_encoding(self) -> Dict[str, int]:
        """Build encoding for grammatical tenses"""
        tenses = list(set(item.tense for item in self.data))
        return {tense: i for i, tense in enumerate(sorted(tenses))}
    
    def _build_voice_encoding(self) -> Dict[str, int]:
        """Build encoding for grammatical voice"""
        voices = list(set(item.voice for item in self.data))
        return {voice: i for i, voice in enumerate(sorted(voices))}
    
    def _build_complexity_encoding(self) -> Dict[str, int]:
        """Build encoding for sentence complexity"""
        complexities = list(set(item.complexity for item in self.data))
        return {complexity: i for i, complexity in enumerate(sorted(complexities))}
    
    def _build_domain_encoding(self) -> Dict[str, int]:
        """Build encoding for content domains"""
        domains = list(set(item.domain for item in self.data))
        return {domain: i for i, domain in enumerate(sorted(domains))}
    
    def _compute_grammatical_stats(self) -> Dict[str, Any]:
        """Compute statistics about grammatical patterns"""
        stats = {
            'total_sentences': len(self.data),
            'avg_quality_score': np.mean([item.quality_score for item in self.data]),
            'tense_distribution': {},
            'voice_distribution': {},
            'complexity_distribution': {},
            'domain_distribution': {}
        }
        
        # Compute distributions
        for category, encoding in [
            ('tense', self.tense_to_id),
            ('voice', self.voice_to_id), 
            ('complexity', self.complexity_to_id),
            ('domain', self.domain_to_id)
        ]:
            dist = {}
            for item in self.data:
                value = getattr(item, category)
                dist[value] = dist.get(value, 0) + 1
            
            # Normalize to probabilities
            total = sum(dist.values())
            stats[f'{category}_distribution'] = {k: v/total for k, v in dist.items()}
        
        return stats
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize and encode sentence components
        sentence_tokens = self.tokenizer(
            item.sentence,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        
        subject_tokens = self.tokenizer(
            item.subject,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32
        )
        
        verb_tokens = self.tokenizer(
            item.verb,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=16
        )
        
        complement_tokens = self.tokenizer(
            item.complement,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64
        )
        
        # Get semantic encodings
        with torch.no_grad():
            sentence_encoding = self.encoder(**sentence_tokens).last_hidden_state.mean(dim=1).squeeze(0)
            subject_encoding = self.encoder(**subject_tokens).last_hidden_state.mean(dim=1).squeeze(0)
            verb_encoding = self.encoder(**verb_tokens).last_hidden_state.mean(dim=1).squeeze(0)
            complement_encoding = self.encoder(**complement_tokens).last_hidden_state.mean(dim=1).squeeze(0)
        
        # Create grammatical feature vectors
        tense_onehot = torch.zeros(len(self.tense_to_id))
        tense_onehot[self.tense_to_id[item.tense]] = 1.0
        
        voice_onehot = torch.zeros(len(self.voice_to_id))
        voice_onehot[self.voice_to_id[item.voice]] = 1.0
        
        complexity_onehot = torch.zeros(len(self.complexity_to_id))
        complexity_onehot[self.complexity_to_id[item.complexity]] = 1.0
        
        domain_onehot = torch.zeros(len(self.domain_to_id))
        domain_onehot[self.domain_to_id[item.domain]] = 1.0
        
        # Combine all features
        grammatical_features = torch.cat([
            tense_onehot,
            voice_onehot,
            complexity_onehot,
            domain_onehot
        ])
        
        # Create structured representation
        svc_structure = torch.cat([
            subject_encoding,
            verb_encoding,
            complement_encoding,
            grammatical_features
        ])
        
        return {
            'input': svc_structure,
            'target': sentence_encoding,  # Reconstruct full sentence from structure
            'grammatical_features': grammatical_features,
            'quality_score': torch.tensor(item.quality_score),
            'components': {
                'subject': subject_encoding,
                'verb': verb_encoding,
                'complement': complement_encoding
            },
            'metadata': {
                'subject': item.subject,
                'verb': item.verb,
                'complement': item.complement,
                'tense': item.tense,
                'voice': item.voice,
                'complexity': item.complexity,
                'domain': item.domain,
                'sentence': item.sentence,
                'source': item.source
            }
        }

class GrammaticalAnalyzer(nn.Module):
    """
    Neural network for grammatical structure analysis and generation
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Grammatical structure encoder
        self.structure_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # subject + verb + complement
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Grammatical feature processor (will be initialized dynamically)
        self.feature_processor = None
        
        # SVC relationship modeling
        self.svc_relationship = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Sentence reconstruction
        self.sentence_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )
        
        # Grammatical correctness classifier
        self.correctness_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # simple, moderate, complex
        )
    
    def forward(self, svc_components: Dict[str, torch.Tensor], 
                grammatical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze grammatical structure and generate predictions"""
        
        # Extract components
        subject = svc_components['subject']
        verb = svc_components['verb']
        complement = svc_components['complement']
        
        # Encode SVC structure
        svc_input = torch.cat([subject, verb, complement], dim=-1)
        structure_encoding = self.structure_encoder(svc_input)
        
        # Initialize feature processor if needed
        if self.feature_processor is None:
            feature_dim = grammatical_features.size(-1)
            self.feature_processor = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ).to(grammatical_features.device)
        
        # Process grammatical features
        feature_encoding = self.feature_processor(grammatical_features)
        
        # Model SVC relationships with attention
        # Reshape for attention: [batch, seq_len=3, hidden_dim]
        batch_size = structure_encoding.size(0)
        svc_sequence = torch.stack([
            subject[:, :self.hidden_dim] if subject.size(-1) >= self.hidden_dim else F.pad(subject, (0, self.hidden_dim - subject.size(-1))),
            verb[:, :self.hidden_dim] if verb.size(-1) >= self.hidden_dim else F.pad(verb, (0, self.hidden_dim - verb.size(-1))),
            complement[:, :self.hidden_dim] if complement.size(-1) >= self.hidden_dim else F.pad(complement, (0, self.hidden_dim - complement.size(-1)))
        ], dim=1)  # (batch, 3, hidden_dim)
        
        attended_svc, attention_weights = self.svc_relationship(
            svc_sequence, svc_sequence, svc_sequence
        )
        
        # Use attended representation
        integrated_structure = attended_svc.mean(dim=1)  # Average across SVC components
        
        # Combine structure and features
        combined_features = torch.cat([integrated_structure, feature_encoding], dim=-1)
        
        # Generate outputs
        reconstructed_sentence = self.sentence_decoder(combined_features)
        correctness_score = self.correctness_classifier(combined_features)
        complexity_prediction = self.complexity_predictor(integrated_structure)
        
        return {
            'reconstructed_sentence': reconstructed_sentence,
            'correctness_score': correctness_score,
            'complexity_prediction': complexity_prediction,
            'structure_encoding': integrated_structure,
            'feature_encoding': feature_encoding,
            'attention_weights': attention_weights
        }

class SVCIntegrator:
    """
    Integrates SVC dataset with the self-awareness engine for grammatical learning
    """
    
    def __init__(self, self_awareness_engine, device: str = 'mps'):
        self.engine = self_awareness_engine
        self.device = device
        self.grammatical_analyzer = GrammaticalAnalyzer().to(device)
        
        self.svc_datasets = {}
        self.svc_dataloaders = {}
        self.grammatical_stats = {
            'parsing_accuracy': [],
            'reconstruction_quality': [],
            'grammatical_awareness': []
        }
    
    def register_svc_dataset(self, dataset_name: str, dataset_path: str):
        """Register SVC dataset for grammatical learning"""
        dataset = SVCDataset(dataset_path)
        
        self.svc_datasets[dataset_name] = dataset
        self.svc_dataloaders[dataset_name] = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        print(f"Registered SVC dataset '{dataset_name}' with {len(dataset)} grammatical structures")
        print(f"Grammatical statistics: {dataset.grammatical_stats}")
    
    def train_grammatical_awareness(self, dataset_name: str, num_epochs: int = 3) -> Dict[str, float]:
        """Train grammatical structure understanding"""
        if dataset_name not in self.svc_datasets:
            raise ValueError(f"SVC dataset {dataset_name} not registered")
        
        dataset = self.svc_datasets[dataset_name]
        dataloader = self.svc_dataloaders[dataset_name]
        
        self.grammatical_analyzer.train()
        losses = []
        parsing_accuracies = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                svc_components = {k: v.to(self.device) for k, v in batch['components'].items()}
                grammatical_features = batch['grammatical_features'].to(self.device)
                target_sentence = batch['target'].to(self.device)
                quality_scores = batch['quality_score'].to(self.device)
                
                # Forward pass through grammatical analyzer
                output = self.grammatical_analyzer(svc_components, grammatical_features)
                
                # Compute losses
                reconstruction_loss = F.mse_loss(output['reconstructed_sentence'], target_sentence)
                
                # Quality-weighted reconstruction loss
                weighted_reconstruction = (reconstruction_loss * quality_scores.unsqueeze(-1)).mean()
                
                # Grammatical correctness loss (self-supervised)
                correctness_targets = (quality_scores > 0.8).float().unsqueeze(-1)
                correctness_loss = F.binary_cross_entropy(output['correctness_score'], correctness_targets)
                
                # Combined loss
                total_loss = weighted_reconstruction + 0.1 * correctness_loss
                
                epoch_losses.append(total_loss.item())
                
                # Compute parsing accuracy
                correct_predictions = ((output['correctness_score'] > 0.5) == (quality_scores > 0.8)).float()
                accuracy = correct_predictions.mean().item()
                epoch_accuracies.append(accuracy)
                
                # Integrate with self-awareness engine
                # Create grammatical awareness state
                grammatical_state = torch.cat([
                    output['structure_encoding'].mean(dim=0),
                    output['feature_encoding'].mean(dim=0)
                ], dim=0)
                
                # Pad to match engine state dimension
                if grammatical_state.size(0) < self.engine.state_dim:
                    padding_size = self.engine.state_dim - grammatical_state.size(0)
                    grammatical_state = F.pad(grammatical_state, (0, padding_size))
                elif grammatical_state.size(0) > self.engine.state_dim:
                    grammatical_state = grammatical_state[:self.engine.state_dim]
                
                # Process through self-awareness engine
                awareness_result = self.engine.process_experience(
                    grammatical_state,
                    context={
                        'task': 'grammatical_analysis',
                        'dataset': dataset_name,
                        'batch_idx': batch_idx,
                        'epoch': epoch,
                        'parsing_accuracy': accuracy,
                        'reconstruction_quality': 1.0 - reconstruction_loss.item()
                    }
                )
                
                # Adapt learning based on awareness level
                if awareness_result['awareness_level'].value >= 3:  # Metacognitive
                    # Can handle complex grammatical structures
                    current_difficulty = 0.9
                else:
                    # Focus on simpler structures
                    current_difficulty = 0.6
                
                # Update curriculum in the engine's dataset learner
                self.engine.dataset_learner.update_curriculum(current_difficulty)
            
            avg_epoch_loss = np.mean(epoch_losses)
            avg_epoch_accuracy = np.mean(epoch_accuracies)
            losses.append(avg_epoch_loss)
            parsing_accuracies.append(avg_epoch_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, "
                  f"Parsing Accuracy: {avg_epoch_accuracy:.3f}")
        
        # Update statistics
        self.grammatical_stats['parsing_accuracy'].extend(parsing_accuracies)
        self.grammatical_stats['reconstruction_quality'].append(1.0 - np.mean(losses))
        
        return {
            'final_loss': losses[-1],
            'final_accuracy': parsing_accuracies[-1],
            'improvement': parsing_accuracies[-1] - parsing_accuracies[0] if len(parsing_accuracies) > 1 else 0.0,
            'grammatical_awareness': self.engine.current_awareness_level.value
        }
    
    def analyze_sentence_structure(self, sentence: str) -> Dict[str, Any]:
        """Analyze grammatical structure of a new sentence"""
        self.grammatical_analyzer.eval()
        
        # Simple extraction (in practice, would use proper parsing)
        words = sentence.split()
        
        # Mock SVC extraction for demonstration
        subject = words[0] if words else "Unknown"
        verb = words[1] if len(words) > 1 else "Unknown"
        complement = " ".join(words[2:]) if len(words) > 2 else "Unknown"
        
        # Encode components
        tokenizer = self.svc_datasets[list(self.svc_datasets.keys())[0]].tokenizer
        encoder = self.svc_datasets[list(self.svc_datasets.keys())[0]].encoder
        
        def encode_text(text):
            tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                encoding = encoder(**tokens).last_hidden_state.mean(dim=1).squeeze(0)
            return encoding
        
        svc_components = {
            'subject': encode_text(subject).unsqueeze(0).to(self.device),
            'verb': encode_text(verb).unsqueeze(0).to(self.device),
            'complement': encode_text(complement).unsqueeze(0).to(self.device)
        }
        
        # Mock grammatical features
        grammatical_features = torch.zeros(1, 20, device=self.device)  # All zeros for unknown
        
        with torch.no_grad():
            analysis = self.grammatical_analyzer(svc_components, grammatical_features)
        
        return {
            'extracted_svc': {
                'subject': subject,
                'verb': verb,
                'complement': complement
            },
            'correctness_score': analysis['correctness_score'].item(),
            'predicted_complexity': torch.argmax(analysis['complexity_prediction'], dim=-1).item(),
            'attention_weights': analysis['attention_weights'].cpu().numpy(),
            'structure_encoding': analysis['structure_encoding'].cpu().numpy()
        }
    
    def get_grammatical_report(self) -> Dict[str, Any]:
        """Generate comprehensive grammatical learning report"""
        report = {
            'registered_datasets': list(self.svc_datasets.keys()),
            'dataset_statistics': {},
            'learning_progress': self.grammatical_stats,
            'self_awareness_report': self.engine.get_self_report(),
            'grammatical_capabilities': {}
        }
        
        # Add dataset-specific statistics
        for name, dataset in self.svc_datasets.items():
            report['dataset_statistics'][name] = dataset.grammatical_stats
        
        # Add grammatical capability assessment
        if self.grammatical_stats['parsing_accuracy']:
            current_accuracy = np.mean(self.grammatical_stats['parsing_accuracy'][-5:])
            report['grammatical_capabilities'] = {
                'parsing_proficiency': current_accuracy,
                'structure_awareness': self.engine.current_awareness_level.name,
                'complexity_handling': 'high' if current_accuracy > 0.8 else 'moderate' if current_accuracy > 0.6 else 'basic'
            }
        
        return report
