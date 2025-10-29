# SPDX-License-Identifier: Apache-2.0
"""
AURA Neuromorphic Tokenizer
Converts discrete tokens into spike-based embeddings with temporal position encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math
from aura.neural.embedding import BinarySpikeEmbedding
from aura.neural.interfaces import AURAModule

class TemporalPositionEncoder(nn.Module):
    """
    Encode token positions into spike patterns with sinusoidal and learnable components
    Each position adds a unique temporal offset to the embedding
    """
    
    def __init__(self, embed_dim: int, max_length: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Sinusoidal position encodings (like transformer but for spikes)
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        # Create sinusoidal patterns
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable)
        self.register_buffer('pe', pe)
        
        # Learnable position embeddings for spike adaptation
        self.adaptive_position = nn.Parameter(torch.randn(max_length, embed_dim) * 0.02)
        
        # Position-aware spike threshold
        self.position_threshold = nn.Parameter(torch.ones(max_length) * 0.5)
        
    def forward(self, spike_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_embeddings: (batch, seq_len, embed_dim)
        Returns:
            temporal_spikes: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = spike_embeddings.shape
        
        # Truncate if sequence is longer than max_length
        if seq_len > self.max_length:
            spike_embeddings = spike_embeddings[:, :self.max_length, :]
            seq_len = self.max_length
        
        # Get sinusoidal position encodings
        pe = self.pe[:seq_len].unsqueeze(0)  # (1, seq_len, embed_dim)
        
        # Get adaptive position encodings
        adaptive_pe = self.adaptive_position[:seq_len].unsqueeze(0)  # (1, seq_len, embed_dim)
        
        # Get position-aware thresholds
        pos_thresholds = self.position_threshold[:seq_len].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Combine encodings
        combined_encoding = spike_embeddings + torch.tanh(pe + adaptive_pe)
        
        # Apply position-aware spiking
        spike_probabilities = torch.sigmoid((combined_encoding - pos_thresholds) * 5.0)
        
        # Generate spikes (training: stochastic, inference: deterministic)
        if self.training:
            temporal_spikes = torch.bernoulli(spike_probabilities)
            # Add surrogate gradient
            surrogate_grad = torch.clamp(1 - torch.abs(combined_encoding - pos_thresholds), 0, 1)
            temporal_spikes = temporal_spikes + surrogate_grad - surrogate_grad.detach()
        else:
            temporal_spikes = (spike_probabilities > 0.5).float()
        
        return temporal_spikes

class VocabularyExpander(nn.Module):
    """
    Dynamically expand vocabulary through neurogenesis
    Creates new token embeddings when encountering unknown tokens
    """
    
    def __init__(self, initial_vocab_size: int, embed_dim: int, max_vocab_size: int = 100000):
        super().__init__()
        self.initial_vocab_size = initial_vocab_size
        self.embed_dim = embed_dim
        self.max_vocab_size = max_vocab_size
        self.current_vocab_size = initial_vocab_size
        
        # Track token usage for vocabulary management
        self.register_buffer('token_usage', torch.zeros(max_vocab_size))
        self.register_buffer('token_creation_time', torch.zeros(max_vocab_size))
        
        # Dynamic embedding expansion buffer
        self.register_buffer('expanded_embeddings', torch.zeros(max_vocab_size - initial_vocab_size, embed_dim))
        
    def expand_vocabulary(self, new_token_id: int) -> torch.Tensor:
        """Create new token embedding through neurogenesis"""
        if new_token_id >= self.max_vocab_size:
            raise ValueError(f"Token ID {new_token_id} exceeds max vocabulary size {self.max_vocab_size}")
        
        if new_token_id >= self.current_vocab_size:
            # Create new embedding based on similar tokens
            expansion_idx = new_token_id - self.initial_vocab_size
            
            # Initialize with noise + average of frequent tokens
            frequent_tokens = torch.topk(self.token_usage[:self.current_vocab_size], k=10).indices
            if len(frequent_tokens) > 0:
                # Average embeddings from base vocabulary (this would need to be passed in)
                new_embedding = torch.randn(self.embed_dim) * 0.02
            else:
                new_embedding = torch.randn(self.embed_dim) * 0.02
            
            self.expanded_embeddings[expansion_idx] = new_embedding
            self.current_vocab_size = max(self.current_vocab_size, new_token_id + 1)
            self.token_creation_time[new_token_id] = torch.tensor(float('inf'))  # Mark as new
        
        # Update usage statistics
        self.token_usage[new_token_id] += 1
        
        return self.expanded_embeddings[new_token_id - self.initial_vocab_size] if new_token_id >= self.initial_vocab_size else None

class SpikingLayerNorm(nn.Module):
    """
    Layer normalization adapted for spike sequences
    Normalizes across the feature dimension while preserving spike patterns
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to spike sequences
        
        Args:
            spike_input: (batch, seq_len, feature_dim)
        Returns:
            normalized_spikes: (batch, seq_len, feature_dim)
        """
        # Compute statistics across feature dimension
        mean = spike_input.mean(-1, keepdim=True)
        std = spike_input.std(-1, keepdim=True)
        
        # Normalize
        normalized = (spike_input - mean) / (std + self.eps)
        
        # Apply learnable parameters
        normalized = normalized * self.weight + self.bias
        
        # Convert back to spike probabilities
        spike_probs = torch.sigmoid(normalized)
        
        # Generate spikes
        if self.training:
            spikes = torch.bernoulli(spike_probs)
            # Surrogate gradient
            surrogate_grad = torch.clamp(1 - torch.abs(normalized), 0, 1)
            spikes = spikes + surrogate_grad - surrogate_grad.detach()
        else:
            spikes = (spike_probs > 0.5).float()
        
        return spikes

class NeuromorphicTokenizer(AURAModule):
    """
    Complete neuromorphic tokenizer with spike-based embeddings and temporal encoding
    Integrates with AURA module system
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        
        # Configuration
        self.vocab_size = config.get('vocab_size', 50000)
        self.embed_dim = config.get('embed_dim', 512)
        self.spike_threshold = config.get('spike_threshold', 0.5)
        self.max_length = config.get('max_length', 2048)
        self.enable_expansion = config.get('enable_vocab_expansion', True)
        
        # Core components
        self.spike_embeddings = BinarySpikeEmbedding(
            module_id='spike_embeddings',
            config={
                'num_embeddings': self.vocab_size,
                'embedding_dim': self.embed_dim,
                'spike_threshold': self.spike_threshold,
                'timesteps': 1,  # Single timestep for tokenizer
                'encoding_scheme': 'rate',
                'enable_temporal_dynamics': False
            }
        )
        
        self.temporal_encoder = TemporalPositionEncoder(
            embed_dim=self.embed_dim,
            max_length=self.max_length
        )
        
        self.layer_norm = SpikingLayerNorm(self.embed_dim)
        
        # Vocabulary expansion (optional)
        if self.enable_expansion:
            self.vocab_expander = VocabularyExpander(
                initial_vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                max_vocab_size=config.get('max_vocab_size', 100000)
            )
        
        # Statistics tracking
        self.register_buffer('total_tokens_processed', torch.tensor(0))
        self.register_buffer('avg_spike_rate', torch.tensor(0.0))
        
    def initialize(self) -> bool:
        """Initialize the tokenizer"""
        try:
            # Initialize spike embeddings
            if hasattr(self.spike_embeddings, 'initialize'):
                self.spike_embeddings.initialize()
            
            self.state = self.ModuleState.ACTIVE if hasattr(self, 'ModuleState') else 'active'
            self.logger.info(f"Neuromorphic tokenizer initialized with vocab_size={self.vocab_size}")
            return True
        except Exception as e:
            self.logger.error(f"Tokenizer initialization failed: {e}")
            return False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through neuromorphic tokenizer
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask (optional)
        
        Returns:
            Dict containing spike_sequences and metadata
        """
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        batch_size, seq_len = input_ids.shape
        
        # Handle vocabulary expansion if enabled
        if self.enable_expansion:
            max_token_id = torch.max(input_ids).item()
            if max_token_id >= self.vocab_size:
                for token_id in range(self.vocab_size, max_token_id + 1):
                    self.vocab_expander.expand_vocabulary(token_id)
        
        # 1. Convert token IDs to spike embeddings
        spike_embeddings_4d = self.spike_embeddings(input_ids)  # (batch, seq, timesteps, dim)
        
        # Convert to 3D by averaging over timesteps (since we set timesteps=1)
        spike_embeddings = spike_embeddings_4d.squeeze(2)  # (batch, seq, dim)
        
        # 2. Add temporal position encoding
        # Ensure temporal encoder is on the same device
        self.temporal_encoder = self.temporal_encoder.to(device)
        temporal_spikes = self.temporal_encoder(spike_embeddings)
        
        # 3. Apply layer normalization
        # Ensure layer norm is on the same device
        self.layer_norm = self.layer_norm.to(device)
        normalized_spikes = self.layer_norm(temporal_spikes)
        
        # 4. Apply attention mask if provided
        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(normalized_spikes)
            normalized_spikes = normalized_spikes * mask
        
        # Update statistics
        self.total_tokens_processed += batch_size * seq_len
        current_spike_rate = torch.mean(normalized_spikes).item()
        self.avg_spike_rate = 0.99 * self.avg_spike_rate + 0.01 * current_spike_rate
        
        return {
            'spike_sequences': normalized_spikes,
            'spike_embeddings': spike_embeddings,
            'temporal_encoding': temporal_spikes,
            'attention_mask': attention_mask,
            'spike_rate': current_spike_rate,
            'sequence_length': seq_len
        }
    
    def process(self, input_data: Any) -> Any:
        """AURAModule interface"""
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data)
        elif isinstance(input_data, dict):
            input_ids = input_data.get('input_ids')
            attention_mask = input_data.get('attention_mask')
            return self.forward(input_ids, attention_mask)
        else:
            raise ValueError("Input must be tensor or dict with 'input_ids'")
    
    def get_tokenizer_statistics(self) -> Dict[str, float]:
        """Get comprehensive tokenizer statistics"""
        stats = {
            'total_tokens_processed': float(self.total_tokens_processed),
            'avg_spike_rate': float(self.avg_spike_rate),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'max_length': self.max_length,
            'spike_threshold': self.spike_threshold
        }
        
        if self.enable_expansion:
            stats.update({
                'current_vocab_size': self.vocab_expander.current_vocab_size,
                'vocab_expansion_rate': (self.vocab_expander.current_vocab_size - self.vocab_size) / self.vocab_size,
                'most_used_tokens': torch.topk(self.vocab_expander.token_usage, k=10).indices.tolist()
            })
        
        return stats
    
    def get_state(self) -> Dict[str, Any]:
        """Get tokenizer state for serialization"""
        state = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'statistics': self.get_tokenizer_statistics(),
            'total_tokens_processed': self.total_tokens_processed.item(),
            'avg_spike_rate': self.avg_spike_rate.item()
        }
        
        if self.enable_expansion:
            state.update({
                'vocab_expander_state': {
                    'current_vocab_size': self.vocab_expander.current_vocab_size,
                    'token_usage': self.vocab_expander.token_usage,
                    'expanded_embeddings': self.vocab_expander.expanded_embeddings
                }
            })
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set tokenizer state from serialization"""
        try:
            self.load_state_dict(state['model_state_dict'])
            self.total_tokens_processed = torch.tensor(state['total_tokens_processed'])
            self.avg_spike_rate = torch.tensor(state['avg_spike_rate'])
            
            if self.enable_expansion and 'vocab_expander_state' in state:
                expander_state = state['vocab_expander_state']
                self.vocab_expander.current_vocab_size = expander_state['current_vocab_size']
                self.vocab_expander.token_usage = expander_state['token_usage']
                self.vocab_expander.expanded_embeddings = expander_state['expanded_embeddings']
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set tokenizer state: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate tokenizer functionality"""
        try:
            # Test with random input
            test_input = torch.randint(0, self.vocab_size, (2, 64))
            output = self.forward(test_input)
            
            # Check output format
            if 'spike_sequences' not in output:
                return False, "Missing spike_sequences in output"
            
            spike_seq = output['spike_sequences']
            expected_shape = (2, 64, self.embed_dim)
            if spike_seq.shape != expected_shape:
                return False, f"Wrong output shape: {spike_seq.shape} vs {expected_shape}"
            
            # Check spike rate
            spike_rate = torch.mean(spike_seq).item()
            if spike_rate < 0.01 or spike_rate > 0.99:
                return False, f"Unrealistic spike rate: {spike_rate}"
            
            return True, "Tokenizer validation successful"
        except Exception as e:
            return False, f"Tokenizer validation error: {str(e)}"

# Utility functions
def create_neuromorphic_tokenizer(config: Dict[str, Any]) -> NeuromorphicTokenizer:
    """Factory function to create neuromorphic tokenizer"""
    return NeuromorphicTokenizer('neuromorphic_tokenizer', config)

def test_tokenizer():
    """Test function for neuromorphic tokenizer"""
    config = {
        'vocab_size': 10000,
        'embed_dim': 256,
        'spike_threshold': 0.5,
        'max_length': 512,
        'enable_vocab_expansion': True
    }
    
    tokenizer = create_neuromorphic_tokenizer(config)
    assert tokenizer.initialize() == True
    
    # Test forward pass
    input_ids = torch.randint(0, config['vocab_size'], (4, 128))
    attention_mask = torch.ones_like(input_ids)
    
    output = tokenizer(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output spike sequences shape: {output['spike_sequences'].shape}")
    print(f"Spike rate: {output['spike_rate']:.3f}")
    print(f"Tokenizer stats: {tokenizer.get_tokenizer_statistics()}")
    
    # Test validation
    is_valid, message = tokenizer.validate()
    print(f"Validation: {is_valid}, {message}")
    
    return tokenizer

if __name__ == "__main__":
    test_tokenizer()
