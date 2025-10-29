# SPDX-License-Identifier: Apache-2.0
"""
AURA Binary Spike Embedding Layer
Converts discrete tokens into event-driven binary spike patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

class BinarySpikeEmbedding(nn.Module):
    """
    Binary Spike Embedding Layer with enhanced temporal coding
    - Converts token IDs to binary spike patterns
    - Supports multiple encoding schemes
    - Temporal dynamics with adaptive patterns
    - Hot-swappable module interface
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__()
        
        # Embedding parameters
        self.num_embeddings = config.get('num_embeddings', 32000)
        self.embedding_dim = config.get('embedding_dim', 768)
        self.timesteps = config.get('timesteps', 10)
        
        # Encoding parameters
        self.encoding_scheme = config.get('encoding_scheme', 'rate')  # 'rate', 'temporal', 'population'
        self.spike_threshold = config.get('spike_threshold', 0.0)
        self.noise_level = config.get('noise_level', 0.0)
        
        # Temporal dynamics
        self.enable_temporal_dynamics = config.get('enable_temporal_dynamics', True)
        self.decay_rate = config.get('decay_rate', 0.9)
        
        # Core embedding layer
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Optional pretrained embedding weight (PyTorch tensor file)
        self.pretrained_weight_path = config.get('pretrained_weight_path', None)
        
        # Temporal encoding weights (learnable)
        if self.enable_temporal_dynamics:
            self.temporal_weights = nn.Parameter(torch.randn(self.timesteps, 1))
            self.phase_shifts = nn.Parameter(torch.randn(self.embedding_dim))
        
        # Population encoding (if used)
        if self.encoding_scheme == 'population':
            self.population_centers = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
            self.population_widths = nn.Parameter(torch.ones(self.embedding_dim))
        
        # Adaptive threshold (learnable)
        self.adaptive_threshold = nn.Parameter(torch.tensor(self.spike_threshold))
        
        # Initialize weights
        self._initialize_weights()
        
        # Device management
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def _initialize_weights(self):
        """Initialize embedding weights for spike generation"""
        nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)
        
        if self.enable_temporal_dynamics:
            nn.init.xavier_uniform_(self.temporal_weights, gain=0.5)
            nn.init.uniform_(self.phase_shifts, -np.pi, np.pi)
        
        if self.encoding_scheme == 'population':
            nn.init.xavier_uniform_(self.population_centers, gain=1.0)
            nn.init.uniform_(self.population_widths, 0.5, 2.0)
    
    def initialize(self) -> bool:
        """Initialize the embedding module"""
        try:
            self.to(self.device)
            # Load optional pretrained weight
            if self.pretrained_weight_path:
                try:
                    import torch
                    w = torch.load(self.pretrained_weight_path, map_location=self.device)
                    # Accept dict formats like {'weight': tensor}
                    if isinstance(w, dict) and 'weight' in w:
                        w = w['weight']
                    if isinstance(w, torch.Tensor):
                        if w.dtype != self.embedding.weight.dtype:
                            w = w.to(dtype=self.embedding.weight.dtype)
                        if w.shape != self.embedding.weight.shape:
                            raise ValueError(f"Pretrained embedding shape {tuple(w.shape)} != expected {tuple(self.embedding.weight.shape)}")
                        with torch.no_grad():
                            self.embedding.weight.copy_(w.to(self.device))
                    else:
                        raise ValueError("Unsupported pretrained weight format; expected Tensor or {'weight': Tensor}")
                except Exception as e:
                    logging.getLogger('aura.neural.embedding').warning(f"Failed to load pretrained embedding: {e}")
            return True
        except Exception as e:
            return False
    
    def rate_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Rate-based spike encoding"""
        batch, seq, dim = embeddings.shape
        
        # Normalize embeddings to [0, 1] for spike probability
        normalized = torch.sigmoid(embeddings)
        
        # Expand temporal dimension
        expanded = normalized.unsqueeze(2).expand(-1, -1, self.timesteps, -1)
        
        # Generate spikes based on rate
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(expanded) * self.noise_level
            expanded = expanded + noise
        
        # Apply adaptive threshold
        spikes = (expanded > torch.sigmoid(self.adaptive_threshold)).float()
        
        # Add surrogate gradient for training
        if self.training:
            surrogate_grad = torch.clamp(1 - torch.abs(expanded - self.adaptive_threshold), 0, 1)
            spikes = spikes + surrogate_grad - surrogate_grad.detach()
        
        return spikes
    
    def temporal_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Temporal pattern-based spike encoding"""
        batch, seq, dim = embeddings.shape
        
        # Create temporal patterns
        time_steps = torch.arange(self.timesteps, device=embeddings.device).float()
        
        # Generate sinusoidal patterns based on embedding values
        patterns = []
        for t in range(self.timesteps):
            # Phase modulation based on embedding values and learned phase shifts
            phases = embeddings * self.phase_shifts.unsqueeze(0).unsqueeze(0)
            temporal_pattern = torch.sin(phases + t * self.temporal_weights[t])
            patterns.append(temporal_pattern)
        
        # Stack temporal patterns
        temporal_spikes = torch.stack(patterns, dim=2)  # (batch, seq, time, dim)
        
        # Apply threshold to generate binary spikes
        spikes = (temporal_spikes > torch.sigmoid(self.adaptive_threshold)).float()
        
        return spikes
    
    def population_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Population vector-based spike encoding"""
        batch, seq, dim = embeddings.shape
        
        # Compute distances to population centers
        expanded_emb = embeddings.unsqueeze(3)  # (batch, seq, dim, 1)
        centers = self.population_centers.unsqueeze(0).unsqueeze(0)  # (1, 1, dim, dim)
        
        # Gaussian response functions
        distances = torch.norm(expanded_emb - centers, dim=2)  # (batch, seq, dim)
        responses = torch.exp(-distances ** 2 / (2 * self.population_widths ** 2))
        
        # Generate temporal spike patterns
        temporal_patterns = []
        for t in range(self.timesteps):
            # Time-varying activation
            time_factor = torch.exp(torch.tensor(-t / (self.timesteps * self.decay_rate), device=embeddings.device))
            pattern = responses * time_factor
            temporal_patterns.append(pattern)
        
        temporal_spikes = torch.stack(temporal_patterns, dim=2)
        
        # Apply threshold
        spikes = (temporal_spikes > torch.sigmoid(self.adaptive_threshold)).float()
        
        return spikes
    
    def embed(self, token_ids: Any) -> torch.Tensor:
        """
        Convert token IDs to spike embeddings
        
        Args:
            token_ids: (batch, seq) token indices
            
        Returns:
            spikes: (batch, seq, timesteps, embedding_dim) binary spikes
        """
        # Normalize input to 2D LongTensor on device
        logging.getLogger('aura.neural.embedding').info(f"BSE.embed: input type={type(token_ids)}")
        if isinstance(token_ids, int):
            token_ids = torch.tensor([[token_ids]], dtype=torch.long)
        elif isinstance(token_ids, list):
            token_ids = torch.tensor([token_ids], dtype=torch.long)
        elif isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            token_ids = token_ids.to(dtype=torch.long)
        else:
            raise ValueError("token_ids must be int, list, or torch.Tensor")
        token_ids = token_ids.to(self.device)
        # Also ensure embedding weights are on same device (MPS fix)
        if self.embedding.weight.device != self.device:
            self.embedding.to(self.device)
        
        # Get continuous embeddings
        try:
            embeddings = self.embedding(token_ids)  # (batch, seq, embedding_dim)
            logging.getLogger('aura.neural.embedding').info(f"BSE.embed: embeddings shape={tuple(embeddings.shape)} dtype={embeddings.dtype}")
        except Exception as e:
            logging.getLogger('aura.neural.embedding').error(
                f"BSE.embed: embedding() failed: {e}; token_ids type={type(token_ids)}, "
                f"tensor?={isinstance(token_ids, torch.Tensor)}, dtype={(token_ids.dtype if isinstance(token_ids, torch.Tensor) else 'n/a')}, "
                f"shape={(tuple(token_ids.shape) if isinstance(token_ids, torch.Tensor) else 'n/a')}"
            )
            raise
        
        # Apply chosen encoding scheme
        if self.encoding_scheme == 'rate':
            spikes = self.rate_encoding(embeddings)
        elif self.encoding_scheme == 'temporal':
            spikes = self.temporal_encoding(embeddings)
        elif self.encoding_scheme == 'population':
            spikes = self.population_encoding(embeddings)
        else:
            raise ValueError(f"Unknown encoding scheme: {self.encoding_scheme}")
        logging.getLogger('aura.neural.embedding').info(f"BSE.embed: spikes shape={tuple(spikes.shape)} dtype={spikes.dtype}")
        
        return spikes
    
    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Decode spikes back to continuous embeddings
        
        Args:
            spikes: (batch, seq, timesteps, embedding_dim) binary spikes
            
        Returns:
            embeddings: (batch, seq, embedding_dim) continuous embeddings
        """
        # Simple decoding: sum over time and normalize
        spike_sums = torch.sum(spikes, dim=2)  # (batch, seq, embedding_dim)
        
        # Normalize by timesteps
        decoded = spike_sums / self.timesteps
        
        return decoded
    
    def get_spike_rate(self, spikes: torch.Tensor) -> torch.Tensor:
        """Compute spike rate across time dimension"""
        return torch.mean(spikes, dim=2)  # Average over timesteps
    
    def get_encoding_statistics(self, spikes: torch.Tensor) -> Dict[str, float]:
        """Get statistics about spike encoding"""
        spike_rate = torch.mean(spikes.float())
        spike_sparsity = 1.0 - spike_rate
        
        # Temporal statistics
        temporal_variance = torch.var(torch.mean(spikes, dim=[0, 1, 3]))  # Variance across time
        spatial_variance = torch.var(torch.mean(spikes, dim=[0, 1, 2]))   # Variance across dims
        
        return {
            'spike_rate': float(spike_rate),
            'sparsity': float(spike_sparsity),
            'temporal_variance': float(temporal_variance),
            'spatial_variance': float(spatial_variance),
            'adaptive_threshold': float(self.adaptive_threshold)
        }
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass - alias for embed"""
        return self.embed(token_ids)
    
    # AURAModule interface methods
    def process(self, input_data: Any) -> Any:
        """Process input through embedding layer"""
        if isinstance(input_data, torch.Tensor):
            return self.embed(input_data)
        elif isinstance(input_data, dict):
            token_ids = input_data.get('token_ids')
            if token_ids is not None:
                return self.embed(token_ids)
        raise ValueError("BinarySpikeEmbedding requires token_ids tensor input")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current embedding state for hot-swapping"""
        state = {
            'model_state_dict': self.state_dict(),
            'config': {
                'num_embeddings': self.num_embeddings,
                'embedding_dim': self.embedding_dim,
                'timesteps': self.timesteps,
                'encoding_scheme': self.encoding_scheme,
                'spike_threshold': self.spike_threshold,
                'enable_temporal_dynamics': self.enable_temporal_dynamics,
                'decay_rate': self.decay_rate
            }
        }
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set embedding state during hot-swapping"""
        try:
            # Restore model parameters
            self.load_state_dict(state['model_state_dict'])
            
            # Restore configuration
            config = state['config']
            self.num_embeddings = config['num_embeddings']
            self.embedding_dim = config['embedding_dim']
            self.timesteps = config['timesteps']
            self.encoding_scheme = config['encoding_scheme']
            self.spike_threshold = config['spike_threshold']
            self.enable_temporal_dynamics = config['enable_temporal_dynamics']
            self.decay_rate = config['decay_rate']
            
            return True
        except Exception as e:
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate embedding functionality"""
        try:
            test_tokens = torch.randint(0, min(100, self.num_embeddings), (2, 8), device=self.device)
            test_spikes = self.embed(test_tokens)
            
            expected_shape = (2, 8, self.timesteps, self.embedding_dim)
            if test_spikes.shape != expected_shape:
                return False, f"Output shape mismatch: {test_spikes.shape} vs {expected_shape}"
            
            if torch.any(torch.isnan(test_spikes)) or torch.any(torch.isinf(test_spikes)):
                return False, "Output contains NaN or Inf values"
            
            # Check binary nature of spikes (allow for surrogate gradients)
            unique_vals = torch.unique(test_spikes)
            if len(unique_vals) > 3:  # Allow for 0, 1, and intermediate values from surrogate gradients
                return False, f"Spikes not binary enough: unique values {unique_vals}"
            
            return True, "BinarySpikeEmbedding validation successful"
        except Exception as e:
            return False, f"BinarySpikeEmbedding validation error: {str(e)}"