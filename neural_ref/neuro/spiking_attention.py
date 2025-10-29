# SPDX-License-Identifier: Apache-2.0
"""
AURA Spiking Multi-Head Attention
Neuromorphic attention mechanism for spike-based transformer processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from aura.neural.interfaces import AURAModule

# Try to import Norse for advanced neuromorphic features
try:
    import norse.torch as snn
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    print("Norse not available, using custom neuromorphic implementations")

class SpikingLinear(nn.Module):
    """
    Linear layer that processes spike inputs and produces spike outputs
    Uses learnable thresholds and surrogate gradients
    Implements PyTorch-to-neuromorphic translation patterns
    """
    
    def __init__(self, in_features: int, out_features: int, spike_threshold: float = 0.5, 
                 use_norse: bool = False, dt: float = 1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spike_threshold = spike_threshold
        self.use_norse = use_norse and NORSE_AVAILABLE
        self.dt = dt
        
        # Standard linear layer (synaptic weights)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # Learnable spike threshold per output neuron
        self.adaptive_threshold = nn.Parameter(torch.full((out_features,), spike_threshold))
        
        # Initialize weights for spike processing
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        
        # Norse LIF neuron if available
        if self.use_norse:
            self.neuron = snn.LIFCell(snn.LIFParameters(
                tau_mem_inv=torch.tensor(1.0),
                tau_syn_inv=torch.tensor(1.0),
                v_leak=torch.tensor(0.0),
                v_th=torch.tensor(1.0),
                v_reset=torch.tensor(0.0),
            ))
            self.register_buffer('neuron_state', None)
        else:
            self.neuron = None
            self.neuron_state = None
    
    def forward(self, spike_input: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process spike input through neuromorphic transformation
        Implements PyTorch-to-neuromorphic translation patterns
        
        Args:
            spike_input: (batch, seq_len, in_features) or (batch, in_features)
            state: Optional neuron state for temporal processing
        Returns:
            spike_output: Same shape as input but with out_features
        """
        if self.use_norse and self.neuron is not None:
            # Use Norse LIF neuron for true neuromorphic processing
            return self._norse_forward(spike_input, state)
        else:
            # Use custom surrogate gradient approach
            return self._custom_forward(spike_input)
    
    def _norse_forward(self, spike_input: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using Norse LIF neurons"""
        batch_size = spike_input.size(0)
        
        # Initialize neuron state if needed
        if state is None or self.neuron_state is None:
            # Norse API compatibility across versions
            try:
                st = self.neuron.initial_state(batch_size)
            except TypeError:
                try:
                    st = self.neuron.initial_state(batch_size, self.out_features)
                except TypeError:
                    st = self.neuron.initial_state(batch_size)
            # Move state tensors to input device
            try:
                self.neuron_state = type(st)(*(t.to(spike_input.device) for t in st))
            except Exception:
                self.neuron_state = st
        
        # Linear transformation (synaptic current injection)
        current = self.linear(spike_input)
        
        # Process through LIF neuron
        spikes, self.neuron_state = self.neuron(current, self.neuron_state)
        
        return spikes
    
    def _custom_forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """Custom forward pass with surrogate gradients"""
        # Linear transformation
        linear_output = self.linear(spike_input)
        
        # Apply adaptive threshold
        threshold = self.adaptive_threshold.unsqueeze(0).unsqueeze(0) if spike_input.dim() == 3 else self.adaptive_threshold.unsqueeze(0)
        
        # Generate spike probabilities
        spike_probs = torch.sigmoid((linear_output - threshold) * 5.0)
        
        # Generate spikes (stochastic in training, deterministic in eval)
        if self.training:
            spikes = torch.bernoulli(spike_probs)
            # Surrogate gradient for backpropagation (scaled to avoid exceeding 1)
            surrogate_grad = torch.clamp(1 - torch.abs(linear_output - threshold), 0, 1) * 0.1
            spikes = spikes + surrogate_grad - surrogate_grad.detach()
            # Ensure spikes stay in [0,1] range
            spikes = torch.clamp(spikes, 0, 1)
        else:
            spikes = (spike_probs > 0.5).float()
        
        return spikes

class TemporalSpikingAttention(nn.Module):
    """
    Temporal attention mechanism for spike sequences
    Computes attention weights based on spike timing and patterns
    """
    
    def __init__(self, embed_dim: int, spike_threshold: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.spike_threshold = spike_threshold
        self.temperature = temperature
        self.scale = 1.0 / math.sqrt(embed_dim)
        
        # Temporal kernel for spike timing dependencies
        self.register_buffer('temporal_kernel', self._build_temporal_kernel())
        
        # Learnable parameters for spike attention
        self.spike_attention_weight = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.temporal_decay = nn.Parameter(torch.tensor(0.95))  # Decay factor for temporal attention
        
    def _build_temporal_kernel(self, kernel_size: int = 21) -> torch.Tensor:
        """Build temporal kernel for spike timing relationships"""
        center = kernel_size // 2
        positions = torch.arange(kernel_size).float() - center
        # Gaussian-like kernel but optimized for spikes
        kernel = torch.exp(-positions**2 / (2 * (center/3)**2))
        return kernel / kernel.sum()
    
    def compute_spike_attention(self, queries: torch.Tensor, keys: torch.Tensor, 
                               values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention weights for spike sequences
        
        Args:
            queries: (batch, num_heads, seq_len, head_dim)
            keys: (batch, num_heads, seq_len, head_dim)  
            values: (batch, num_heads, seq_len, head_dim)
            attention_mask: (batch, seq_len) optional mask
            
        Returns:
            attended_spikes: (batch, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = queries.shape
        
        # Standard scaled dot-product attention computation
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Add temporal bias based on spike timing
        if seq_len <= self.temporal_kernel.size(0):
            temporal_bias = self.temporal_kernel[:seq_len].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            temporal_bias = temporal_bias.expand(batch_size, num_heads, seq_len, seq_len)
            attention_scores = attention_scores + temporal_bias
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            mask = mask.expand(batch_size, num_heads, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Compute attention weights with temperature scaling
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)
        
        # Apply temporal decay (recent tokens get higher weight)
        if seq_len > 1:
            decay_weights = torch.pow(self.temporal_decay, torch.arange(seq_len, 0, -1).float().to(queries.device))
            decay_weights = decay_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            attention_weights = attention_weights * decay_weights
            # Renormalize
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Convert back to spike patterns
        spike_probs = torch.sigmoid(attended_values / self.temperature)
        
        if self.training:
            attended_spikes = torch.bernoulli(spike_probs)
            # Surrogate gradient (scaled to avoid exceeding 1)
            surrogate_grad = torch.clamp(1 - torch.abs(attended_values), 0, 1) * 0.1
            attended_spikes = attended_spikes + surrogate_grad - surrogate_grad.detach()
            # Ensure spikes are in [0,1] range
            attended_spikes = torch.clamp(attended_spikes, 0, 1)
        else:
            attended_spikes = (spike_probs > 0.5).float()
        
        return attended_spikes, attention_weights

class SpikingMultiHeadAttention(AURAModule):
    """
    Multi-head attention mechanism for neuromorphic spike processing
    Processes spike sequences with multiple attention heads
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        
        # Configuration
        self.hidden_size = config.get('hidden_size', 512)
        self.num_attention_heads = config.get('num_attention_heads', 8)
        self.spike_threshold = config.get('spike_threshold', 0.5)
        self.attention_temperature = config.get('attention_temperature', 1.0)
        self.dropout_rate = config.get('attention_dropout', 0.1)
        
        # Validate configuration
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_attention_heads}")
        
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Spiking projection layers for Q, K, V
        self.query_proj = SpikingLinear(self.hidden_size, self.hidden_size, self.spike_threshold)
        self.key_proj = SpikingLinear(self.hidden_size, self.hidden_size, self.spike_threshold)  
        self.value_proj = SpikingLinear(self.hidden_size, self.hidden_size, self.spike_threshold)
        self.output_proj = SpikingLinear(self.hidden_size, self.hidden_size, self.spike_threshold)
        
        # Temporal spiking attention mechanism
        self.temporal_attention = TemporalSpikingAttention(
            embed_dim=self.head_dim,
            spike_threshold=self.spike_threshold,
            temperature=self.attention_temperature
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Statistics tracking
        self.register_buffer('total_attention_steps', torch.tensor(0))
        self.register_buffer('avg_attention_entropy', torch.tensor(0.0))
        self.register_buffer('avg_spike_rate', torch.tensor(0.0))
        
    def initialize(self) -> bool:
        """Initialize spiking attention module"""
        try:
            # Initialize projection layers
            for proj in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
                if hasattr(proj, 'linear'):
                    nn.init.xavier_uniform_(proj.linear.weight, gain=1/math.sqrt(2))
                    
            self.state = 'active'
            self.logger.info(f"SpikingMultiHeadAttention initialized: {self.num_attention_heads} heads, {self.head_dim} dim each")
            return True
        except Exception as e:
            self.logger.error(f"SpikingAttention initialization failed: {e}")
            return False
    
    def forward(self, spike_input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through spiking multi-head attention
        
        Args:
            spike_input: (batch, seq_len, hidden_size) spike sequences
            attention_mask: (batch, seq_len) attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dict containing attention output and optional weights
        """
        batch_size, seq_len, hidden_size = spike_input.shape
        
        # Generate Q, K, V spike patterns
        queries = self.query_proj(spike_input)  # (batch, seq_len, hidden_size)
        keys = self.key_proj(spike_input)
        values = self.value_proj(spike_input)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Apply temporal spiking attention
        attended_spikes, attention_weights = self.temporal_attention.compute_spike_attention(
            queries, keys, values, attention_mask
        )
        
        # Reshape back to concatenated heads
        attended_spikes = attended_spikes.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Final output projection
        output_spikes = self.output_proj(attended_spikes)
        
        # Apply dropout (but preserve binary nature of spikes)
        if self.training and self.dropout_rate > 0:
            # For spikes, we need to apply dropout differently to maintain binary values
            dropout_mask = torch.bernoulli(torch.ones_like(output_spikes) * (1 - self.dropout_rate))
            output_spikes = output_spikes * dropout_mask
        else:
            output_spikes = self.dropout(output_spikes)
        
        # Apply attention mask to ensure masked positions are zero
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(output_spikes)
            output_spikes = output_spikes * mask
        
        # Update statistics
        self.total_attention_steps += 1
        current_spike_rate = torch.mean(output_spikes).item()
        self.avg_spike_rate = 0.99 * self.avg_spike_rate + 0.01 * current_spike_rate
        
        # Compute attention entropy for monitoring
        if attention_weights.numel() > 0:
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1).mean().item()
            self.avg_attention_entropy = 0.99 * self.avg_attention_entropy + 0.01 * entropy
        
        result = {
            'attention_output': output_spikes,
            'spike_rate': current_spike_rate,
            'attention_entropy': entropy if 'entropy' in locals() else 0.0
        }
        
        if return_attention_weights:
            result['attention_weights'] = attention_weights
            
        return result
    
    def process(self, input_data: Any) -> Any:
        """AURAModule interface"""
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data)
        elif isinstance(input_data, dict):
            spike_input = input_data.get('spike_sequences', input_data.get('input'))
            attention_mask = input_data.get('attention_mask')
            return_weights = input_data.get('return_attention_weights', False)
            return self.forward(spike_input, attention_mask, return_weights)
        else:
            raise ValueError("Input must be tensor or dict with spike sequences")
    
    def get_attention_statistics(self) -> Dict[str, float]:
        """Get comprehensive attention statistics"""
        return {
            'total_attention_steps': float(self.total_attention_steps),
            'avg_attention_entropy': float(self.avg_attention_entropy),
            'avg_spike_rate': float(self.avg_spike_rate),
            'num_attention_heads': self.num_attention_heads,
            'head_dim': self.head_dim,
            'hidden_size': self.hidden_size,
            'spike_threshold': self.spike_threshold,
            'attention_temperature': self.attention_temperature
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get attention state for serialization"""
        return {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'statistics': self.get_attention_statistics()
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set attention state from serialization"""
        try:
            self.load_state_dict(state['model_state_dict'])
            return True
        except Exception as e:
            self.logger.error(f"Failed to set attention state: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate attention functionality"""
        try:
            # Test with random spike input
            test_input = torch.randint(0, 2, (2, 64, self.hidden_size)).float()
            test_mask = torch.ones(2, 64)
            
            result = self.forward(test_input, test_mask)
            
            if 'attention_output' not in result:
                return False, "Missing attention_output in result"
                
            output = result['attention_output']
            expected_shape = (2, 64, self.hidden_size)
            if output.shape != expected_shape:
                return False, f"Wrong output shape: {output.shape} vs {expected_shape}"
            
            # Check spike rate reasonableness
            spike_rate = result['spike_rate']
            if spike_rate < 0.01 or spike_rate > 0.99:
                return False, f"Unrealistic spike rate: {spike_rate}"
                
            return True, "Spiking attention validation successful"
        except Exception as e:
            return False, f"Spiking attention validation error: {str(e)}"

# Utility functions
def create_spiking_attention(config: Dict[str, Any]) -> SpikingMultiHeadAttention:
    """Factory function to create spiking attention"""
    return SpikingMultiHeadAttention('spiking_attention', config)

def test_spiking_attention():
    """Test function for spiking attention"""
    config = {
        'hidden_size': 256,
        'num_attention_heads': 8,
        'spike_threshold': 0.5,
        'attention_temperature': 1.0,
        'attention_dropout': 0.1
    }
    
    attention = create_spiking_attention(config)
    assert attention.initialize() == True
    
    # Test forward pass
    batch_size, seq_len = 4, 32
    spike_input = torch.randint(0, 2, (batch_size, seq_len, config['hidden_size'])).float()
    attention_mask = torch.ones(batch_size, seq_len)
    
    result = attention(spike_input, attention_mask, return_attention_weights=True)
    
    print(f"Input shape: {spike_input.shape}")
    print(f"Output shape: {result['attention_output'].shape}")
    print(f"Spike rate: {result['spike_rate']:.3f}")
    print(f"Attention entropy: {result['attention_entropy']:.3f}")
    print(f"Attention weights shape: {result['attention_weights'].shape}")
    print(f"Statistics: {attention.get_attention_statistics()}")
    
    # Test validation
    is_valid, message = attention.validate()
    print(f"Validation: {is_valid}, {message}")
    
    return attention

if __name__ == "__main__":
    test_spiking_attention()
