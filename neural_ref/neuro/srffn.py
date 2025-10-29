# SPDX-License-Identifier: Apache-2.0
"""
AURA Spiking Receptance Feed-Forward Network (SRFFN)
O(T·d) complexity spike-based feed-forward processing with enhanced content analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re
from aura.neural.interfaces import FeedForwardNetwork

class SpikingSRFFN(FeedForwardNetwork, nn.Module):
    """
    Spiking Receptance Feed-Forward Network with enhanced content processing
    - O(T·d) complexity for efficient processing
    - Temporal state mixing with decay
    - Enhanced content feature extraction
    - Hot-swappable module interface
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        FeedForwardNetwork.__init__(self, module_id, config)
        nn.Module.__init__(self)
        
        # Core dimensions
        self.d_model = config.get('d_model', 256)
        self.d_ff = config.get('d_ff', 1024)
        self.dropout_rate = config.get('dropout', 0.1)
        
        # Spiking parameters
        self.spike_threshold = config.get('spike_threshold', 0.5)
        self.decay_factor = config.get('decay_factor', 0.9)
        self.reset_mode = config.get('reset_mode', 'soft')
        
        # Enhanced content analysis
        self.enable_content_analysis = config.get('enable_content_analysis', True)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        
        # Network layers
        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Receptance gating (key SRFFN component)
        self.receptance = nn.Linear(self.d_model, self.d_ff, bias=False)
        
        # Temporal state buffers
        self.register_buffer('prev_activation', torch.zeros(1, self.d_ff))
        self.register_buffer('prev_receptance', torch.zeros(1, self.d_ff))
        self.register_buffer('adaptation_state', torch.zeros(1))
        
        # Content feature extraction weights (learnable)
        if self.enable_content_analysis:
            self.content_weights = nn.Parameter(torch.ones(5))  # 5 content features
        
        # Initialize weights
        self._initialize_weights()
        
        # Device management
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def _initialize_weights(self):
        """Initialize network weights for spiking dynamics"""
        nn.init.xavier_uniform_(self.w1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.receptance.weight, gain=0.3)  # Lower gain for gating
    
    def initialize(self) -> bool:
        """Initialize the SRFFN module"""
        try:
            self.to(self.device)
            return True
        except Exception as e:
            return False
    
    def spike_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced spike activation with adaptive threshold"""
        # Adaptive threshold based on adaptation state
        adaptive_threshold = self.spike_threshold * (1.0 + 0.1 * self.adaptation_state)
        
        spikes = (x > adaptive_threshold).float()
        
        if self.training:
            # Surrogate gradient for backpropagation
            spike_grad = torch.clamp(1 - torch.abs(x - adaptive_threshold), 0, 1)
            spikes = spikes + spike_grad - spike_grad.detach()
        
        return spikes
    
    def receptance_gating(self, x: torch.Tensor, prev_receptance: torch.Tensor) -> torch.Tensor:
        """Compute receptance gating for temporal mixing"""
        # Receptance computation
        r_t = torch.sigmoid(self.receptance(x))
        
        # Temporal mixing with previous receptance
        mixed_receptance = self.decay_factor * prev_receptance + (1 - self.decay_factor) * r_t
        
        return r_t, mixed_receptance
    
    def extract_enhanced_content_features(self, input_tensor: torch.Tensor, 
                                        text: Optional[str] = None) -> Dict[str, float]:
        """Enhanced content feature extraction for better categorization"""
        features = {}
        
        if input_tensor is not None:
            # Tensor-based features
            features['activation_sparsity'] = float(torch.mean((input_tensor > 0).float()))
            features['activation_variance'] = float(torch.var(input_tensor))
            features['spectral_norm'] = float(torch.norm(input_tensor, p='fro'))
            
            # Frequency domain analysis (CPU fallback for MPS)
            if input_tensor.dim() >= 2:
                try:
                    fft_tensor = torch.fft.fft(input_tensor.flatten()).abs()
                    features['spectral_centroid'] = float(torch.mean(fft_tensor))
                except NotImplementedError:
                    # MPS fallback - use simple variance instead
                    features['spectral_centroid'] = float(torch.var(input_tensor))
        
        if text is not None and self.enable_content_analysis:
            # Enhanced linguistic features
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            
            # Lexical complexity
            features['lexical_diversity'] = len(set(words)) / max(len(words), 1)
            features['avg_sentence_length'] = len(words) / max(len(sentences), 1)
            
            # Syntactic complexity
            features['subordinate_clauses'] = len(re.findall(r'\b(that|which|who|when|where|why)\b', text.lower()))
            features['passive_voice'] = len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', text.lower()))
            
            # Semantic indicators
            technical_terms = len(re.findall(r'\b\w{8,}\b', text))  # Long words often technical
            features['technical_density'] = technical_terms / max(len(words), 1)
            
            # Domain-specific patterns
            features['numerical_density'] = len(re.findall(r'\d+', text)) / max(len(text), 1)
            features['citation_patterns'] = len(re.findall(r'\([12][0-9]{3}\)', text))  # Year citations
            
            # Narrative structure
            narrative_indicators = ['story', 'character', 'plot', 'scene', 'chapter']
            features['narrative_score'] = sum(text.lower().count(word) for word in narrative_indicators)
            
            # Academic writing indicators
            academic_indicators = ['therefore', 'however', 'furthermore', 'methodology', 'analysis']
            features['academic_score'] = sum(text.lower().count(word) for word in academic_indicators)
        
        return features
    
    def forward(self, spikes: torch.Tensor, text: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass through SRFFN
        
        Args:
            spikes: (batch, seq, d_model) input spike tensor
            text: Optional text for enhanced content analysis
            
        Returns:
            output: (batch, seq, d_model) processed output
        """
        batch_size, seq_len, d_model = spikes.shape
        
        # Move to device
        spikes = spikes.to(self.device)
        
        # Resize state buffers if needed
        if self.prev_activation.size(0) != batch_size:
            self.prev_activation = torch.zeros(batch_size, self.d_ff, device=self.device)
            self.prev_receptance = torch.zeros(batch_size, self.d_ff, device=self.device)
        
        outputs = []
        current_activation = self.prev_activation.clone()
        current_receptance = self.prev_receptance.clone()
        
        # Enhanced content analysis
        if self.enable_content_analysis and text is not None:
            content_features = self.extract_enhanced_content_features(spikes, text)
            
            # Adaptive processing based on content
            if 'technical_density' in content_features:
                tech_factor = content_features['technical_density']
                adaptive_gain = 1.0 + 0.2 * tech_factor  # Boost for technical content
            else:
                adaptive_gain = 1.0
        else:
            adaptive_gain = 1.0
            
        # Process sequence step by step for O(T·d) complexity
        for t in range(seq_len):
            x_t = spikes[:, t, :]  # (batch, d_model)
            
            # Receptance gating
            r_t, current_receptance = self.receptance_gating(x_t, current_receptance)
            
            # Feed-forward processing
            hidden = self.w1(x_t)  # (batch, d_ff)
            
            # Temporal mixing with previous activation
            mixed_hidden = (self.decay_factor * current_activation + 
                           (1 - self.decay_factor) * hidden)
            
            # Apply content-aware adaptive gain
            mixed_hidden = mixed_hidden * adaptive_gain
            
            # Spike activation
            spike_hidden = self.spike_activation(mixed_hidden)
            
            # Apply receptance gating
            gated_hidden = spike_hidden * r_t
            
            # Dropout for regularization
            if self.training:
                gated_hidden = self.dropout(gated_hidden)
            
            # Output projection
            output_t = self.w2(gated_hidden)  # (batch, d_model)
            
            outputs.append(output_t)
            
            # Update states
            current_activation = mixed_hidden.detach()
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq, d_model)
        
        # Update stored states
        self.prev_activation = current_activation
        self.prev_receptance = current_receptance
        
        # Update adaptation state based on processing
        if self.enable_content_analysis:
            processing_intensity = torch.mean(torch.abs(output))
            self.adaptation_state = (0.9 * self.adaptation_state + 
                                   0.1 * processing_intensity.detach())
        
        return output
    
    def process_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """Interface method for spike processing"""
        return self.forward(spikes)
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network structure information"""
        return {
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'spike_threshold': self.spike_threshold,
            'decay_factor': self.decay_factor,
            'reset_mode': self.reset_mode,
            'enable_content_analysis': self.enable_content_analysis,
            'adaptation_state': float(self.adaptation_state),
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def get_processing_statistics(self) -> Dict[str, float]:
        """Get processing statistics"""
        return {
            'prev_activation_norm': float(torch.norm(self.prev_activation)),
            'prev_receptance_norm': float(torch.norm(self.prev_receptance)),
            'adaptation_state': float(self.adaptation_state),
            'spike_threshold': self.spike_threshold,
            'decay_factor': self.decay_factor
        }
    
    # AURAModule interface methods
    def process(self, input_data: Any) -> Any:
        """Process input through SRFFN"""
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data)
        elif isinstance(input_data, dict):
            spikes = input_data.get('spikes')
            text = input_data.get('text')
            if spikes is not None:
                return self.forward(spikes, text)
        raise ValueError("SRFFN requires spike tensor input")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current SRFFN state for hot-swapping"""
        return {
            'prev_activation': self.prev_activation.detach().cpu().numpy(),
            'prev_receptance': self.prev_receptance.detach().cpu().numpy(),
            'adaptation_state': self.adaptation_state.detach().cpu().numpy(),
            'model_state_dict': self.state_dict(),
            'config': {
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'spike_threshold': self.spike_threshold,
                'decay_factor': self.decay_factor,
                'dropout_rate': self.dropout_rate,
                'enable_content_analysis': self.enable_content_analysis
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set SRFFN state during hot-swapping"""
        try:
            # Restore buffer states
            self.prev_activation = torch.from_numpy(state['prev_activation']).to(self.device)
            self.prev_receptance = torch.from_numpy(state['prev_receptance']).to(self.device)
            self.adaptation_state = torch.from_numpy(state['adaptation_state']).to(self.device)
            
            # Restore model parameters
            self.load_state_dict(state['model_state_dict'])
            
            # Restore configuration
            config = state['config']
            self.d_model = config['d_model']
            self.d_ff = config['d_ff']
            self.spike_threshold = config['spike_threshold']
            self.decay_factor = config['decay_factor']
            self.dropout_rate = config['dropout_rate']
            self.enable_content_analysis = config['enable_content_analysis']
            
            return True
        except Exception as e:
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate SRFFN functionality"""
        try:
            test_input = torch.randn(2, 10, self.d_model, device=self.device)
            test_output = self.forward(test_input)
            
            if test_output.shape != test_input.shape:
                return False, f"Output shape mismatch: {test_output.shape} vs {test_input.shape}"
            
            if torch.any(torch.isnan(test_output)) or torch.any(torch.isinf(test_output)):
                return False, "Output contains NaN or Inf values"
            
            return True, "SRFFN validation successful"
        except Exception as e:
            return False, f"SRFFN validation error: {str(e)}"