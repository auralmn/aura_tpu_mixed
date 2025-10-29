"""
Attention Mechanism Interfaces for AURA

Abstract base classes for attention mechanisms in the AURA neuromorphic system.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, Any
from aura.core.base import AURAModule


class AttentionMechanism(AURAModule):
    """
    Abstract base class for attention mechanisms in AURA.
    
    All attention mechanisms must implement spike processing capabilities
    and maintain compatibility with the hot-swappable module system.
    """
    
    @abstractmethod
    def process_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Process spike tensor through attention mechanism.
        
        Args:
            spikes: Input spike tensor of shape (batch, seq_len, embed_dim)
            
        Returns:
            Output tensor with attention applied
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> torch.Tensor:
        """
        Get current attention weight matrix.
        
        Returns:
            Attention weights tensor
        """
        pass
    
    @abstractmethod
    def get_spike_statistics(self) -> Dict[str, float]:
        """
        Get statistics about processed spikes.
        
        Returns:
            Dictionary containing spike processing statistics
        """
        pass
