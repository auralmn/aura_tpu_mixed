import torch
import torch.nn as nn
from typing import Optional
from .expert import Expert

class HierarchicalMoELayer(nn.Module):
    """Hierarchical Mixture of Experts layer with category × specialty structure"""
    
    def __init__(self, d_model: int, d_ff: int, num_categories: int = 8, 
                 num_specialties: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_categories = num_categories
        self.num_specialties = num_specialties
        self.top_k = top_k
        
        # Create hierarchical expert structure: categories × specialties
        self.experts = nn.ModuleList([
            nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_specialties)])
            for _ in range(num_categories)
        ])
    
    def to(self, device):
        """Move the entire hierarchical MoE layer to the specified device"""
        super().to(device)
        return self
    
    def forward(self, hidden: torch.Tensor, cat_indices: torch.Tensor, 
                sub_indices: torch.Tensor, weights: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through hierarchical experts
        """
        output = torch.zeros_like(hidden)
        
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, self.top_k)
        else:
            expanded_mask = None
        
        for cat in range(self.num_categories):
            for sub in range(self.num_specialties):
                expert_mask = ((cat_indices == cat) & (sub_indices == sub))
                
                if expanded_mask is not None:
                    expert_mask = expert_mask & expanded_mask
                
                if isinstance(expert_mask, bool):
                    if not expert_mask:
                        continue
                elif not expert_mask.any():
                    continue
                
                batch_idx, seq_idx, k_idx = expert_mask.nonzero(as_tuple=True)
                
                tokens = hidden[batch_idx, seq_idx]
                expert_weights = weights[batch_idx, seq_idx, k_idx]
                
                expert_output = self.experts[cat][sub](tokens)
                
                output[batch_idx, seq_idx] += expert_output * expert_weights.unsqueeze(-1)
        
        if mask is not None:
            output *= mask.unsqueeze(-1)
        
        return output
