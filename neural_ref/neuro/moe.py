# SPDX-License-Identifier: Apache-2.0
"""
AURA Modular MoE Layer with Integrated Neuromorphic SRWKV Routing
- Single-file specification for maintainability
- Uses neuromorphic, liquid, bandit, and spiking enhancements via SpikingSRWKV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from aura.neural.attention.srwkv import SpikingSRWKV
from .expert_evaluator import ExpertEvaluator, create_expert_evaluator
from .expert import Expert, SpikingExpert
from .hierarchical_moe import HierarchicalMoELayer
from aura.neural.neurogenesis import NeurogenesisMoELayer, NeurogenesisConfig

# Try to import Norse for advanced neuromorphic features
try:
    import norse.torch as snn
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    print("Norse not available, using custom neuromorphic implementations")

class SRWKVRouter(nn.Module):
    """SRWKV-based router for MoE expert selection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.srwkv = SpikingSRWKV("srwkv_router", config)
        self.d_model = config['embedding_dim']
        self.num_experts = config['num_neurons_per_head']
        self.top_k = config['neuron_top_k']
        
        # Hierarchical routing parameters
        self.num_categories = config.get('num_categories', 8)
        self.num_specialties = config.get('num_specialties', 8)
        self.hierarchical_mode = config.get('hierarchical_mode', False)
        
        # Initialize SRWKV first to get device
        self.srwkv.initialize()
        device = self.srwkv.device
        
        if self.hierarchical_mode:
            # Hierarchical routing: separate projections for categories and specialties
            self.category_projection = nn.Linear(self.d_model, self.num_categories, bias=False)
            self.specialty_projection = nn.Linear(self.d_model, self.num_specialties, bias=False)
            nn.init.normal_(self.category_projection.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.specialty_projection.weight, mean=0.0, std=0.02)
            # Move to correct device
            self.category_projection = self.category_projection.to(device)
            self.specialty_projection = self.specialty_projection.to(device)
        else:
            # Legacy flat routing
            self.router_projection = nn.Linear(self.d_model, self.num_experts, bias=False)
            nn.init.normal_(self.router_projection.weight, mean=0.0, std=0.02)
            # Move to correct device
            self.router_projection = self.router_projection.to(device)
        
    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None, 
                text: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Route hidden states to top-k experts using SRWKV attention"""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Process through SRWKV for enhanced representations
        srwkv_output = self.srwkv.process_spikes(hidden_states, input_ids, text)
        
        if self.hierarchical_mode:
            # Hierarchical routing: separate category and specialty selection
            cat_logits = self.category_projection(srwkv_output)
            sub_logits = self.specialty_projection(srwkv_output)
            
            # Apply temperature scaling
            cat_logits = cat_logits / self.srwkv.temperature
            sub_logits = sub_logits / self.srwkv.temperature
            
            # Softmax for probabilities
            cat_probs = F.softmax(cat_logits, dim=-1)
            sub_probs = F.softmax(sub_logits, dim=-1)
            
            # Select top-k categories and specialties
            top_k_cat_probs, top_k_cat_indices = torch.topk(cat_probs, self.top_k, dim=-1)
            top_k_sub_probs, top_k_sub_indices = torch.topk(sub_probs, self.top_k, dim=-1)
            
            # Normalize weights
            top_k_cat_weights = top_k_cat_probs / (top_k_cat_probs.sum(dim=-1, keepdim=True) + 1e-8)
            top_k_sub_weights = top_k_sub_probs / (top_k_sub_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Combine category and specialty weights
            combined_weights = top_k_cat_weights * top_k_sub_weights
            combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Load balancing loss (separate for categories and specialties)
            cat_load_balance = F.kl_div(
                cat_probs.mean(dim=(0, 1)).log(),
                (torch.ones(self.num_categories, device=self.srwkv.device) / self.num_categories),
                reduction='batchmean'
            )
            sub_load_balance = F.kl_div(
                sub_probs.mean(dim=(0, 1)).log(),
                (torch.ones(self.num_specialties, device=self.srwkv.device) / self.num_specialties),
                reduction='batchmean'
            )
            load_balance_loss = cat_load_balance + sub_load_balance
            
            # Get SRWKV statistics
            srwkv_stats = self.srwkv.get_spike_statistics()
            
            router_info = {
                'load_balance_loss': load_balance_loss,
                'cat_probs': cat_probs,
                'sub_probs': sub_probs,
                'utilization_entropy': srwkv_stats.get('head_0_utilization_entropy', 0.0),
                'energy_metrics': {
                    'total_macs': srwkv_stats.get('total_macs', 0),
                    'total_energy_j': srwkv_stats.get('total_energy_j', 0),
                    'ops_per_joule': srwkv_stats.get('ops_per_joule', 0)
                },
                'bandit_rewards_ema': srwkv_stats.get('bandit_rewards_ema', 0.0),
                'temperature': srwkv_stats.get('temperature', 1.0)
            }
            
            return top_k_cat_indices, top_k_sub_indices, combined_weights, router_info
            
        else:
            # Legacy flat routing
            router_logits = self.router_projection(srwkv_output)
            router_logits = router_logits / self.srwkv.temperature
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Select top-k experts
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Load balancing loss
            mean_probs = router_probs.mean(dim=(0, 1))
            uniform_dist = torch.ones_like(mean_probs) / self.num_experts
            load_balance_loss = F.kl_div(mean_probs.log(), uniform_dist, reduction='batchmean')
            
            # Get SRWKV statistics
            srwkv_stats = self.srwkv.get_spike_statistics()
            
            router_info = {
                'load_balance_loss': load_balance_loss,
                'router_probs': router_probs,
                'utilization_entropy': srwkv_stats.get('head_0_utilization_entropy', 0.0),
                'expert_counts': torch.bincount(top_k_indices.flatten(), minlength=self.num_experts),
                'energy_metrics': {
                    'total_macs': srwkv_stats.get('total_macs', 0),
                    'total_energy_j': srwkv_stats.get('total_energy_j', 0),
                    'ops_per_joule': srwkv_stats.get('ops_per_joule', 0)
                },
                'bandit_rewards_ema': srwkv_stats.get('bandit_rewards_ema', 0.0),
                'temperature': srwkv_stats.get('temperature', 1.0)
            }
            
            return top_k_indices, top_k_weights, router_info
    
    def update_bandit_stats(self, expert_indices: torch.Tensor, rewards: torch.Tensor):
        """Update bandit statistics for expert selection"""
        # Use the first router for bandit updates
        if hasattr(self.srwkv, 'neuron_routers') and len(self.srwkv.neuron_routers) > 0:
            self.srwkv.neuron_routers[0].update_bandit_stats(expert_indices, rewards)

class MoELayer(nn.Module):
    """Mixture of Experts layer with dynamic expert routing"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2, dropout: float = 0.1,
                 use_spiking: bool = False, use_norse: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_spiking = use_spiking
        
        # Create experts (spiking or standard)
        if use_spiking:
            self.experts = nn.ModuleList([
                SpikingExpert(d_model, d_ff, dropout, use_norse=use_norse) 
                for _ in range(num_experts)
            ])
        else:
            self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])
    
    def forward(self, hidden: torch.Tensor, top_k_indices: torch.Tensor, 
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Forward pass through selected experts"""
        output = torch.zeros_like(hidden)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (top_k_indices == expert_idx)
            if not mask.any():
                continue
            
            # Get batch, sequence, and k indices where this expert is selected
            batch_idx, seq_idx, k_idx = mask.nonzero(as_tuple=True)
            
            # Extract tokens and weights for this expert
            tokens = hidden[batch_idx, seq_idx]
            weights = top_k_weights[batch_idx, seq_idx, k_idx]
            
            # Process through expert
            expert_output = self.experts[expert_idx](tokens)
            
            # Add weighted expert output back to result
            output[batch_idx, seq_idx] += expert_output * weights.unsqueeze(-1)
        
        return output

class MoEFFN(nn.Module):
    """
    Mixture of Experts Feed-Forward Network with SRWKV routing
    Integrates neuromorphic attention mechanisms for expert selection
    Supports PyTorch-to-neuromorphic translation patterns
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 num_experts: int,
                 vocab_size: int,
                 top_k: int = 2,
                 dropout: float = 0.1,
                 router_temperature: float = 1.0,
                 load_balance_weight: float = 0.01,
                 use_spiking: bool = False,
                 use_norse: bool = False,
                 router_config: Optional[Dict[str, Any]] = None,
                 enable_expert_evaluation: bool = True,
                 enable_neurogenesis: bool = False,
                 neurogenesis_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.enable_neurogenesis = enable_neurogenesis
        self.num_experts = num_experts

        # Build router configuration
        config = router_config or {}
        config.update({
            'embedding_dim': d_model,
            'num_heads': config.get('num_heads', 8),
            'spike_threshold': config.get('spike_threshold', 0.5),
            'decay_factor': config.get('decay_factor', 0.9),
            'liquid_hidden': config.get('liquid_hidden', 32),
            'liquid_dt': config.get('liquid_dt', 0.02),
            'neuron_top_k': top_k,
            'num_neurons_per_head': num_experts,
            'vocab_size': vocab_size,
            'k_winners': config.get('k_winners', 3),
            'exploration_bonus': config.get('exploration_bonus', 0.1),
            'adaptation_rate': config.get('adaptation_rate', 0.1),
            'temp_adaptation_rate': config.get('temp_adaptation_rate', 0.05)
        })
        
        # Initialize SRWKV router and MoE layer
        self.router = SRWKVRouter(config)
        
        if self.enable_neurogenesis:
            n_config = NeurogenesisConfig(**(neurogenesis_config or {}))
            self.moe_layer = NeurogenesisMoELayer(
                d_model=d_model,
                d_ff=d_ff,
                num_categories=8, # Example, should be in config
                initial_specialists=num_experts // 8 if num_experts else 2, # Example
                neurogenesis_config=n_config
            )
        else:
            self.moe_layer = MoELayer(d_model, d_ff, num_experts, top_k, dropout, 
                                     use_spiking=use_spiking, use_norse=use_norse)
        
        self.load_balance_weight = load_balance_weight
        self.use_spiking = use_spiking
        
        # Move MoE layer to same device as router
        device = self.router.srwkv.device
        self.moe_layer = self.moe_layer.to(device)
        
        # Expert evaluator
        self.enable_expert_evaluation = enable_expert_evaluation
        if self.enable_expert_evaluation:
            self.expert_evaluator = create_expert_evaluator(
                num_experts=num_experts,
                evaluation_window=100,
                utility_weights={
                    'performance': 0.3,
                    'cost': 0.2,
                    'marginal': 0.2,
                    'diversity': 0.15,
                    'calibration': 0.15
                }
            )
        else:
            self.expert_evaluator = None

    def _update_expert_metrics(self,
                               hidden_states: torch.Tensor,
                               top_k_indices: torch.Tensor,
                               top_k_weights: torch.Tensor,
                               output: torch.Tensor,
                               router_info: Dict[str, Any]) -> None:
        """Safely update expert evaluation metrics if evaluator is enabled.

        This maintains backward compatibility with tests expecting the method.
        """
        if not getattr(self, 'enable_expert_evaluation', False):
            return
        if self.expert_evaluator is None:
            return
        try:
            # Basic metrics: utilization and average weights per expert
            # Flatten over batch/seq
            bsz, seqlen, k = top_k_indices.shape
            flat_indices = top_k_indices.reshape(-1, k)
            flat_weights = top_k_weights.reshape(-1, k)
            # Accumulate per-expert weight sum
            num_experts_to_eval = self.num_experts
            if self.enable_neurogenesis and hasattr(self.moe_layer, 'get_neurogenesis_stats'):
                stats = self.moe_layer.get_neurogenesis_stats()
                num_experts_to_eval = stats.get('total_experts', self.num_experts)

            util = torch.zeros(num_experts_to_eval, device=flat_indices.device)
            for i in range(k):
                idx = flat_indices[:, i]
                w = flat_weights[:, i]
                util.index_add_(0, idx, w)
            # Router temperature and entropy if provided
            temp = float(getattr(self.router.srwkv, 'temperature', torch.tensor(1.0)))
            entropy = float(router_info.get('utilization_entropy', 0.0)) if isinstance(router_info, dict) else 0.0
            # Update evaluator (API: update(expert_utilization, metadata))
            self.expert_evaluator.update(
                expert_utilization=util.detach().cpu().tolist(),
                metadata={
                    'router_temperature': temp,
                    'utilization_entropy': entropy
                }
            )
        except Exception:
            # Never break forward path due to metrics
            pass
    
    def forward(self, hidden_states: torch.Tensor, 
                token_ids: Optional[torch.Tensor] = None, 
                text: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through MoE with SRWKV routing"""
        
        # Route to experts using SRWKV
        top_k_indices, top_k_weights, router_info = self.router(hidden_states, token_ids, text)
        
        # Process through selected experts
        output = self.moe_layer(hidden_states, top_k_indices, top_k_weights)
        
        # Update expert evaluation metrics if enabled
        if self.enable_expert_evaluation and self.expert_evaluator is not None:
            self._update_expert_metrics(hidden_states, top_k_indices, top_k_weights, output, router_info)
        
        # Prepare auxiliary information
        aux = {
            'load_balance_loss': router_info.get('load_balance_loss', 0.0) * self.load_balance_weight,
            'router_probs': router_info.get('router_probs'),
            'utilization_entropy': router_info.get('utilization_entropy'),
            'expert_counts': router_info.get('expert_counts'),
            'energy_metrics': router_info.get('energy_metrics'),
            'bandit_rewards_ema': router_info.get('bandit_rewards_ema'),
            'temperature': router_info.get('temperature'),
            'top_k_indices': top_k_indices,
            'top_k_weights': top_k_weights
        }
        
        return output, aux
    
    def update_router_stats(self, expert_indices: torch.Tensor, token_losses: torch.Tensor):
        """Update router bandit statistics with expert performance feedback"""
        self.router.update_bandit_stats(expert_indices, token_losses)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing and efficiency statistics"""
        return self.router.srwkv.get_spike_statistics()

    def get_neurogenesis_stats(self) -> Dict[str, Any]:
        """Get neurogenesis statistics if the layer supports it."""
        if self.enable_neurogenesis and hasattr(self.moe_layer, 'get_neurogenesis_stats'):
            return self.moe_layer.get_neurogenesis_stats()
        return {}

class HierarchicalMoEFFN(nn.Module):
    """
    Hierarchical Mixture of Experts Feed-Forward Network with SRWKV routing
    Supports 8 categories × 8 specialties = 64 total experts
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 num_categories: int = 8,
                 num_specialties: int = 8,
                 vocab_size: int = 5000,
                 top_k: int = 2,
                 dropout: float = 0.1,
                 load_balance_weight: float = 0.01,
                 router_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Build router configuration for hierarchical mode
        config = router_config or {}
        config.update({
            'embedding_dim': d_model,
            'num_heads': config.get('num_heads', 8),
            'spike_threshold': config.get('spike_threshold', 0.5),
            'decay_factor': config.get('decay_factor', 0.9),
            'liquid_hidden': config.get('liquid_hidden', 32),
            'neuron_top_k': top_k,
            'num_neurons_per_head': num_categories * num_specialties,  # Total experts
            'vocab_size': vocab_size,
            'exploration_bonus': config.get('exploration_bonus', 0.1),
            'adaptation_rate': config.get('adaptation_rate', 0.1),
            'temp_adaptation_rate': config.get('temp_adaptation_rate', 0.05),
            # Hierarchical routing parameters
            'hierarchical_mode': True,
            'num_categories': num_categories,
            'num_specialties': num_specialties
        })
        
        # Initialize hierarchical router and MoE layer
        self.router = SRWKVRouter(config)
        self.moe_layer = HierarchicalMoELayer(d_model, d_ff, num_categories, num_specialties, top_k, dropout)
        self.load_balance_weight = load_balance_weight
        
        # Move MoE layer to correct device
        device = self.router.srwkv.device
        self.moe_layer = self.moe_layer.to(device)
    
    def forward(self, hidden_states: torch.Tensor, 
                token_ids: Optional[torch.Tensor] = None, 
                text: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through hierarchical MoE with SRWKV routing"""
        # Get hierarchical routing decisions
        cat_indices, sub_indices, weights, router_info = self.router(hidden_states, token_ids, text)
        
        # Process through hierarchical MoE layer
        output = self.moe_layer(hidden_states, cat_indices, sub_indices, weights)
        
        # Build auxiliary information
        aux = {
            'load_balance_loss': router_info.get('load_balance_loss', 0.0) * self.load_balance_weight,
            'cat_probs': router_info.get('cat_probs'),
            'sub_probs': router_info.get('sub_probs'),
            'utilization_entropy': router_info.get('utilization_entropy'),
            'energy_metrics': router_info.get('energy_metrics'),
            'bandit_rewards_ema': router_info.get('bandit_rewards_ema'),
            'temperature': router_info.get('temperature'),
            'cat_indices': cat_indices,
            'sub_indices': sub_indices,
            'weights': weights
        }
        
        return output, aux
    
    def update_router_stats(self, expert_indices: torch.Tensor, token_losses: torch.Tensor):
        """Update router bandit statistics with expert performance feedback"""
        self.router.update_bandit_stats(expert_indices, token_losses)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing and efficiency statistics"""
        return self.router.srwkv.get_spike_statistics()
    
    def get_expert_utilization(self) -> Dict[str, torch.Tensor]:
        """Get expert utilization statistics for hierarchical structure"""
        stats = self.get_routing_statistics()
        return {
            'cat_utilization': stats.get('cat_probs', torch.zeros(self.router.num_categories)),
            'sub_utilization': stats.get('sub_probs', torch.zeros(self.router.num_specialties)),
            'total_experts': self.router.num_categories * self.router.num_specialties
        }
