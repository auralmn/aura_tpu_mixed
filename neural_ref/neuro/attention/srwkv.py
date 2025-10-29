# SPDX-License-Identifier: Apache-2.0
"""
AURA Neuromorphic SRWKV: Spiking attention with bandit-aided routing
- SRWKV O(T·d) attention with liquid dynamics  
- Multi-channel spiking attention with prosody
- k-WTA competition and energy metering
- Bandit-aided expert routing with UCB exploration
- Adaptive temperature and NLMS learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import re
import math
from .interfaces import AttentionMechanism
try:
    import norse.torch as snn  # type: ignore
    from norse.torch.module.lif import LIFCell  # type: ignore
    _NORSE_AVAILABLE = True
except Exception:
    LIFCell = object  # fallback
    _NORSE_AVAILABLE = False

@dataclass
class EnergyMeter:
    """Energy tracking for neuromorphic efficiency analysis"""
    mac_energy_j: float = 3e-12  # ~3 pJ per MAC operation
    total_energy: float = 0.0
    operation_counts: Dict[str, int] = field(default_factory=dict)
    
    def count_operations(self, **ops):
        """Count operations: macs=N, adds=M, etc."""
        for op_type, count in ops.items():
            self.operation_counts[op_type] = self.operation_counts.get(op_type, 0) + count
            if op_type == 'macs':
                self.total_energy += count * self.mac_energy_j
    
    def reset(self):
        """Reset counters"""
        self.total_energy = 0.0
        self.operation_counts.clear()
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get energy efficiency metrics"""
        return {
            'total_energy_j': self.total_energy,
            'energy_per_mac_j': self.mac_energy_j,
            'total_macs': self.operation_counts.get('macs', 0),
            'ops_per_joule': self.operation_counts.get('macs', 0) / max(self.total_energy, 1e-12)
        }

class BanditStats:
    """Maintains bandit statistics for each expert/neuron"""

    def __init__(self, num_experts: int, device: torch.device):
        self.num_experts = num_experts
        self.device = device

        self.q_values = torch.zeros(num_experts, device=device)
        self.counts = torch.zeros(num_experts, device=device)
        self.total_steps = 0

    def update(self, expert_ids: torch.Tensor, rewards: torch.Tensor):
        """Update Q-values with new rewards"""
        for i, reward in zip(expert_ids, rewards):
            self.counts[i] += 1
            self.q_values[i] += (reward - self.q_values[i]) / self.counts[i]
        self.total_steps += len(expert_ids)

    def get_ucb_bonus(self, c: float = 2.0) -> torch.Tensor:
        """Compute UCB exploration bonus"""
        log_t = np.log(max(self.total_steps, 1))
        bonus = c * torch.sqrt(log_t / (self.counts + 1))
        return bonus

    def get_utilization_entropy(self) -> float:
        """Compute entropy of expert utilization"""
        if self.total_steps == 0:
            return 0.0

        probs = self.counts / self.counts.sum()
        probs = probs + 1e-8
        entropy = -torch.sum(probs * torch.log(probs))
        return entropy.item()

class LiquidCell(nn.Module):
    """
    Liquid (continuous-time) neural dynamics cell
    Based on Neural ODEs with adaptive time constants
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 0.02):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        
        self.W_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_in = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.W_tau = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.tau_bias = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        
        self.register_buffer('hidden_state', torch.zeros(1, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One step of liquid dynamics"""
        batch_size = x.size(0)
        
        if self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_dim, 
                                          device=x.device, dtype=x.dtype)
        
        tau = torch.sigmoid(torch.matmul(x, self.W_tau.t()) + self.tau_bias)
        tau = tau * 0.5 + 0.02
        
        input_contrib = torch.matmul(x, self.W_in.t()) + self.bias
        rec_contrib = torch.matmul(self.hidden_state, self.W_rec.t())
        
        activation = torch.tanh(input_contrib + rec_contrib)
        dhdt = (-self.hidden_state / tau) + activation
        
        self.hidden_state = self.hidden_state + self.dt * dhdt
        return self.hidden_state
    
    def reset_state(self, batch_size: int = 1):
        """Reset hidden state"""
        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, device=self.hidden_state.device)

class MultiChannelSpikingAttention(nn.Module):
    """Multi-channel spiking attention with amplitude/pitch/boundary streams"""
    
    def __init__(self, 
                 vocab_size: int,
                 k_winners: int = 5,
                 decay: float = 0.7,
                 threshold: float = 1.0,
                 gain_up: float = 1.8,
                 gain_down: float = 0.6):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.k_winners = k_winners
        self.decay = decay
        self.threshold = threshold
        self.gain_up = gain_up
        self.gain_down = gain_down
        
        self.channel_weights = nn.Parameter(torch.ones(3))  # [amplitude, pitch, boundary]
        self.energy_meter = EnergyMeter()
        
    def extract_prosody_features(self, text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract amplitude, pitch, and boundary features from text"""
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        amplitude = []
        pitch = []
        boundary = []
        
        for token in tokens:
            amp = 0.5
            if token.isupper():
                amp += 0.3
            if len(token) > 5:
                amp += 0.2
            amplitude.append(amp)
            
            vowels = sum(1 for c in token.lower() if c in 'aeiouy')
            pitch_val = 0.5 + 0.3 * min(vowels / max(len(token), 1), 1.0)
            pitch.append(pitch_val)
            
            boundary.append(1.0 if re.match(r'[.!?,:;]', token) else 0.0)
        
        return (np.array(amplitude, dtype=np.float32),
                np.array(pitch, dtype=np.float32), 
                np.array(boundary, dtype=np.float32))
    
    def lif_process(self, channel_data: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """Leaky integrate-and-fire processing for one channel"""
        membrane_potential = 0.0
        spikes = []
        spike_counts = {}
        
        for i, value in enumerate(channel_data):
            membrane_potential = self.decay * membrane_potential + value
            
            if membrane_potential >= self.threshold:
                spikes.append(i)
                spike_counts[i] = spike_counts.get(i, 0) + 1
                membrane_potential -= self.threshold
        
        return np.array(spikes), spike_counts
    
    def forward(self, token_ids: torch.Tensor, text: Optional[str] = None) -> torch.Tensor:
        """Compute multi-channel attention gains"""
        if text is not None:
            amp_data, pitch_data, bound_data = self.extract_prosody_features(text)
        else:
            seq_len = len(token_ids)
            amp_data = np.ones(seq_len) * 0.5
            pitch_data = np.ones(seq_len) * 0.5  
            bound_data = np.zeros(seq_len)
        
        amp_spikes, amp_counts = self.lif_process(amp_data)
        pitch_spikes, pitch_counts = self.lif_process(pitch_data)
        bound_spikes, bound_counts = self.lif_process(bound_data)
        
        total_ops = len(amp_data) * 3
        self.energy_meter.count_operations(macs=total_ops)
        
        channel_weights = torch.softmax(self.channel_weights, dim=0)
        w_amp, w_pitch, w_bound = channel_weights
        
        token_activity = {}
        for i, token_id in enumerate(token_ids):
            token_id = token_id.item()
            
            activity = (w_amp * amp_counts.get(i, 0) + 
                       w_pitch * pitch_counts.get(i, 0) +
                       w_bound * bound_counts.get(i, 0))
            
            token_activity[token_id] = token_activity.get(token_id, 0) + activity
        
        if len(token_activity) == 0:
            return torch.ones(self.vocab_size)
        
        ranked_tokens = sorted(token_activity.items(), key=lambda x: -x[1])
        winners = set([token_id for token_id, _ in ranked_tokens[:self.k_winners]])
        
        gains = torch.ones(self.vocab_size)
        for token_id, activity in token_activity.items():
            if activity > 0:
                gains[token_id] = self.gain_up if token_id in winners else self.gain_down
        
        return gains

class BanditEnhancedNeuronRouter(nn.Module):
    """Bandit-aided neuron/expert routing for SRWKV heads"""

    def __init__(self, d_model: int, num_neurons: int, top_k: int = 2, 
                 temperature: float = 1.0, exploration_bonus: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_neurons = num_neurons
        # Vitals-related counters (for external vitals/fatigue consumers)
        self.total_routed_spikes: int = 0
        self._usage_updates: int = 0
        self.avg_usage_ratio: float = 0.0  # fraction of active neurons per routing step

        # Router params and layers
        self.top_k = top_k
        self.temperature = temperature
        self.exploration_bonus = exploration_bonus

        self.router = nn.Linear(d_model, num_neurons, bias=True)
        self.bandit_stats = None

        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)

    def record_activity(self, active_count: int):
        """Record how many neurons were active this routing step.

        Keeps a running average of usage ratio and updates spike counter proxy.
        """
        try:
            active = max(0, int(active_count))
            total = max(1, int(self.num_neurons))
            ratio = min(1.0, max(0.0, active / total))
            # Update running average
            self._usage_updates += 1
            self.avg_usage_ratio += (ratio - self.avg_usage_ratio) / float(self._usage_updates)
            # Treat active neurons as a proxy for spikes routed this step
            self.total_routed_spikes += active
        except Exception:
            pass

    def record_spikes(self, spikes: int):
        """Optionally record an external spike count for this step."""
        try:
            self.total_routed_spikes += max(0, int(spikes))
        except Exception:
            pass

    def get_bandit_stats(self) -> dict:
        """Expose lightweight routing vitals for diagnostics/vitals."""
        return {
            'avg_usage_ratio': float(self.avg_usage_ratio),
            'total_routed_spikes': int(self.total_routed_spikes),
            'num_neurons': int(self.num_neurons)
        }

    def update_bandit_stats(self, neuron_indices: torch.Tensor, rewards: torch.Tensor):
        """Update internal bandit stats with observed rewards for chosen neurons."""
        try:
            if self.bandit_stats is None:
                device = neuron_indices.device if isinstance(neuron_indices, torch.Tensor) else torch.device('cpu')
                self.bandit_stats = BanditStats(self.num_neurons, device)
            self.bandit_stats.update(neuron_indices.view(-1), rewards.view(-1))
        except Exception:
            pass

def _count_spiking_neurons(module: nn.Module) -> int:
    """Sum LIF neuron populations across a module tree (Norse)."""
    if not _NORSE_AVAILABLE:
        return 0
    total = 0
    for sub in module.modules():
        if isinstance(sub, LIFCell):
            for name, param in sub.named_parameters():
                if name == 'weight' and param.dim() >= 2:
                    total += int(param.shape[0])
                    break
    return total


def _count_synapses(module: nn.Module) -> int:
    """Sum synaptic weights for all LIF populations (Norse)."""
    if not _NORSE_AVAILABLE:
        return 0
    total = 0
    for sub in module.modules():
        if isinstance(sub, LIFCell):
            for name, param in sub.named_parameters():
                if name == 'weight':
                    total += int(param.numel())
                    break
    return total

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Route to top-k neurons with bandit exploration"""
        batch_size, seq_len, d_model = hidden_states.shape

        if self.bandit_stats is None:
            self.bandit_stats = BanditStats(self.num_neurons, hidden_states.device)

        hidden_flat = hidden_states.view(-1, d_model)

        # Compute router logits
        router_logits = self.router(hidden_flat)

        # Add UCB exploration bonus
        if self.training and self.exploration_bonus > 0:
            ucb_bonus = self.bandit_stats.get_ucb_bonus() * self.exploration_bonus
            router_logits = router_logits + ucb_bonus.unsqueeze(0)

        router_logits = router_logits / self.temperature
        router_probs = F.softmax(router_logits, dim=-1)

        # Select Top-K neurons
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        top_k_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
        top_k_weights = top_k_weights.view(batch_size, seq_len, self.top_k)

        # Record activity for vitals (how many neurons were active)
        try:
            # Correctly calculate unique active neurons across the batch and sequence
            unique_neurons = torch.unique(top_k_indices.view(-1))
            self.record_activity(active_count=len(unique_neurons))
        except Exception:
            pass

        # Load balancing loss
        mean_probs = router_probs.mean(dim=0)
        uniform_dist = torch.ones_like(mean_probs) / self.num_neurons
        load_balance_loss = F.kl_div(mean_probs.log(), uniform_dist, reduction='batchmean')

        router_info = {
            'load_balance_loss': load_balance_loss,
            'router_probs': router_probs.view(batch_size, seq_len, self.num_neurons),
            'utilization_entropy': self.bandit_stats.get_utilization_entropy(),
            'neuron_counts': self.bandit_stats.counts.clone()
        }

        return top_k_indices, top_k_weights, router_info

    def update_bandit_stats(self, neuron_indices: torch.Tensor, rewards: torch.Tensor):
        """Update bandit statistics with neuron rewards"""
        if self.bandit_stats is not None:
            self.bandit_stats.update(neuron_indices, rewards)

class SpikingSRWKV(AttentionMechanism, nn.Module):
    """
    Neuromorphic SRWKV with enhanced features:
    - SRWKV O(T·d) attention with liquid dynamics
    - Multi-channel spiking attention (amplitude, pitch, boundary)  
    - k-WTA competition and adaptive learning rates
    - Bandit-aided neuron routing with UCB exploration
    - Energy metering and efficiency tracking
    - Adaptive temperature and membrane dynamics
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        AttentionMechanism.__init__(self, module_id, config)
        nn.Module.__init__(self)
        
        # Core SRWKV parameters
        self.embedding_dim = config.get('embedding_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.head_dim = self.embedding_dim // self.num_heads
        self.spike_threshold = config.get('spike_threshold', 0.5)
        self.decay_factor = config.get('decay_factor', 0.9)
        
        # Liquid dynamics parameters
        self.liquid_hidden = config.get('liquid_hidden', 64)
        self.liquid_dt = config.get('liquid_dt', 0.02)
        
        # Multi-channel attention parameters
        self.vocab_size = config.get('vocab_size', 32000)
        self.k_winners = config.get('k_winners', 5)

        # Bandit routing parameters
        self.num_neurons_per_head = config.get('num_neurons_per_head', 16)
        self.neuron_top_k = config.get('neuron_top_k', 2)
        self.exploration_bonus = config.get('exploration_bonus', 0.1)
        
        # Adaptive learning parameters
        self.mu_token = config.get('mu_token', 0.3)
        self.mu_context = config.get('mu_context', 0.8)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.temp_adaptation_rate = config.get('temp_adaptation_rate', 0.1)
        
        # SRWKV projections with liquid enhancement
        self.receptance = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) 
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.output_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Liquid dynamics cells for each projection
        self.liquid_k = LiquidCell(self.embedding_dim, self.liquid_hidden, self.liquid_dt)
        self.liquid_v = LiquidCell(self.embedding_dim, self.liquid_hidden, self.liquid_dt)
        self.liquid_r = LiquidCell(self.embedding_dim, self.liquid_hidden, self.liquid_dt)
        
        # Liquid-to-embedding projections
        self.liquid_k_proj = nn.Linear(self.liquid_hidden, self.embedding_dim)
        self.liquid_v_proj = nn.Linear(self.liquid_hidden, self.embedding_dim)
        self.liquid_r_proj = nn.Linear(self.liquid_hidden, self.embedding_dim)

        # Bandit-aided neuron routers for each head
        self.neuron_routers = nn.ModuleList([
            BanditEnhancedNeuronRouter(
                d_model=self.head_dim,
                num_neurons=self.num_neurons_per_head,
                top_k=self.neuron_top_k,
                exploration_bonus=self.exploration_bonus
            ) for _ in range(self.num_heads)
        ])
        
        # Time mixing weights
        self.time_mix_k = nn.Parameter(torch.ones(self.embedding_dim))
        self.time_mix_v = nn.Parameter(torch.ones(self.embedding_dim)) 
        self.time_mix_r = nn.Parameter(torch.ones(self.embedding_dim))
        
        # Multi-channel attention
        self.multichannel_attention = MultiChannelSpikingAttention(
            vocab_size=self.vocab_size,
            k_winners=self.k_winners,
            gain_up=config.get('gain_up', 1.8),
            gain_down=config.get('gain_down', 0.6)
        )
        
        # Energy metering
        self.energy_meter = EnergyMeter()
        
        # Adaptive components
        self.register_buffer('learning_rates', torch.ones(self.embedding_dim))
        self.register_buffer('adaptation_ema', torch.tensor(0.0))
        self.register_buffer('temperature', torch.tensor(1.0))
        self.register_buffer('bandit_rewards_ema', torch.tensor(0.0))
        
        # State buffers
        self.register_buffer('prev_state', torch.zeros(1, self.embedding_dim))
        self.register_buffer('attention_weights', torch.zeros(1, 1))
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    def initialize(self) -> bool:
        try:
            # Initialize SRWKV weights
            nn.init.xavier_uniform_(self.receptance.weight, gain=0.5)
            nn.init.xavier_uniform_(self.key.weight, gain=0.5)
            nn.init.xavier_uniform_(self.value.weight, gain=0.5)
            nn.init.xavier_uniform_(self.output_projection.weight, gain=0.5)
            
            # Initialize liquid projections
            nn.init.xavier_uniform_(self.liquid_k_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.liquid_v_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.liquid_r_proj.weight, gain=0.5)
            
            # Initialize time mixing
            nn.init.uniform_(self.time_mix_k, 0.1, 0.9)
            nn.init.uniform_(self.time_mix_v, 0.1, 0.9)
            nn.init.uniform_(self.time_mix_r, 0.1, 0.9)
            
            self.to(self.device)
            return True
        except Exception as e:
            self.logger.error(f"SpikingSRWKV initialization failed: {e}")
            return False
    
    def spike_activation(self, x: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """Spike activation with bandit-influenced threshold"""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Get bandit stats for adaptive threshold
        bandit_entropy = 0.0
        if head_idx < len(self.neuron_routers):
            router = self.neuron_routers[head_idx]
            if router.bandit_stats is not None:
                bandit_entropy = router.bandit_stats.get_utilization_entropy()

        # Adaptive threshold based on bandit exploration and temperature
        exploration_factor = 1.0 + 0.1 * bandit_entropy
        adaptive_threshold = self.spike_threshold * exploration_factor / self.temperature

        spikes = (x > adaptive_threshold).float()

        if self.training:
            spike_grad = torch.clamp(1 - torch.abs(x - adaptive_threshold), 0, 1)
            spikes = spikes + spike_grad - spike_grad.detach()

        return spikes
    
    def time_mixing(self, x: torch.Tensor, prev_x: torch.Tensor, 
                    mix_weight: torch.Tensor, liquid_cell: Optional[LiquidCell] = None,
                    liquid_proj: Optional[nn.Module] = None, learning_gains: Optional[torch.Tensor] = None,
                    head_idx: int = 0) -> torch.Tensor:
        """Time mixing with liquid dynamics and bandit-influenced routing"""
        x = x.to(self.device)
        prev_x = prev_x.to(self.device)
        mix_weight = mix_weight.to(self.device)

        # Process through liquid dynamics (if available)
        if liquid_cell is not None and liquid_proj is not None:
            liquid_output = liquid_cell(x)
            liquid_features = liquid_proj(liquid_output)
        else:
            liquid_features = torch.zeros_like(x)

        # Bandit routing influence (simplified to avoid dimension issues)
        bandit_influence = 1.0
        if head_idx < len(self.neuron_routers) and x.shape[-1] == self.embedding_dim:
            router = self.neuron_routers[head_idx]
            if router.bandit_stats is not None:
                # Simple bandit influence based on utilization entropy
                entropy = router.bandit_stats.get_utilization_entropy()
                bandit_influence = 1.0 + 0.05 * entropy  # Small influence to avoid instability

        # Apply learning rate gains
        if learning_gains is not None:
            learning_gains = learning_gains.to(self.device)
            if learning_gains.shape[-1] == mix_weight.shape[-1]:
                mix_weight = mix_weight * learning_gains

        # Enhanced mixing with liquid features and bandit influence
        mixed = (mix_weight * (x + 0.1 * liquid_features * bandit_influence) + 
                (1 - mix_weight) * prev_x)

        return mixed
    
    def compute_attention(self, query: torch.Tensor, key: torch.Tensor, 
                         value: torch.Tensor, token_ids: Optional[torch.Tensor] = None,
                         text: Optional[str] = None) -> torch.Tensor:
        """Attention computation with enhanced features"""
        # Ensure input is on correct device
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        
        batch_size, seq_len, embed_dim = query.shape
        
        # Reset liquid cells
        self.liquid_k.reset_state(batch_size)
        self.liquid_v.reset_state(batch_size)
        self.liquid_r.reset_state(batch_size)
        
        # Compute multi-channel attention gains
        if token_ids is not None:
            token_ids = token_ids.to(self.device)
            multichannel_gains = self.multichannel_attention(token_ids.flatten(), text)
            
            if len(multichannel_gains) >= embed_dim:
                learning_gains = multichannel_gains[:embed_dim]
            else:
                learning_gains = torch.ones(embed_dim, device=self.device)
                learning_gains[:len(multichannel_gains)] = multichannel_gains
        else:
            learning_gains = torch.ones(embed_dim, device=self.device)
        
        # Initialize state for this batch
        current_state = torch.zeros(batch_size, embed_dim, device=self.device)
        outputs = []
        
        # Track metrics for comprehensive adaptation
        attention_entropies = []
        bandit_rewards = []
        
        for t in range(seq_len):
            x_t = query[:, t, :]
            
            # Enhanced time mixing with bandit influence
            k_input = self.time_mixing(x_t, current_state, self.time_mix_k, 
                                     self.liquid_k, self.liquid_k_proj, learning_gains, 0)
            v_input = self.time_mixing(x_t, current_state, self.time_mix_v,
                                     self.liquid_v, self.liquid_v_proj, learning_gains, 1 % self.num_heads)
            r_input = self.time_mixing(x_t, current_state, self.time_mix_r,
                                     self.liquid_r, self.liquid_r_proj, learning_gains, 2 % self.num_heads)
            
            # SRWKV projections
            k_t = self.key(k_input)
            v_t = self.value(v_input)
            r_t = self.receptance(r_input)
            
            # Spike activation with bandit influence
            k_spikes = self.spike_activation(k_t, 0)
            v_spikes = self.spike_activation(v_t, 1 % self.num_heads) 
            r_spikes = self.spike_activation(r_t, 2 % self.num_heads)
            
            # Record spike counts for vitals before they are used
            if self.neuron_routers:
                # Approximate which router is responsible based on head index
                self.neuron_routers[0].record_spikes(k_spikes.sum().item())
                self.neuron_routers[1 % self.num_heads].record_spikes(v_spikes.sum().item())
                self.neuron_routers[2 % self.num_heads].record_spikes(r_spikes.sum().item())

            # Update bandit routers with simple reward signal
            spike_quality = (torch.mean(k_spikes) + torch.mean(v_spikes) + torch.mean(r_spikes)) / 3.0
            bandit_rewards.append(spike_quality.item())
            
            # Count operations for energy tracking
            macs = embed_dim * embed_dim * 3  # Three projections
            macs += self.liquid_hidden * embed_dim * 3  # Liquid processing
            macs += self.num_neurons_per_head * self.head_dim * self.num_heads  # Bandit routing
            self.energy_meter.count_operations(macs=macs)
            
            # Enhanced attention computation
            attention_score = torch.sum(k_spikes * v_spikes, dim=-1, keepdim=True) / (torch.sum(k_spikes, dim=-1, keepdim=True) + 1e-8)
            
            # Apply k-WTA competition with bandit influence
            if attention_score.shape[0] > 1:
                k_val = min(self.k_winners, attention_score.shape[0])
                top_k_vals, top_k_indices = torch.topk(attention_score.squeeze(-1), k_val, dim=0)
                
                competitive_score = torch.zeros_like(attention_score.squeeze(-1))
                competitive_score[top_k_indices] = top_k_vals
                attention_score = competitive_score.unsqueeze(-1)
            
            # Track comprehensive metrics
            if attention_score.numel() > 1:
                probs = torch.softmax(attention_score.flatten(), dim=0)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                attention_entropies.append(entropy.item())
                
                # Bandit rewards based on attention quality
                attention_quality = torch.var(attention_score) + torch.mean(attention_score)
                bandit_rewards.append(attention_quality.item())
            
            # Final output
            output_t = r_spikes * attention_score
            outputs.append(output_t)
            
            # Update state with comprehensive adaptation
            adaptive_decay = self.decay_factor * learning_gains.mean()
            current_state = adaptive_decay * current_state + (1 - adaptive_decay) * x_t
        
        # Update all adaptation metrics
        if attention_entropies:
            mean_entropy = np.mean(attention_entropies)
            self.adaptation_ema.data = (1 - self.adaptation_rate) * self.adaptation_ema.data + self.adaptation_rate * mean_entropy
            
            # Adaptive temperature with bandit feedback
            if bandit_rewards:
                mean_reward = np.mean(bandit_rewards)
                self.bandit_rewards_ema.data = (1 - self.adaptation_rate) * self.bandit_rewards_ema.data + self.adaptation_rate * mean_reward
                
                target_temp = 0.5 + 1.5 * torch.sigmoid(torch.tensor(mean_entropy - 2.0 + 0.1 * mean_reward, dtype=torch.float32, device=self.device))
                target_temp = torch.clamp(target_temp, 0.5, 2.0)
                self.temperature.data = (1 - self.temp_adaptation_rate) * self.temperature.data + self.temp_adaptation_rate * target_temp
        
        # Stack outputs and apply final projection
        output_sequence = torch.stack(outputs, dim=1)
        output_sequence = self.output_projection(output_sequence)
        
        # Update stored state with last state from batch
        self.prev_state = current_state[-1:].detach()
        
        return output_sequence
    
    def process_spikes(self, spikes: torch.Tensor, token_ids: Optional[torch.Tensor] = None,
                      text: Optional[str] = None) -> torch.Tensor:
        """Process spikes with enhanced features"""
        return self.compute_attention(spikes, spikes, spikes, token_ids, text)
    
    def get_spike_statistics(self) -> Dict[str, float]:
        """Spike and efficiency statistics"""
        base_stats = {
            'avg_attention_weight': float(torch.mean(self.attention_weights)),
            'state_magnitude': float(torch.norm(self.prev_state)),
            'spike_threshold': self.spike_threshold,
            'adaptation_ema': float(self.adaptation_ema),
            'temperature': float(self.temperature),
            'bandit_rewards_ema': float(self.bandit_rewards_ema)
        }
        
        # Add energy efficiency metrics
        base_stats.update(self.energy_meter.get_efficiency_metrics())
        
        # Add multi-channel statistics
        multichannel_stats = self.multichannel_attention.energy_meter.get_efficiency_metrics()
        base_stats.update({f'multichannel_{k}': v for k, v in multichannel_stats.items()})
        
        # Add bandit routing statistics
        for i, router in enumerate(self.neuron_routers):
            if router.bandit_stats is not None:
                base_stats[f'head_{i}_utilization_entropy'] = router.bandit_stats.get_utilization_entropy()
                base_stats[f'head_{i}_total_steps'] = router.bandit_stats.total_steps
        
        return base_stats
    
    def get_attention_weights(self) -> torch.Tensor:
        return self.attention_weights.clone()
    
    # AURAModule interface methods
    def process(self, input_data: Any) -> Any:
        if isinstance(input_data, torch.Tensor):
            return self.process_spikes(input_data)
        elif isinstance(input_data, dict):
            spikes = input_data.get('spikes')
            token_ids = input_data.get('token_ids')
            text = input_data.get('text')
            if spikes is not None:
                return self.process_spikes(spikes, token_ids, text)
        raise ValueError("SpikingSRWKV requires spike tensor input")
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'prev_state': self.prev_state.detach().cpu().numpy(),
            'attention_weights': self.attention_weights.detach().cpu().numpy(),
            'learning_rates': self.learning_rates.detach().cpu().numpy(),
            'adaptation_ema': self.adaptation_ema.detach().cpu().numpy(),
            'temperature': self.temperature.detach().cpu().numpy(),
            'bandit_rewards_ema': self.bandit_rewards_ema.detach().cpu().numpy(),
            'liquid_states': {
                'liquid_k': self.liquid_k.hidden_state.detach().cpu().numpy(),
                'liquid_v': self.liquid_v.hidden_state.detach().cpu().numpy(),
                'liquid_r': self.liquid_r.hidden_state.detach().cpu().numpy()
            },
            'multichannel_state': {
                'channel_weights': self.multichannel_attention.channel_weights.detach().cpu().numpy()
            },
            'bandit_states': [
                {
                    'q_values': router.bandit_stats.q_values.cpu().numpy() if router.bandit_stats else None,
                    'counts': router.bandit_stats.counts.cpu().numpy() if router.bandit_stats else None,
                    'total_steps': router.bandit_stats.total_steps if router.bandit_stats else 0
                } for router in self.neuron_routers
            ],
            'energy_metrics': self.energy_meter.get_efficiency_metrics(),
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'num_heads': self.num_heads,
                'spike_threshold': self.spike_threshold,
                'decay_factor': self.decay_factor,
                'liquid_hidden': self.liquid_hidden,
                'k_winners': self.k_winners,
                'vocab_size': self.vocab_size,
                'num_neurons_per_head': self.num_neurons_per_head,
                'neuron_top_k': self.neuron_top_k
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        try:
            # Restore basic state
            self.prev_state = torch.from_numpy(state['prev_state']).to(self.device)
            self.attention_weights = torch.from_numpy(state['attention_weights']).to(self.device)
            self.learning_rates = torch.from_numpy(state['learning_rates']).to(self.device)
            self.adaptation_ema = torch.from_numpy(state['adaptation_ema']).to(self.device)
            self.temperature = torch.from_numpy(state['temperature']).to(self.device)
            self.bandit_rewards_ema = torch.from_numpy(state['bandit_rewards_ema']).to(self.device)
            
            # Restore liquid states
            liquid_states = state['liquid_states']
            self.liquid_k.hidden_state = torch.from_numpy(liquid_states['liquid_k']).to(self.device)
            self.liquid_v.hidden_state = torch.from_numpy(liquid_states['liquid_v']).to(self.device)
            self.liquid_r.hidden_state = torch.from_numpy(liquid_states['liquid_r']).to(self.device)
            
            # Restore multi-channel state
            multichannel_state = state['multichannel_state']
            self.multichannel_attention.channel_weights.data = torch.from_numpy(multichannel_state['channel_weights']).to(self.device)
            
            # Restore bandit states
            bandit_states = state['bandit_states']
            for i, (router, bandit_state) in enumerate(zip(self.neuron_routers, bandit_states)):
                if bandit_state['q_values'] is not None:
                    if router.bandit_stats is None:
                        router.bandit_stats = BanditStats(self.num_neurons_per_head, self.device)
                    router.bandit_stats.q_values = torch.from_numpy(bandit_state['q_values']).to(self.device)
                    router.bandit_stats.counts = torch.from_numpy(bandit_state['counts']).to(self.device)
                    router.bandit_stats.total_steps = bandit_state['total_steps']
            
            # Restore model parameters
            self.load_state_dict(state['model_state_dict'])
            
            return True
        except Exception as e:
            self.logger.error(f"SpikingSRWKV state setting failed: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        try:
            test_input = torch.randn(2, 10, self.embedding_dim, device=self.device)
            test_tokens = torch.randint(0, self.vocab_size, (2, 10), device=self.device)
            test_text = "This is a comprehensive test sentence with punctuation!"
            
            test_output = self.process_spikes(test_input, test_tokens, test_text)
            
            if test_output.shape != test_input.shape:
                return False, f"Output shape mismatch: {test_output.shape} vs {test_input.shape}"
            
            if torch.any(torch.isnan(test_output)) or torch.any(torch.isinf(test_output)):
                return False, "Output contains NaN or Inf values"
            
            # Validate comprehensive functionality
            stats = self.get_spike_statistics()
            required_stats = ['total_macs', 'bandit_rewards_ema']
            for stat in required_stats:
                if stat not in stats:
                    return False, f"Missing required statistic: {stat}"
            
            return True, "SpikingSRWKV validation successful"
        except Exception as e:
            return False, f"SpikingSRWKV validation error: {str(e)}"