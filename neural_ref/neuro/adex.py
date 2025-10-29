# SPDX-License-Identifier: Apache-2.0
"""
AURA Adaptive Exponential Integrate-and-Fire (AdEx) Neuron Model - FIXED VERSION
Biologically realistic neuron with adaptation and exponential spike initiation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from aura.neural.interfaces import NeuronModel

@dataclass
class AdExConfig:
    """Configuration for AdEx neuron model"""
    # Membrane parameters
    C_m: float = 281.0          # Membrane capacitance (pF)
    g_L: float = 30.0           # Leak conductance (nS)  
    E_L: float = -70.6          # Leak reversal potential (mV)
    V_T: float = -50.4          # Threshold potential (mV)
    Delta_T: float = 2.0        # Slope factor (mV)
    
    # Adaptation parameters
    a: float = 4.0              # Subthreshold adaptation (nS)
    b: float = 80.5             # Spike-triggered adaptation (pA)
    tau_w: float = 144.0        # Adaptation time constant (ms)
    
    # Reset parameters
    V_reset: float = -70.6      # Reset potential (mV)
    V_spike: float = 20.0       # Spike cutoff (mV)
    
    # Integration
    dt: float = 0.1             # Time step (ms)

class AdExNeuron(NeuronModel):
    """
    Adaptive Exponential Integrate-and-Fire neuron model
    - Exponential spike initiation mechanism
    - Spike-frequency adaptation
    - Subthreshold adaptation currents
    - Biologically realistic dynamics
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        
        # AdEx parameters from config
        adex_config = config.get('adex_config', {})
        self.adex = AdExConfig(**adex_config)
        
        # Network dimensions
        self.input_size = config.get('input_size', 128)
        self.hidden_size = config.get('hidden_size', 256)
        self.output_size = config.get('output_size', 128)
        
        # Synaptic weights
        self.W_in = nn.Parameter(torch.randn(self.input_size, self.hidden_size) * 0.02)
        self.W_rec = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_out = nn.Parameter(torch.randn(self.hidden_size, self.output_size) * 0.02)
        
        # Learnable neuron parameters (can adapt during training)
        self.adaptive_threshold = nn.Parameter(torch.full((self.hidden_size,), self.adex.V_T))
        self.adaptation_strength = nn.Parameter(torch.full((self.hidden_size,), self.adex.a))
        
        # State variables
        self.register_buffer('V_m', torch.full((1, self.hidden_size), self.adex.E_L))  # Membrane potential
        self.register_buffer('w', torch.zeros(1, self.hidden_size))                    # Adaptation current
        self.register_buffer('spike_times', torch.zeros(1, self.hidden_size))         # Last spike time
        self.register_buffer('refractory_mask', torch.ones(1, self.hidden_size))     # Refractory period
        
        # Statistics
        self.register_buffer('spike_count', torch.zeros(1, self.hidden_size))
        self.register_buffer('total_spikes', torch.tensor(0.0))
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    def initialize(self) -> bool:
        """Initialize AdEx neuron model"""
        try:
            # Initialize weights
            nn.init.xavier_uniform_(self.W_in, gain=0.5)
            nn.init.xavier_uniform_(self.W_rec, gain=0.3)  # Smaller for stability
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            
            self.to(self.device)
            return True
        except Exception as e:
            self.logger.error(f"AdEx neuron initialization failed: {e}")
            return False
    
    def reset_state(self, batch_size: int = 1):
        """Reset neuron states"""
        self.V_m = torch.full((batch_size, self.hidden_size), self.adex.E_L, device=self.device)
        self.w = torch.zeros(batch_size, self.hidden_size, device=self.device)
        self.spike_times = torch.zeros(batch_size, self.hidden_size, device=self.device)
        self.refractory_mask = torch.ones(batch_size, self.hidden_size, device=self.device)
        self.spike_count = torch.zeros(batch_size, self.hidden_size, device=self.device)
    
    def exponential_term(self, V_m: torch.Tensor) -> torch.Tensor:
        """Compute exponential spike initiation term"""
        # Use adaptive threshold to ensure gradients flow
        return self.adex.g_L * self.adex.Delta_T * torch.exp(
            (V_m - self.adaptive_threshold.unsqueeze(0).expand_as(V_m)) / self.adex.Delta_T
        )
    
    def compute_membrane_dynamics(self, V_m: torch.Tensor, w: torch.Tensor, 
                                 I_syn: torch.Tensor) -> torch.Tensor:
        """Compute membrane potential dynamics"""
        # Leak current
        I_leak = self.adex.g_L * (self.adex.E_L - V_m)
        
        # Exponential spike initiation (uses adaptive_threshold parameter)
        I_exp = self.exponential_term(V_m)
        
        # Use adaptive adaptation strength parameter
        I_adapt = self.adaptation_strength.unsqueeze(0).expand_as(w) * w
        
        # Total membrane current
        I_total = I_leak + I_exp - I_adapt + I_syn
        
        # Membrane potential derivative
        dV_dt = I_total / self.adex.C_m
        
        return dV_dt
    
    def compute_adaptation_dynamics(self, V_m: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute adaptation current dynamics"""
        # Use adaptive adaptation strength parameter
        adaptation_term = self.adaptation_strength.unsqueeze(0).expand_as(V_m) * (V_m - self.adex.E_L)
        dw_dt = (adaptation_term - w) / self.adex.tau_w
        return dw_dt
    
    def detect_spikes(self, V_m: torch.Tensor) -> torch.Tensor:
        """Detect spike events"""
        return (V_m > self.adex.V_spike).float()
    
    def apply_reset(self, V_m: torch.Tensor, w: torch.Tensor, 
                   spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply post-spike reset"""
        # Reset membrane potential
        V_m_reset = torch.where(spikes.bool(), 
                               torch.full_like(V_m, self.adex.V_reset), 
                               V_m)
        
        # Increment adaptation current
        w_increment = w + spikes * self.adex.b
        
        return V_m_reset, w_increment
    
    def forward(self, input_spikes: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass through AdEx neuron layer
        
        Args:
            input_spikes: (batch, seq_len, input_size) input spike trains
            deterministic: If True, use deterministic spiking for testing
            
        Returns:
            output_spikes: (batch, seq_len, output_size) output spike trains
        """
        batch_size, seq_len, _ = input_spikes.shape
        input_spikes = input_spikes.to(self.device)
        
        # Reset states for new batch
        if self.V_m.size(0) != batch_size:
            self.reset_state(batch_size)
        
        output_spikes = []
        
        for t in range(seq_len):
            # Current input
            x_t = input_spikes[:, t, :]  # (batch, input_size)
            
            # Synaptic input
            I_syn_in = torch.matmul(x_t, self.W_in)  # (batch, hidden_size)
            
            # Recurrent input from previous spikes
            if t > 0:
                prev_spikes = output_spikes[-1]  # Previous output spikes
                I_syn_rec = torch.matmul(prev_spikes, self.W_rec.t())
            else:
                I_syn_rec = torch.zeros_like(I_syn_in)
            
            I_syn_total = I_syn_in + I_syn_rec
            
            # Compute dynamics
            dV_dt = self.compute_membrane_dynamics(self.V_m, self.w, I_syn_total)
            dw_dt = self.compute_adaptation_dynamics(self.V_m, self.w)
            
            # Euler integration
            V_m_new = self.V_m + self.adex.dt * dV_dt
            w_new = self.w + self.adex.dt * dw_dt
            
            # Detect spikes
            spikes = self.detect_spikes(V_m_new)  # (batch, hidden_size)
            
            # Apply reset
            V_m_reset, w_reset = self.apply_reset(V_m_new, w_new, spikes)
            
            # Update states
            self.V_m = V_m_reset
            self.w = w_reset
            
            # Update statistics
            self.spike_count += spikes
            self.total_spikes += torch.sum(spikes)
            
            # Generate output spikes through output layer
            output_current = torch.matmul(spikes, self.W_out)
            output_spike_prob = torch.sigmoid(output_current)
            
            # Generate output spikes (stochastic or deterministic)
            if self.training and not deterministic:
                # Stochastic spiking during training
                output_spikes_t = torch.bernoulli(output_spike_prob)
                # Add surrogate gradient
                surrogate_grad = torch.clamp(1 - torch.abs(output_current), 0, 1)
                output_spikes_t = output_spikes_t + surrogate_grad - surrogate_grad.detach()
            else:
                # Deterministic spiking during inference or testing
                output_spikes_t = (output_spike_prob > 0.5).float()
            
            output_spikes.append(output_spikes_t)
        
        # Stack output spikes
        output_sequence = torch.stack(output_spikes, dim=1)  # (batch, seq_len, output_size)
        
        return output_sequence
    
    def get_neuron_statistics(self) -> Dict[str, float]:
        """Get comprehensive neuron statistics"""
        if torch.sum(self.spike_count) == 0:
            return {
                'avg_spike_rate': 0.0,
                'total_spikes': 0.0,
                'avg_membrane_potential': float(torch.mean(self.V_m)),
                'avg_adaptation_current': float(torch.mean(self.w)),
                'neuron_utilization': 0.0
            }
        
        # Spike statistics
        total_neurons = self.hidden_size * self.V_m.size(0)
        active_neurons = torch.sum(self.spike_count > 0).float()
        
        return {
            'avg_spike_rate': float(torch.mean(self.spike_count)),
            'total_spikes': float(self.total_spikes),
            'avg_membrane_potential': float(torch.mean(self.V_m)),
            'avg_adaptation_current': float(torch.mean(self.w)),
            'membrane_potential_std': float(torch.std(self.V_m)),
            'adaptation_current_std': float(torch.std(self.w)),
            'neuron_utilization': float(active_neurons / total_neurons),
            'adaptive_threshold_mean': float(torch.mean(self.adaptive_threshold)),
            'adaptation_strength_mean': float(torch.mean(self.adaptation_strength))
        }
    
    # AURAModule interface methods
    def process(self, input_data: Any) -> Any:
        """Process input through AdEx neurons"""
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data, deterministic=True)  # Deterministic for consistency
        elif isinstance(input_data, dict):
            spikes = input_data.get('spikes')
            if spikes is not None:
                return self.forward(spikes, deterministic=True)
        raise ValueError("AdEx neuron requires spike tensor input")
    
    def get_state(self) -> Dict[str, Any]:
        """Get neuron state for hot-swapping"""
        return {
            'V_m': self.V_m.detach().cpu().numpy(),
            'w': self.w.detach().cpu().numpy(),
            'spike_times': self.spike_times.detach().cpu().numpy(),
            'spike_count': self.spike_count.detach().cpu().numpy(),
            'total_spikes': self.total_spikes.detach().cpu().numpy(),
            'model_state_dict': self.state_dict(),
            'adex_config': {
                'C_m': self.adex.C_m,
                'g_L': self.adex.g_L,
                'E_L': self.adex.E_L,
                'V_T': self.adex.V_T,
                'Delta_T': self.adex.Delta_T,
                'a': self.adex.a,
                'b': self.adex.b,
                'tau_w': self.adex.tau_w,
                'V_reset': self.adex.V_reset,
                'V_spike': self.adex.V_spike,
                'dt': self.adex.dt
            },
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set neuron state during hot-swapping"""
        try:
            # Restore model parameters first
            self.load_state_dict(state['model_state_dict'])
            
            # Restore neuron states
            self.V_m = torch.from_numpy(state['V_m']).to(self.device)
            self.w = torch.from_numpy(state['w']).to(self.device)
            self.spike_times = torch.from_numpy(state['spike_times']).to(self.device)
            self.spike_count = torch.from_numpy(state['spike_count']).to(self.device)
            self.total_spikes = torch.from_numpy(state['total_spikes']).to(self.device)
            
            # Restore configuration
            config = state['config']
            self.input_size = config['input_size']
            self.hidden_size = config['hidden_size']
            self.output_size = config['output_size']
            
            # Restore AdEx parameters
            adex_config = state['adex_config']
            for key, value in adex_config.items():
                setattr(self.adex, key, value)
            
            return True
        except Exception as e:
            self.logger.error(f"AdEx neuron state setting failed: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate AdEx neuron functionality"""
        try:
            test_spikes = torch.randint(0, 2, (2, 10, self.input_size), 
                                      dtype=torch.float32, device=self.device)
            output_spikes = self.forward(test_spikes, deterministic=True)
            
            expected_shape = (2, 10, self.output_size)
            if output_spikes.shape != expected_shape:
                return False, f"Output shape mismatch: {output_spikes.shape} vs {expected_shape}"
            
            if torch.any(torch.isnan(output_spikes)) or torch.any(torch.isinf(output_spikes)):
                return False, "Output contains NaN or Inf values"
            
            # Check if spikes are binary (approximately)
            unique_vals = torch.unique(output_spikes)
            if len(unique_vals) > 10:  # Allow some gradient values during training
                return False, f"Too many unique spike values: {len(unique_vals)}"
            
            return True, "AdEx neuron validation successful"
        except Exception as e:
            return False, f"AdEx neuron validation error: {str(e)}"
