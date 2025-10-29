# SPDX-License-Identifier: Apache-2.0
"""
AURA Spike Response Model (SRM) Neuron - FIXED VERSION
Temporal kernel-based spiking neuron with STDP learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from aura.neural.interfaces import NeuronModel

@dataclass
class SRMConfig:
    """Configuration for SRM neuron model"""
    # Membrane dynamics
    tau_m: float = 20.0         # Membrane time constant (ms)
    tau_s: float = 5.0          # Synaptic time constant (ms)
    tau_ref: float = 2.0        # Refractory time constant (ms)
    
    # Spiking parameters
    V_threshold: float = 1.0    # Spike threshold
    V_reset: float = 0.0        # Reset potential
    eta_0: float = -10.0        # Refractory amplitude
    
    # Temporal parameters
    dt: float = 0.1             # Time step (ms)
    kernel_length: int = 50     # Length of temporal kernels
    
    # STDP parameters
    enable_stdp: bool = True    # Enable STDP learning
    A_plus: float = 0.01        # LTP amplitude
    A_minus: float = -0.008     # LTD amplitude
    tau_plus: float = 20.0      # LTP time constant
    tau_minus: float = 20.0     # LTD time constant

class SRMNeuron(NeuronModel):
    """
    Spike Response Model neuron with temporal kernels
    - PSP (post-synaptic potential) kernels
    - Refractory kernels  
    - STDP plasticity
    - History-dependent dynamics
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        
        # SRM parameters from config
        srm_config = config.get('srm_config', {})
        self.srm = SRMConfig(**srm_config)
        
        # Network dimensions
        self.input_size = config.get('input_size', 128)
        self.hidden_size = config.get('hidden_size', 256)
        self.output_size = config.get('output_size', 128)
        
        # Synaptic weights
        self.W_in = nn.Parameter(torch.randn(self.input_size, self.hidden_size) * 0.02)
        self.W_rec = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_out = nn.Parameter(torch.randn(self.hidden_size, self.output_size) * 0.02)
        
        # Learnable neuron parameters
        self.tau_m_adaptive = nn.Parameter(torch.full((self.hidden_size,), self.srm.tau_m))
        self.tau_s_adaptive = nn.Parameter(torch.full((self.hidden_size,), self.srm.tau_s))
        self.threshold_adaptive = nn.Parameter(torch.full((self.hidden_size,), self.srm.V_threshold))
        
        # Build temporal kernels
        self.register_buffer('eta_kernel', self._build_refractory_kernel())
        self.register_buffer('epsilon_kernel', self._build_psp_kernel())
        
        # State variables - use register_buffer for proper device handling
        self.register_buffer('spike_history', 
                           torch.zeros(1, self.hidden_size, self.srm.kernel_length))
        self.register_buffer('input_history', 
                           torch.zeros(1, self.input_size, self.srm.kernel_length))
        self.register_buffer('membrane_potential', 
                           torch.zeros(1, self.hidden_size))
        
        # STDP traces
        if self.srm.enable_stdp:
            self.register_buffer('pre_trace', torch.zeros(1, self.input_size))
            self.register_buffer('post_trace', torch.zeros(1, self.hidden_size))
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def _build_refractory_kernel(self) -> torch.Tensor:
        """Build refractory kernel eta(s)"""
        s = torch.arange(0, self.srm.kernel_length * self.srm.dt, self.srm.dt)
        eta = self.srm.eta_0 * torch.exp(-s / self.srm.tau_ref)
        return eta
    
    def _build_psp_kernel(self) -> torch.Tensor:
        """Build post-synaptic potential kernel epsilon(s)"""
        s = torch.arange(0, self.srm.kernel_length * self.srm.dt, self.srm.dt)
        epsilon = (torch.exp(-s / self.srm.tau_m) - torch.exp(-s / self.srm.tau_s))
        # Normalize
        epsilon = epsilon / torch.max(epsilon)
        return epsilon
    
    def initialize(self) -> bool:
        """Initialize SRM neuron model"""
        try:
            # Initialize weights
            nn.init.xavier_uniform_(self.W_in, gain=0.5)
            nn.init.xavier_uniform_(self.W_rec, gain=0.3)
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            
            self.to(self.device)
            return True
        except Exception as e:
            self.logger.error(f"SRM neuron initialization failed: {e}")
            return False
    
    def reset_state(self, batch_size: int = 1):
        """Reset neuron states"""
        self.spike_history = torch.zeros(batch_size, self.hidden_size, 
                                       self.srm.kernel_length, device=self.device)
        self.input_history = torch.zeros(batch_size, self.input_size, 
                                       self.srm.kernel_length, device=self.device)
        self.membrane_potential = torch.zeros(batch_size, self.hidden_size, device=self.device)
        
        if self.srm.enable_stdp:
            self.pre_trace = torch.zeros(batch_size, self.input_size, device=self.device)
            self.post_trace = torch.zeros(batch_size, self.hidden_size, device=self.device)
    
    def compute_membrane_potential(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Compute membrane potential using temporal kernels"""
        batch_size = input_spikes.size(0)
        
        # Shift histories
        self.input_history = torch.roll(self.input_history, 1, dims=-1)
        self.input_history[:, :, 0] = input_spikes
        
        self.spike_history = torch.roll(self.spike_history, 1, dims=-1)
        # spike_history[:, :, 0] will be set after spike generation
        
        # Synaptic input: convolve input history with PSP kernel
        # Use adaptive time constants
        epsilon_kernel_adaptive = self._build_adaptive_psp_kernel()
        input_contrib = torch.sum(self.input_history * epsilon_kernel_adaptive.unsqueeze(0).unsqueeze(0), dim=-1)
        
        # Apply input weights with gradient flow
        synaptic_input = torch.matmul(input_contrib, self.W_in)
        
        # Refractory contribution: convolve spike history with refractory kernel
        refractory_contrib = torch.sum(self.spike_history * self.eta_kernel.unsqueeze(0).unsqueeze(0), dim=-1)
        
        # Recurrent input from previous spikes
        recent_spikes = self.spike_history[:, :, 1] if self.spike_history.size(-1) > 1 else torch.zeros_like(self.spike_history[:, :, 0])
        recurrent_input = torch.matmul(recent_spikes, self.W_rec)
        
        # Total membrane potential (use adaptive threshold for gradient flow)
        membrane_potential = synaptic_input + recurrent_input + refractory_contrib
        
        # Apply adaptive threshold scaling
        membrane_potential = membrane_potential * (self.threshold_adaptive.unsqueeze(0) / self.srm.V_threshold)
        
        self.membrane_potential = membrane_potential
        return membrane_potential
    
    def _build_adaptive_psp_kernel(self) -> torch.Tensor:
        """Build PSP kernel with adaptive time constants"""
        s = torch.arange(0, self.srm.kernel_length * self.srm.dt, self.srm.dt, device=self.device)
        
        # Use mean of adaptive time constants for kernel
        tau_m_mean = torch.mean(self.tau_m_adaptive)
        tau_s_mean = torch.mean(self.tau_s_adaptive)
        
        epsilon = (torch.exp(-s / tau_m_mean) - torch.exp(-s / tau_s_mean))
        epsilon = epsilon / torch.max(epsilon) if torch.max(epsilon) > 0 else epsilon
        return epsilon
    
    def generate_spikes(self, membrane_potential: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Generate spikes based on membrane potential"""
        # Use adaptive threshold
        threshold = self.threshold_adaptive.unsqueeze(0).expand_as(membrane_potential)
        
        if self.training and not deterministic:
            # Soft spiking with surrogate gradient
            spike_prob = torch.sigmoid((membrane_potential - threshold) * 5.0)
            spikes = torch.bernoulli(spike_prob)
            
            # Add surrogate gradient for backprop
            surrogate_grad = torch.clamp(1 - torch.abs(membrane_potential - threshold), 0, 1)
            spikes = spikes + surrogate_grad - surrogate_grad.detach()
        else:
            # Hard threshold for testing
            spikes = (membrane_potential > threshold).float()
        
        return spikes
    
    def update_stdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update STDP traces and weights"""
        if not self.srm.enable_stdp:
            return
        
        # Update traces
        self.pre_trace = self.pre_trace * (1 - self.srm.dt / self.srm.tau_minus) + pre_spikes
        self.post_trace = self.post_trace * (1 - self.srm.dt / self.srm.tau_plus) + post_spikes
        
        # Weight updates
        # LTP: pre before post
        ltp_update = torch.outer(self.pre_trace.squeeze(0), post_spikes.squeeze(0)) * self.srm.A_plus
        
        # LTD: post before pre  
        ltd_update = torch.outer(pre_spikes.squeeze(0), self.post_trace.squeeze(0)) * self.srm.A_minus
        
        # Apply updates
        with torch.no_grad():
            self.W_in.data += self.srm.dt * (ltp_update + ltd_update).t()
            # Clip weights
            self.W_in.data = torch.clamp(self.W_in.data, -1.0, 1.0)
    
    def forward(self, input_spikes: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass through SRM neuron layer
        
        Args:
            input_spikes: (batch, seq_len, input_size) input spike trains
            deterministic: If True, use deterministic spiking for testing
            
        Returns:
            output_spikes: (batch, seq_len, output_size) output spike trains
        """
        batch_size, seq_len, _ = input_spikes.shape
        input_spikes = input_spikes.to(self.device)
        
        # Reset states for new batch size
        if self.spike_history.size(0) != batch_size:
            self.reset_state(batch_size)
        
        output_spikes = []
        
        for t in range(seq_len):
            # Current input
            x_t = input_spikes[:, t, :]  # (batch, input_size)
            
            # Compute membrane potential
            membrane_potential = self.compute_membrane_potential(x_t)
            
            # Generate spikes
            spikes = self.generate_spikes(membrane_potential, deterministic)  # (batch, hidden_size)
            
            # Update spike history
            self.spike_history[:, :, 0] = spikes
            
            # STDP update
            if self.srm.enable_stdp and self.training:
                self.update_stdp(x_t, spikes)
            
            # Generate output through output layer
            output_current = torch.matmul(spikes, self.W_out)
            output_spike_prob = torch.sigmoid(output_current)
            
            if self.training and not deterministic:
                # Stochastic output
                output_spikes_t = torch.bernoulli(output_spike_prob)
                # Add surrogate gradient
                surrogate_grad = torch.clamp(1 - torch.abs(output_current), 0, 1)
                output_spikes_t = output_spikes_t + surrogate_grad - surrogate_grad.detach()
            else:
                # Deterministic output
                output_spikes_t = (output_spike_prob > 0.5).float()
            
            output_spikes.append(output_spikes_t)
        
        # Stack output spikes
        output_sequence = torch.stack(output_spikes, dim=1)  # (batch, seq_len, output_size)
        
        return output_sequence
    
    def get_neuron_statistics(self) -> Dict[str, float]:
        """Get comprehensive neuron statistics"""
        stats = {
            'avg_membrane_potential': float(torch.mean(self.membrane_potential)),
            'membrane_potential_std': float(torch.std(self.membrane_potential)),
            'total_spikes': float(torch.sum(self.spike_history)),
            'avg_spike_rate': float(torch.mean(torch.sum(self.spike_history, dim=-1))),
            'adaptive_tau_m_mean': float(torch.mean(self.tau_m_adaptive)),
            'adaptive_tau_s_mean': float(torch.mean(self.tau_s_adaptive)),
            'adaptive_threshold_mean': float(torch.mean(self.threshold_adaptive))
        }
        
        if self.srm.enable_stdp:
            stats.update({
                'avg_pre_trace': float(torch.mean(self.pre_trace)),
                'avg_post_trace': float(torch.mean(self.post_trace)),
                'weight_norm_in': float(torch.norm(self.W_in)),
                'weight_norm_rec': float(torch.norm(self.W_rec))
            })
        
        return stats
    
    # AURAModule interface methods
    def process(self, input_data: Any) -> Any:
        """Process input through SRM neurons"""
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data, deterministic=True)  # Deterministic for consistency
        elif isinstance(input_data, dict):
            spikes = input_data.get('spikes')
            if spikes is not None:
                return self.forward(spikes, deterministic=True)
        raise ValueError("SRM neuron requires spike tensor input")
    
    def get_state(self) -> Dict[str, Any]:
        """Get neuron state for hot-swapping"""
        state = {
            'spike_history': self.spike_history.detach().cpu().numpy(),
            'input_history': self.input_history.detach().cpu().numpy(),
            'membrane_potential': self.membrane_potential.detach().cpu().numpy(),
            'model_state_dict': self.state_dict(),
            'srm_config': {
                'tau_m': self.srm.tau_m,
                'tau_s': self.srm.tau_s,
                'tau_ref': self.srm.tau_ref,
                'V_threshold': self.srm.V_threshold,
                'V_reset': self.srm.V_reset,
                'eta_0': self.srm.eta_0,
                'dt': self.srm.dt,
                'kernel_length': self.srm.kernel_length,
                'enable_stdp': self.srm.enable_stdp,
                'A_plus': self.srm.A_plus,
                'A_minus': self.srm.A_minus
            },
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size
            }
        }
        
        if self.srm.enable_stdp:
            state.update({
                'pre_trace': self.pre_trace.detach().cpu().numpy(),
                'post_trace': self.post_trace.detach().cpu().numpy()
            })
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set neuron state during hot-swapping"""
        try:
            # Restore model parameters
            self.load_state_dict(state['model_state_dict'])
            
            # Restore neuron states
            self.spike_history = torch.from_numpy(state['spike_history']).to(self.device)
            self.input_history = torch.from_numpy(state['input_history']).to(self.device)
            self.membrane_potential = torch.from_numpy(state['membrane_potential']).to(self.device)
            
            if self.srm.enable_stdp and 'pre_trace' in state:
                self.pre_trace = torch.from_numpy(state['pre_trace']).to(self.device)
                self.post_trace = torch.from_numpy(state['post_trace']).to(self.device)
            
            # Restore configuration
            config = state['config']
            self.input_size = config['input_size']
            self.hidden_size = config['hidden_size']
            self.output_size = config['output_size']
            
            # Restore SRM parameters
            srm_config = state['srm_config']
            for key, value in srm_config.items():
                setattr(self.srm, key, value)
            
            return True
        except Exception as e:
            self.logger.error(f"SRM neuron state setting failed: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate SRM neuron functionality"""
        try:
            test_spikes = torch.randint(0, 2, (2, 10, self.input_size), 
                                      dtype=torch.float32, device=self.device)
            output_spikes = self.forward(test_spikes, deterministic=True)
            
            expected_shape = (2, 10, self.output_size)
            if output_spikes.shape != expected_shape:
                return False, f"Output shape mismatch: {output_spikes.shape} vs {expected_shape}"
            
            if torch.any(torch.isnan(output_spikes)) or torch.any(torch.isinf(output_spikes)):
                return False, "Output contains NaN or Inf values"
            
            # Check if spikes are reasonable
            unique_vals = torch.unique(output_spikes)
            if len(unique_vals) > 10:  # Allow some gradient values
                return False, f"Too many unique spike values: {len(unique_vals)}"
            
            return True, "SRM neuron validation successful"
        except Exception as e:
            return False, f"SRM neuron validation error: {str(e)}"
