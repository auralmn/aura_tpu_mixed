#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
from aura.utils.rope_jax import apply_rope, build_rope_cache


class SpikingLanguageCore(nn.Module):
    """
    Spiking language core for self-teaching LLM.
    Processes context vectors with biologically plausible spiking dynamics.
    """
    hidden_dim: int
    backend: str = 'lif'  # 'lif' or 'srwkv'
    use_rope: bool = False
    rope_base: float = 10000.0
    dt: float = 1e-3  # Time step
    T: int = 20  # Number of time steps for temporal simulation
    tau: float = 20e-3  # Membrane time constant
    v_th: float = 1.0  # Spike threshold
    v_reset: float = 0.0  # Reset voltage
    
    def setup(self):
        # Input projection to match dimensions
        self.input_projection = nn.Dense(self.hidden_dim)
        # Backend-specific parameters
        if self.backend == 'lif':
            self.recurrent_weights = self.param('recurrent_weights', 
                                             nn.initializers.normal(stddev=0.1),
                                             (self.hidden_dim, self.hidden_dim))
        elif self.backend == 'srwkv':
            # Simple SRWKV-like gating with per-channel time constants
            self.gate_r = nn.Dense(self.hidden_dim)
            self.time_param = self.param('time_param', nn.initializers.normal(stddev=0.1), (self.hidden_dim,))
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def __call__(self, input_state: jnp.ndarray, prev_state: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Process input through spiking language core.
        
        Args:
            input_state: Input context vector [batch, hidden_dim]
            prev_state: Previous state tuple (voltage, spike) [batch, hidden_dim] each
            
        Returns:
            Tuple of (output_rate, next_state)
        """
        prev_voltage, prev_spike = prev_state
        
        # Project input to match hidden dimensions
        projected_input = self.input_projection(input_state)  # [batch, hidden_dim]
        # Prepare RoPE caches if enabled (applied per internal time step)
        if self.use_rope:
            dim_even = projected_input.shape[-1] - (projected_input.shape[-1] % 2)
            if dim_even > 0:
                cos_cache, sin_cache = build_rope_cache(self.T, dim_even, base=self.rope_base)
        
        if self.backend == 'lif':
            # Initialize membrane potentials
            v = prev_voltage  # [batch, hidden_dim]
            s = prev_spike    # [batch, hidden_dim]
            spike_accum = jnp.zeros_like(s)
            for t in range(self.T):
                recurrent_input = jnp.dot(s, self.recurrent_weights)
                inp_t = projected_input
                if self.use_rope:
                    dim_even = projected_input.shape[-1] - (projected_input.shape[-1] % 2)
                    if dim_even > 0:
                        rot_even = apply_rope(projected_input[..., :dim_even], cos_cache[t], sin_cache[t])
                        inp_t = jnp.concatenate([rot_even, projected_input[..., dim_even:]], axis=-1)
                dv = (-v + recurrent_input + inp_t) / self.tau * self.dt
                v = v + dv
                spike = (v >= self.v_th).astype(jnp.float32)
                v = v * (1 - spike) + self.v_reset * spike
                spike_accum = spike_accum + spike
                s = spike
            avg_spikes = spike_accum / self.T
            next_state = (v, s)
            return avg_spikes, next_state
        else:  # srwkv
            # Use prev_voltage as running state; prev_spike as last output
            state = prev_voltage
            last_out = prev_spike
            out_accum = jnp.zeros_like(last_out)
            # Per-channel decay in (0,1)
            alpha = jax.nn.sigmoid(self.time_param)  # [hidden]
            for t in range(self.T):
                inp_t = projected_input
                if self.use_rope:
                    dim_even = projected_input.shape[-1] - (projected_input.shape[-1] % 2)
                    if dim_even > 0:
                        rot_even = apply_rope(projected_input[..., :dim_even], cos_cache[t], sin_cache[t])
                        inp_t = jnp.concatenate([rot_even, projected_input[..., dim_even:]], axis=-1)
                r = jax.nn.sigmoid(self.gate_r(state + inp_t))  # [batch, hidden]
                state = alpha * state + (1.0 - alpha) * inp_t
                out_t = r * state + (1.0 - r) * inp_t
                out_accum = out_accum + out_t
                last_out = out_t
            avg_out = out_accum / self.T
            next_state = (state, last_out)
            return avg_out, next_state
    
    def initialize_state(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize recurrent state for spiking core.
        
        Args:
            batch_size: Batch size for state initialization
            
        Returns:
            Tuple of (voltage, spike) initialized to zeros
        """
        voltage = jnp.zeros((batch_size, self.hidden_dim))
        spike = jnp.zeros((batch_size, self.hidden_dim))
        return (voltage, spike)
