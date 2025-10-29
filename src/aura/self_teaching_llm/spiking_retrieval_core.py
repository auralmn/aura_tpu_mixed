from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn


class SpikingRetrievalCore(nn.Module):
    """
    Liquid-MoE spiking retrieval core for self-teaching LLM.
    Retrieves context vectors from neuromorphic memory based on input queries.
    """
    hidden_dim: int
    num_experts: int
    expert_dim: int = 64
    T: int = 20
    poisson_encoding: bool = True
    dt: float = 1e-3
    tau: float = 20e-3
    v_th: float = 0.5
    v_reset: float = 0.0

    def setup(self):
        self.seed = 0;
        self.key = random.key(0)
        # Linear projection to expert_dim so downstream shapes are stable.
        self.in_proj = nn.Dense(self.expert_dim, use_bias=False, name="in_proj")

        # Experts: [num_experts, expert_dim, hidden_dim]
        self.experts = self.param(
            "experts",
            nn.initializers.normal(stddev=0.1),
            (self.num_experts, self.expert_dim, self.hidden_dim),
        )

        # Gating kernel: [expert_dim, num_experts]
        self.gate_kernel = self.param(
            "gate_kernel",
            nn.initializers.normal(stddev=0.1),
            (self.expert_dim, self.num_experts),
        )

    def __call__(
        self,
        query_embedding: jnp.ndarray,
        gate_bias: jnp.ndarray | None = None,
        temperature: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Args:
            query_embedding: [batch, embed_dim]
        Returns:
            context: [batch, hidden_dim]
        """
        batch = query_embedding.shape[0]
        dtype = query_embedding.dtype

        # 1) Project input to expert_dim once; use everywhere below.
        x = self.in_proj(query_embedding).astype(dtype)  # [batch, expert_dim]

        # 2) (Optional) Poisson encoding across time
        if self.poisson_encoding:
            
            
            # Normalize to [0,1] for rate -> spikes
            rates = jax.nn.sigmoid(x)  # [batch, expert_dim]
            _, key= random.split(self.key)
            input_spikes = jax.random.bernoulli(
                key, rates, shape=(self.T,) + rates.shape
            ).astype(dtype)  # [T, batch, expert_dim]
        else:
            input_spikes = jnp.repeat(x[None, ...], self.T, axis=0)  # [T, batch, expert_dim]

        # 3) Gating on the same projected features x
        gate_logits = x @ self.gate_kernel  # [batch, num_experts]
        if gate_bias is not None:
            gate_logits = gate_logits + jnp.asarray(gate_bias, dtype=gate_logits.dtype)[None, :]
        temp = jnp.asarray(1.0 if temperature is None else temperature, dtype=jnp.float32)
        gate_probs = jax.nn.softmax((gate_logits.astype(jnp.float32) / temp), axis=-1).astype(gate_logits.dtype)
        # shape: [batch, num_experts]

        # 4) LIF dynamics over time using scan (no Python loop)
        def expert_step(v_i, spikes_t, expert_matrix):
            # spikes_t: [batch, expert_dim], expert_matrix: [expert_dim, hidden_dim]
            current = spikes_t @ expert_matrix  # [batch, hidden_dim]
            dv = (current - v_i) / self.tau * self.dt
            new_v = v_i + dv
            spike = (new_v >= self.v_th).astype(dtype)
            new_v = new_v * (1.0 - spike) + self.v_reset * spike
            return new_v, spike  # both [batch, hidden_dim]

        def time_step(v_all, spikes_t):
            # v_all: [num_experts, batch, hidden_dim]
            # spikes_t: [batch, expert_dim]
            # map across experts dimension
            v_all, spike_t = jax.vmap(expert_step, in_axes=(0, None, 0))(v_all, spikes_t, self.experts)
            # spike_t: [num_experts, batch, hidden_dim]
            return v_all, spike_t

        v0 = jnp.zeros((self.num_experts, batch, self.hidden_dim), dtype=dtype)
        _, spikes_all = jax.lax.scan(time_step, v0, input_spikes)
        # spikes_all: [T, num_experts, batch, hidden_dim]

        # 5) Average over time -> [num_experts, batch, hidden_dim] -> [batch, num_experts, hidden_dim]
        avg_spikes = jnp.mean(spikes_all, axis=0).transpose(1, 0, 2)

        # 6) Combine with gate probabilities
        gate_probs_expanded = gate_probs[:, :, None]  # [batch, num_experts, 1]
        weighted_outputs = avg_spikes * gate_probs_expanded  # [batch, num_experts, hidden_dim]
        final_output = jnp.sum(weighted_outputs, axis=1)  # [batch, hidden_dim]

        return final_output

    def retrieve_context(
        self,
        memory_query: jnp.ndarray,
        gate_bias: jnp.ndarray | None = None,
        temperature: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # Delegates to __call__; keep API for backwards compat
        return self.__call__(memory_query, gate_bias=gate_bias, temperature=temperature)
