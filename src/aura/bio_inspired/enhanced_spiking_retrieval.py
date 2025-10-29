#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced Spiking Retrieval Core with Phasor-Based Temporal Features
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

# Import bio-inspired components
from .phasor_bank import PhasorBankJAX
from .spiking_attention import SpikingAttentionJAX
from .experts import MLPExpert, Conv1DExpert, RationalExpert, CodeExpert, SelfImproveExpert


class EnhancedSpikingRetrievalCore(nn.Module):
    """
    Liquid-MoE spiking retrieval core with phasor-based temporal features.
    Retrieves context vectors from neuromorphic memory based on input queries.
    """
    hidden_dim: int = 768
    embed_dim: int = 768
    num_experts: int = 2
    expert_dim: int = 64
    T: int = 20  # Number of time steps for temporal simulation
    dt: float = 1e-3  # Time step
    tau: float = 20e-3  # Membrane time constant
    v_th: float = 0.5  # Spike threshold
    v_reset: float = 0.0  # Reset voltage
    phasor_harmonics: int = 192  # Number of harmonics for temporal features
    expert_types: tuple = ("mlp", "mlp")  # tuple of expert type strings
    freeze_experts: bool = False
    group_count: int = 1  # hierarchical gating groups
    top_k_route: int = 1  # number of experts to route to (soft mixture)
    predictive_weight: float = 0.0  # subtract predictor(head) * weight from gate logits
    use_bio_gating: bool = True  # if False, use simple features for gating (ablation)
    
    def setup(self):
        # Create experts based on expert_types; default to MLP
        expert_list = []
        for i in range(self.num_experts):
            t = self.expert_types[i] if i < len(self.expert_types) else "mlp"
            if t == "conv1d":
                expert_list.append(Conv1DExpert(hidden_dim=self.hidden_dim, name=f"expert_{i}"))
            elif t == "rational":
                expert_list.append(RationalExpert(hidden_dim=self.hidden_dim, name=f"expert_{i}"))
            elif t == "code":
                expert_list.append(CodeExpert(hidden_dim=self.hidden_dim, name=f"expert_{i}"))
            elif t == "self_improve":
                expert_list.append(SelfImproveExpert(hidden_dim=self.hidden_dim, name=f"expert_{i}"))
            else:
                expert_list.append(MLPExpert(hidden_dim=self.hidden_dim, bottleneck=max(32, self.hidden_dim // 4), name=f"expert_{i}"))
        self.experts = expert_list
        self.gate = nn.Dense(self.num_experts)
        # Optional predictive head to estimate per-expert loss or utility
        self.predictor = nn.Dense(self.num_experts)
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=self.phasor_harmonics)
        self.spiking_attention = SpikingAttentionJAX()
        # Hierarchical gating: assign experts to groups in round-robin if enabled
        if int(self.group_count) > 1:
            self.group_gate = nn.Dense(self.group_count)
            self.group_index = [i % self.group_count for i in range(self.num_experts)]
            # Precompute one-hot group map: [num_experts, group_count]
            gi = jnp.asarray(self.group_index, dtype=jnp.int32)
            ar = jnp.arange(self.num_experts, dtype=jnp.int32)
            gm = jnp.zeros((self.num_experts, self.group_count), dtype=jnp.float32)
            gm = gm.at[ar, gi].set(1.0)
            self.group_map = gm
        else:
            self.group_index = [0 for _ in range(self.num_experts)]
    
    def __call__(self, query_embedding: jnp.ndarray, active_experts: int = None, freeze_mask: jnp.ndarray = None, merit_bias: jnp.ndarray = None, temperature: jnp.ndarray = None, inactive_mask: jnp.ndarray = None, thalamic_bias: jnp.ndarray = None) -> jnp.ndarray:
        """
        Retrieve context with temporal feature enhancement.
        
        Args:
            query_embedding: Input query embedding [batch, embed_dim]
            
        Returns:
            Context vector [batch, hidden_dim]
        """
        x = self._normalize_in_dim(query_embedding)
        batch_size = x.shape[0]
        query_mean = jnp.mean(x, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(query_mean)
        K = min(32, x.shape[-1])
        topk_idx = jax.lax.top_k(jnp.abs(x), K)[1]
        vocab_size = int(query_embedding.shape[-1])
        attention_gains = jax.vmap(self.spiking_attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        if self.use_bio_gating:
            gate_inputs = jnp.stack([jnp.mean(temporal_features, axis=-1), jnp.mean(attention_gains, axis=-1)], axis=-1)
        else:
            q_mean = jnp.mean(query_embedding, axis=-1)
            q_std = jnp.std(query_embedding, axis=-1)
            gate_inputs = jnp.stack([q_mean, q_std], axis=-1)
        gate_logits = self.gate(gate_inputs)  # [batch, num_experts]
        if self.predictive_weight is not None and self.predictive_weight != 0.0:
            pred = self.predictor(gate_inputs)
            gate_logits = gate_logits - float(self.predictive_weight) * pred
        # Group bias over expert logits
        if self.group_count and self.group_count > 1:
            group_logits = self.group_gate(gate_inputs)  # [batch, group_count]
            # Bias experts by their group's logit via matmul: [batch, group_count] @ [group_count, num_experts]
            group_bias = jnp.matmul(group_logits, jnp.transpose(self.group_map))  # [batch, num_experts]
            gate_logits = gate_logits + group_bias
        if merit_bias is not None:
            mb = jnp.asarray(merit_bias)
            gate_logits = gate_logits + mb[None, :]
        if thalamic_bias is not None:
            tb = jnp.asarray(thalamic_bias)
            gate_logits = gate_logits + tb[None, :]
        # Limit to active experts if provided
        if active_experts is not None:
            # Create mask: 1 for active indices [0:active_experts), 0 otherwise
            ae = jnp.asarray(active_experts, dtype=jnp.int32)
            mask = jnp.arange(self.num_experts, dtype=jnp.int32)[None, :] < ae
            # Mask logits for inactive experts
            neg_inf = jnp.array(-1e9)
            gate_logits = jnp.where(mask, gate_logits, neg_inf)
        # Apply pruned/inactive expert mask if provided
        if inactive_mask is not None:
            fm = jnp.asarray(inactive_mask, dtype=bool)[None, :]
            neg_inf = jnp.array(-1e9)
            gate_logits = jnp.where(fm, neg_inf, gate_logits)
        # Apply temperature
        temp = jnp.asarray(1.0 if temperature is None else temperature)
        gate_weights = nn.softmax(gate_logits / temp, axis=-1)  # [batch, num_experts]
        # Optional top-k soft routing
        if (self.top_k_route is not None) and (int(self.top_k_route) > 1):
            k = int(min(self.top_k_route, self.num_experts))
            vals, idx = jax.lax.top_k(gate_weights, k)
            mask = jnp.zeros_like(gate_weights)
            # scatter 1.0 at top-k indices per batch
            ar = jnp.arange(gate_weights.shape[0])[:, None]
            mask = mask.at[ar, idx].set(1.0)
            gated = gate_weights * mask
            denom = jnp.sum(gated, axis=-1, keepdims=True) + 1e-9
            gate_weights = gated / denom
        # Compute expert outputs and combine
        expert_outs = [expert(x) for expert in self.experts]  # list of [batch, hidden_dim]
        expert_stack = jnp.stack(expert_outs, axis=1)  # [batch, num_experts, hidden_dim]
        # Apply freezing mask if provided, else use module-level freeze flag
        if freeze_mask is not None:
            fm = jnp.asarray(freeze_mask, dtype=bool)[None, :, None]  # [1, num_experts, 1]
            expert_stack = jnp.where(fm, jax.lax.stop_gradient(expert_stack), expert_stack)
        elif self.freeze_experts:
            expert_stack = jax.lax.stop_gradient(expert_stack)
        context_vector = jnp.einsum('bn,bnh->bh', gate_weights, expert_stack)  # [batch, hidden_dim]
        return context_vector

    def _normalize_in_dim(self, x: jnp.ndarray) -> jnp.ndarray:
        """Ensure last-dim == self.embed_dim by slice-or-pad (zero pad)."""
        d = x.shape[-1]
        if d == self.embed_dim:
            return x
        elif d > self.embed_dim:
            return x[..., :self.embed_dim]
        else:
            pad = self.embed_dim - d
            return jnp.pad(x, ((0, 0), (0, pad)), mode='constant')

    def compute_gate_weights(self, query_embedding: jnp.ndarray, active_experts: int = None, merit_bias: jnp.ndarray = None, temperature: jnp.ndarray = None, inactive_mask: jnp.ndarray = None, thalamic_bias: jnp.ndarray = None) -> jnp.ndarray:
        """Compute routing weights without producing a context vector."""
        query_mean = jnp.mean(query_embedding, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(query_mean)
        K = min(32, query_embedding.shape[-1])
        topk_idx = jax.lax.top_k(jnp.abs(query_embedding), K)[1]
        vocab_size = int(query_embedding.shape[-1])
        attention_gains = jax.vmap(self.spiking_attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        if self.use_bio_gating:
            gate_inputs = jnp.stack([jnp.mean(temporal_features, axis=-1), jnp.mean(attention_gains, axis=-1)], axis=-1)
        else:
            q_mean = jnp.mean(query_embedding, axis=-1)
            q_std = jnp.std(query_embedding, axis=-1)
            gate_inputs = jnp.stack([q_mean, q_std], axis=-1)
        gate_logits = self.gate(gate_inputs)
        if self.predictive_weight is not None and self.predictive_weight != 0.0:
            pred = self.predictor(gate_inputs)
            gate_logits = gate_logits - float(self.predictive_weight) * pred
        if self.group_count and self.group_count > 1:
            group_logits = self.group_gate(gate_inputs)  # [batch, group_count]
            group_bias = jnp.matmul(group_logits, jnp.transpose(self.group_map))  # [batch, num_experts]
            gate_logits = gate_logits + group_bias
        if merit_bias is not None:
            mb = jnp.asarray(merit_bias)
            gate_logits = gate_logits + mb[None, :]
        if active_experts is not None:
            ae = jnp.asarray(active_experts, dtype=jnp.int32)
            mask = jnp.arange(self.num_experts, dtype=jnp.int32)[None, :] < ae
            neg_inf = jnp.array(-1e9)
            gate_logits = jnp.where(mask, gate_logits, neg_inf)
        if inactive_mask is not None:
            fm = jnp.asarray(inactive_mask, dtype=bool)[None, :]
            neg_inf = jnp.array(-1e9)
            gate_logits = jnp.where(fm, neg_inf, gate_logits)
        temp = jnp.asarray(1.0 if temperature is None else temperature)
        gate_weights = nn.softmax(gate_logits / temp, axis=-1)
        if self.top_k_route is not None and self.top_k_route > 1:
            k = jnp.minimum(self.top_k_route, self.num_experts)
            vals, idx = jax.lax.top_k(gate_weights, k)
            mask = jnp.zeros_like(gate_weights)
            ar = jnp.arange(gate_weights.shape[0])[:, None]
            mask = mask.at[ar, idx].set(1.0)
            gated = gate_weights * mask
            denom = jnp.sum(gated, axis=-1, keepdims=True) + 1e-9
            gate_weights = gated / denom
        return gate_weights

    def expert_outputs(self, query_embedding: jnp.ndarray) -> jnp.ndarray:
        """Return stacked outputs of all experts for the given embedding.
        Shape: [batch, num_experts, hidden_dim]
        """
        outs = [expert(query_embedding) for expert in self.experts]
        return jnp.stack(outs, axis=1)

    def distill_loss(self, query_embedding: jnp.ndarray, teacher_idx: int, student_idx: int, teacher_weights: jnp.ndarray = None) -> jnp.ndarray:
        """Compute distillation loss between teacher and student expert outputs.
        Teacher and student are indices into the experts list.
        Returns MSE over the batch.
        """
        outs = self.expert_outputs(query_embedding)  # [batch, num_experts, hidden_dim]
        if teacher_weights is not None:
            # teacher_weights: [num_experts] -> compute weighted teacher target
            w = jnp.asarray(teacher_weights, dtype=outs.dtype)
            teacher_out = jnp.einsum('n,bnh->bh', w, outs)
        else:
            teacher_out = outs[:, teacher_idx, :]
        student_out = outs[:, student_idx, :]
        return jnp.mean((teacher_out - student_out) ** 2)


# Example usage and testing
if __name__ == "__main__":
    # This test won't work when run directly due to relative imports
    # but the module structure is correct for use within the package
    print("Enhanced spiking retrieval core module structure is correct")
    print("Run tests from the package root directory")
