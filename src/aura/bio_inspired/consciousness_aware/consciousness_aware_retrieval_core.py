from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

from aura.bio_inspired.spiking_attention import SpikingAttentionJAX
from aura.bio_inspired.phasor_bank import PhasorBankJAX
from .prosody_extractor_jax import ProsodyExtractorJAX
from .plutchik_emotion_encoder_jax import PlutchikEmotionEncoderJAX


class ConsciousnessAwareRetrievalCore(nn.Module):
    """Retrieval core with prosody + emotion + bio-inspired gating.

    Produces a context vector via expert mixture weighted by a composite gate that
    integrates temporal rhythm (phasor), spiking attention, prosody proxies, and Plutchik emotions.
    """
    num_experts: int = 4
    hidden_dim: int = 128
    expert_dim: int = 64
    v_th: float = 0.5

    use_prosody: bool = True
    use_emotion: bool = True

    @nn.compact
    def __call__(
        self,
        query_embedding: jnp.ndarray,                 # [batch, expert_dim]
        text_tokens: jnp.ndarray | None = None,       # [batch, seq_len, embed_dim]
        pos_tags: jnp.ndarray | None = None,          # [batch, seq_len, pos_dim]
        syntax_features: jnp.ndarray | None = None,   # [batch, seq_len, syntax_dim]
        personality_traits: jnp.ndarray | None = None,
        merit_bias: jnp.ndarray | None = None,
        thalamic_bias: jnp.ndarray | None = None,
        temperature: jnp.ndarray | None = None,
        inactive_mask: jnp.ndarray | None = None,
        top_k_route: int | None = None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        batch = query_embedding.shape[0]
        dtype = query_embedding.dtype

        # Normalize and basic stats
        x = self._normalize_in_dim(query_embedding)  # [b, expert_dim]
        q_mean = jnp.mean(x, axis=-1)                # [b]

        # Temporal rhythm via PhasorBankJAX
        # Use a fixed phasor bank and pass varying inputs to avoid tracer issues
        phasor_bank = PhasorBankJAX(delta0=7.0, H=32)
        def call_bank(u):
            return phasor_bank(u)
        temporal_features = jax.vmap(call_bank)(q_mean)  # [b, phasor_feats]

        # Spiking attention gains using top-k magnitude indices
        K = jnp.minimum(32, x.shape[-1])
        topk_vals, topk_idx = jax.lax.top_k(jnp.abs(x), int(K))
        vocab_size = int(query_embedding.shape[-1])
        spiking_attention = SpikingAttentionJAX()
        attention_gains = jax.vmap(spiking_attention, in_axes=(0, None))(
            topk_idx.astype(jnp.int32), vocab_size
        )  # [b, vocab_size]

        # Prosody features (text-only, optional)
        prosody: Dict[str, jnp.ndarray]
        if self.use_prosody and text_tokens is not None and pos_tags is not None and syntax_features is not None:
            prosody = ProsodyExtractorJAX()(text_tokens, pos_tags, syntax_features)
        else:
            prosody = {
                'pitch': jnp.zeros((batch, 64), dtype=dtype),
                'energy': jnp.zeros((batch, 64), dtype=dtype),
                'rhythm': jnp.zeros((batch, 65), dtype=dtype),
                'duration': jnp.zeros((batch, 1, 1), dtype=dtype),
                'pauses': jnp.zeros((batch, 1), dtype=dtype),
            }

        # Emotion features (optional)
        if self.use_emotion and personality_traits is not None:
            emotions = PlutchikEmotionEncoderJAX()(query_embedding, personality_traits, prosody)
        else:
            emotions = jnp.zeros((batch, 8), dtype=dtype)

        # Composite gate inputs
        gate_inputs = jnp.concatenate([
            jnp.mean(temporal_features, axis=-1, keepdims=True),
            jnp.mean(attention_gains, axis=-1, keepdims=True),
            jnp.mean(prosody['pitch'], axis=-1, keepdims=True),
            jnp.mean(prosody['energy'], axis=-1, keepdims=True),
            emotions,
        ], axis=-1)  # [b, ~12]

        # Gate head and biases
        gate_logits = nn.Dense(self.num_experts, use_bias=True)(gate_inputs)  # [b, n_exp]
        if merit_bias is not None:
            gate_logits = gate_logits + merit_bias[None, :]
        if thalamic_bias is not None:
            gate_logits = gate_logits + thalamic_bias[None, :]

        temp = temperature if temperature is not None else jnp.array(1.0, dtype=gate_logits.dtype)
        gate_weights = nn.softmax(gate_logits / temp, axis=-1)  # [b, n_exp]

        if top_k_route is not None and int(top_k_route) < self.num_experts:
            k = int(top_k_route)
            vals, idx = jax.lax.top_k(gate_weights, k)
            mask = jnp.zeros_like(gate_weights)
            ar = jnp.arange(gate_weights.shape[0])[:, None]
            mask = mask.at[ar, idx].set(1.0)
            gated = gate_weights * mask
            denom = jnp.sum(gated, axis=-1, keepdims=True) + 1e-9
            gate_weights = gated / denom

        if inactive_mask is not None:
            neg_inf = jnp.asarray(-1e9, dtype=gate_logits.dtype)
            mask_logits = jnp.where(inactive_mask[None, :], neg_inf, 0.0)
            gate_weights = nn.softmax((gate_logits + mask_logits) / temp, axis=-1)

        # Expert pool: simple linear experts for now (hidden_dim outputs)
        experts = [self.param(f"expert_{i}", nn.initializers.lecun_normal(), (self.expert_dim, self.hidden_dim))
                   for i in range(self.num_experts)]
        expert_outs = [x @ experts[i] for i in range(self.num_experts)]          # list of [b, hidden]
        expert_stack = jnp.stack(expert_outs, axis=1)                            # [b, n_exp, hidden]

        context_vector = jnp.sum(gate_weights[:, :, None] * expert_stack, axis=1)

        aux = {
            'emotions': emotions,
            'prosody_pitch': prosody['pitch'],
            'prosody_energy': prosody['energy'],
            'gate_weights': gate_weights,
        }
        return context_vector, aux

    def _normalize_in_dim(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True) + 1e-6
        return (x - mean) / std
