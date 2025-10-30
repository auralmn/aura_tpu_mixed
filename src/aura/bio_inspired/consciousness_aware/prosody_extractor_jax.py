from typing import Dict
import jax
import jax.numpy as jnp
from flax import linen as nn

from aura.bio_inspired.phasor_bank import PhasorBankJAX


class ProsodyExtractorJAX(nn.Module):
    """Extract prosodic features from text embeddings without audio.

    Outputs a compact dict with pitch/energy proxies, per-token duration, multi-scale rhythm, and pause indicators.
    """
    hidden_dim: int = 64
    phasor_harmonics: int = 32

    def setup(self):
        # Fixed phasor bank; input value carries variability
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=self.phasor_harmonics)

    @nn.compact
    def __call__(
        self,
        token_embeddings: jnp.ndarray,  # [batch, seq_len, embed_dim]
        pos_tags: jnp.ndarray,          # [batch, seq_len, pos_dim]
        syntax_features: jnp.ndarray    # [batch, seq_len, syntax_dim]
    ) -> Dict[str, jnp.ndarray]:
        # 1) Rhythm features from text structure (simple proxies)
        word_lengths = jnp.sum(jnp.abs(token_embeddings), axis=-1)  # [batch, seq_len]
        pause_indicators = self._detect_boundaries(syntax_features)  # [batch, seq_len]
        stress_patterns = self._predict_stress(pos_tags)             # [batch, seq_len]

        # 2) Temporal rhythm via PhasorBankJAX at multi-scales
        # Use batch-wise mean of word length as a simple sequence-level rhythm signal
        rhythm_signal = jnp.mean(word_lengths, axis=-1)  # [batch]

        # Vectorize phasor bank over batch using input signal
        temporal_rhythm = jax.vmap(lambda u: self.phasor_bank(u))(rhythm_signal)  # [batch, phasor_features]

        # 3) Duration prediction (lightweight, CART-inspired proxy)
        duration_feats = jnp.concatenate([
            word_lengths[..., None],                # [b, t, 1]
            pause_indicators[..., None],            # [b, t, 1]
            stress_patterns[..., None],            # [b, t, 1]
        ], axis=-1)                                 # [b, t, 3]
        duration_pred = nn.Dense(1)(duration_feats)  # [b, t, 1]

        # 4) Sequence-level prosody proxies
        pitch_proxy = nn.Dense(self.hidden_dim)(
            jnp.concatenate([
                jnp.mean(temporal_rhythm, axis=-1, keepdims=True),  # [b, 1]
                jnp.mean(stress_patterns, axis=-1, keepdims=True),  # [b, 1]
            ], axis=-1)
        )  # [b, hidden_dim]

        energy_proxy = nn.Dense(self.hidden_dim)(
            jnp.concatenate([
                jnp.std(word_lengths, axis=-1, keepdims=True),      # [b, 1]
                jnp.sum(pause_indicators, axis=-1, keepdims=True),  # [b, 1]
            ], axis=-1)
        )  # [b, hidden_dim]

        return {
            'pitch': pitch_proxy,           # [batch, hidden_dim]
            'energy': energy_proxy,         # [batch, hidden_dim]
            'duration': duration_pred,      # [batch, seq_len, 1]
            'rhythm': temporal_rhythm,      # [batch, phasor_features]
            'pauses': pause_indicators,     # [batch, seq_len]
        }

    def _detect_boundaries(self, syntax_features: jnp.ndarray) -> jnp.ndarray:
        """Predict phrase boundaries from syntax features (commas/clauses proxies)."""
        logits = nn.Dense(1)(syntax_features)             # [b, t, 1]
        return nn.sigmoid(logits).squeeze(-1)             # [b, t]

    def _predict_stress(self, pos_tags: jnp.ndarray) -> jnp.ndarray:
        """Predict lexical stress from POS tags (content vs function words)."""
        logits = nn.Dense(1)(pos_tags)                    # [b, t, 1]
        return nn.sigmoid(logits).squeeze(-1)             # [b, t]
