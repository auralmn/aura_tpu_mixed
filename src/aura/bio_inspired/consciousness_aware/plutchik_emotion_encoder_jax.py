from typing import Dict
import jax.numpy as jnp
from flax import linen as nn


class PlutchikEmotionEncoderJAX(nn.Module):
    """Map text + personality → Plutchik's 8 basic emotions (probabilities).

    Emotions (order): joy, trust, fear, surprise, sadness, disgust, anger, anticipation
    """
    emotion_dim: int = 8
    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self,
        text_embedding: jnp.ndarray,      # [batch, embed_dim]
        personality_traits: jnp.ndarray,  # [batch, 5]
        prosody_features: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        # 1) Semantic pathway
        semantic = nn.Dense(self.hidden_dim)(text_embedding)
        semantic = nn.gelu(semantic)

        # 2) Prosody pathway (pitch, energy, rhythm summary)
        rhythm_summary = jnp.mean(prosody_features['rhythm'], axis=-1, keepdims=True)  # [b, 1]
        prosody_concat = jnp.concatenate([
            prosody_features['pitch'],                 # [b, hidden]
            prosody_features['energy'],                # [b, hidden]
            rhythm_summary,                            # [b, 1]
        ], axis=-1)
        prosody_h = nn.Dense(self.hidden_dim)(prosody_concat)
        prosody_h = nn.gelu(prosody_h)

        # 3) Personality modulation
        trait_h = nn.Dense(self.hidden_dim)(personality_traits)
        trait_h = nn.gelu(trait_h)

        # 4) Fusion: semantic × prosody + trait bias
        fused = semantic * prosody_h + trait_h
        fused = nn.Dense(self.hidden_dim)(fused)
        fused = nn.gelu(fused)

        # 5) Emotion distribution
        emotion_logits = nn.Dense(self.emotion_dim)(fused)
        emotion_probs = nn.softmax(emotion_logits, axis=-1)
        return emotion_probs
