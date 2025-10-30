import jax
import jax.numpy as jnp
from aura.bio_inspired.consciousness_aware import PlutchikEmotionEncoderJAX


def test_plutchik_emotion_encoder_probabilities():
    key = jax.random.PRNGKey(1)
    batch, embed_dim = 3, 48
    text_embedding = jax.random.normal(key, (batch, embed_dim))
    personality_traits = jnp.clip(jax.random.normal(key, (batch, 5)), -2.0, 2.0)

    # Minimal prosody features
    prosody = {
        'pitch': jax.random.normal(key, (batch, 64)),
        'energy': jax.random.normal(key, (batch, 64)),
        'rhythm': jax.random.normal(key, (batch, 65)),
    }

    model = PlutchikEmotionEncoderJAX(emotion_dim=8, hidden_dim=32)
    params = model.init(key, text_embedding, personality_traits, prosody)
    probs = model.apply(params, text_embedding, personality_traits, prosody)

    assert probs.shape == (batch, 8)
    row_sums = jnp.sum(probs, axis=-1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)
    assert jnp.all((probs >= 0.0) & (probs <= 1.0))
