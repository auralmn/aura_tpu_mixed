import jax
import jax.numpy as jnp
from aura.bio_inspired.consciousness_aware import ProsodyExtractorJAX


def test_prosody_extractor_shapes():
    key = jax.random.PRNGKey(0)
    batch, seq_len, embed_dim = 4, 16, 32
    pos_dim, syn_dim = 12, 10

    token_embeddings = jax.random.normal(key, (batch, seq_len, embed_dim))
    pos_tags = jax.random.normal(key, (batch, seq_len, pos_dim))
    syntax_features = jax.random.normal(key, (batch, seq_len, syn_dim))

    model = ProsodyExtractorJAX(hidden_dim=64, phasor_harmonics=16)
    params = model.init(key, token_embeddings, pos_tags, syntax_features)
    out = model.apply(params, token_embeddings, pos_tags, syntax_features)

    assert set(out.keys()) == {"pitch", "energy", "duration", "rhythm", "pauses"}
    assert out["pitch"].shape == (batch, 64)
    assert out["energy"].shape == (batch, 64)
    assert out["duration"].shape == (batch, seq_len, 1)
    assert out["rhythm"].ndim == 2 and out["rhythm"].shape[0] == batch
    assert out["pauses"].shape == (batch, seq_len)
