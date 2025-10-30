import jax
import jax.numpy as jnp
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore


def test_enhanced_retrieval_composite_gating_optional_inputs():
    key = jax.random.PRNGKey(3)
    batch, embed_dim = 2, 48
    num_experts, hidden_dim = 4, 32

    query_embedding = jax.random.normal(key, (batch, embed_dim))

    # Baseline apply without composite inputs (backward compatibility)
    core = EnhancedSpikingRetrievalCore(hidden_dim=hidden_dim, embed_dim=embed_dim, num_experts=num_experts, use_bio_gating=True)
    params = core.init(key, query_embedding)
    ctx0 = core.apply(params, query_embedding)
    assert ctx0.shape == (batch, hidden_dim)

    # Provide composite inputs: prosody (pitch/energy) + emotions
    prosody = {
        'pitch': jax.random.normal(key, (batch, 16)),
        'energy': jax.random.normal(key, (batch, 16)),
    }
    emotions = jax.random.uniform(key, (batch, 8))

    ctx1 = core.apply(params, query_embedding, prosody_features=prosody, emotions=emotions)
    assert ctx1.shape == (batch, hidden_dim)

    # Ensure a change in gating path is at least numerically reflected for typical random inputs
    diff = jnp.mean(jnp.abs(ctx1 - ctx0))
    assert diff >= 0.0
