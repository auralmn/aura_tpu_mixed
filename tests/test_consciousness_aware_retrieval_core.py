import jax
import jax.numpy as jnp
from aura.bio_inspired.consciousness_aware import (
    ProsodyExtractorJAX,
    PlutchikEmotionEncoderJAX,
    ConsciousnessAwareRetrievalCore,
)


def test_consciousness_aware_retrieval_end_to_end():
    key = jax.random.PRNGKey(2)
    batch, expert_dim = 2, 32
    seq_len, embed_dim = 12, 24
    pos_dim, syn_dim = 8, 6

    query_embedding = jax.random.normal(key, (batch, expert_dim))
    token_embeddings = jax.random.normal(key, (batch, seq_len, embed_dim))
    pos_tags = jax.random.normal(key, (batch, seq_len, pos_dim))
    syntax_features = jax.random.normal(key, (batch, seq_len, syn_dim))
    personality_traits = jax.random.normal(key, (batch, 5))

    # Build prosody and emotion features
    prosody_model = ProsodyExtractorJAX(hidden_dim=32, phasor_harmonics=8)
    prosody_params = prosody_model.init(key, token_embeddings, pos_tags, syntax_features)
    prosody = prosody_model.apply(prosody_params, token_embeddings, pos_tags, syntax_features)

    emotion_model = PlutchikEmotionEncoderJAX(emotion_dim=8, hidden_dim=32)
    emotion_params = emotion_model.init(key, query_embedding, personality_traits, prosody)
    emotions = emotion_model.apply(emotion_params, query_embedding, personality_traits, prosody)

    # Retrieval core
    core = ConsciousnessAwareRetrievalCore(num_experts=3, hidden_dim=64, expert_dim=expert_dim)
    core_params = core.init(
        key,
        query_embedding,
        text_tokens=token_embeddings,
        pos_tags=pos_tags,
        syntax_features=syntax_features,
        personality_traits=personality_traits,
        merit_bias=None,
        thalamic_bias=None,
        temperature=None,
        inactive_mask=None,
        top_k_route=2,
    )
    ctx, aux = core.apply(
        core_params,
        query_embedding,
        text_tokens=token_embeddings,
        pos_tags=pos_tags,
        syntax_features=syntax_features,
        personality_traits=personality_traits,
        top_k_route=2,
    )

    assert ctx.shape == (batch, 64)
    assert set(aux.keys()) >= {"emotions", "prosody_pitch", "prosody_energy", "gate_weights"}
    assert aux["emotions"].shape == (batch, 8)
    assert aux["gate_weights"].shape == (batch, 3)
