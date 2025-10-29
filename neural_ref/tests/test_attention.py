import numpy as np
import trio
from    aura.core.nlms import NLMSHead, SpikingAttention

async def _run():
    D, C = 16, 1
    head = NLMSHead(
        n_features=D, n_outputs=C, enable_attention=True,
        attention_config={"decay":0.7,"theta":1.0,"k_winners":3,"gain_up":1.6,"gain_down":0.6},
        tok_slice=slice(1, 9), realm_slice=slice(9,13), phase_slice=slice(13,16),
        prosody_idx=0
    )
    await head.attach(
        base_w=np.zeros((D,C)),
        tok_slice=slice(1,9), realm_slice=slice(9,13), phase_slice=slice(13,16),
        pos_slice=None, ctop_idx=None, prosody_idx=0
    )
    x = np.random.randn(D).astype(np.float64)
    y0 = await head.step(x, 0.5)                 # plain
    assert isinstance(y0, np.ndarray) and y0.shape == (C,)

    # attention path (legacy single-channel)
    toks = [1,3,5,3,1,7,3,2]
    y1 = await head.step_with_attention(x, 0.5, token_sequence=toks)
    assert isinstance(y1, np.ndarray)

    # multi-channel path (per-feature gains)
    class Multi:
        def compute(self, **kw):
            L = kw["feature_size"]
            return {"mu_scalar": 1.2, "per_feature_gains": np.ones(L, dtype=np.float64)}
    y2 = await head.step_with_multi_channel_attention(
        x, 0.5, token_ids=toks,
        amp=np.ones(len(toks)), pitch=np.ones(len(toks)), boundary=np.zeros(len(toks)),
        multi_channel_attention=Multi(), token_to_feature=None
    )
    assert isinstance(y2, (float, np.ndarray))
    print("ok")

if __name__ == "__main__":
    trio.run(_run)
