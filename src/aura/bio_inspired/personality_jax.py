#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import linen as nn


class PersonalityModulator(nn.Module):
    """Maps Big Five personality traits to expert gating bias logits.
    traits: [5] -> bias_logits: [num_experts]
    """
    num_experts: int
    hidden: int = 32

    @nn.compact
    def __call__(self, traits: jnp.ndarray) -> jnp.ndarray:
        # traits expected shape [5]
        t = jnp.asarray(traits, dtype=jnp.float32)
        h = nn.Dense(self.hidden)(t)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.tanh(h)
        bias = nn.Dense(self.num_experts)(h)
        return bias
