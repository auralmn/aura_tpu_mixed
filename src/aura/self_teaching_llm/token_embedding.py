#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class TokenEmbedding(nn.Module):
    vocab_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        table = self.param('embedding', nn.initializers.normal(stddev=0.02), (self.vocab_size, self.embed_dim))
        # token_ids: [batch] or [batch, ...]; gather along last dim of ids
        return jnp.take(table, token_ids, axis=0)
