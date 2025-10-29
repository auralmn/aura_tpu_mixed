#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import linen as nn


class PersonalityEngineJAX(nn.Module):
    num_experts: int
    hidden_dim: int
    trait_dim: int = 5
    emotion_dim: int = 8
    appraisal_dim: int = 16

    @nn.compact
    def __call__(self, traits: jnp.ndarray, stimulus: jnp.ndarray):
        t = jnp.asarray(traits, dtype=jnp.float32)
        s = jnp.asarray(stimulus, dtype=jnp.float32)
        # Trait pathway
        th = nn.Dense(32)(t)
        th = nn.gelu(th)
        th = nn.Dense(32)(th)
        th = nn.gelu(th)
        # Stimulus pathway
        sh = nn.Dense(64)(s)
        sh = nn.gelu(sh)
        sh = nn.Dense(32)(sh)
        sh = nn.gelu(sh)
        comb = jnp.concatenate([th, sh], axis=-1)
        # Emotions
        emo = nn.Dense(self.emotion_dim)(comb)
        emo = nn.sigmoid(emo)
        # Bias logits over experts
        bias_in = jnp.concatenate([emo, th], axis=-1)
        bias_logits = nn.Dense(self.num_experts)(bias_in)
        # Temperature in [0.5, 2.0]
        temp_raw = nn.Dense(1)(comb)
        temperature = 0.5 + 1.5 * nn.sigmoid(temp_raw)
        # Distillation alpha in [0,1]
        alpha_raw = nn.Dense(1)(comb)
        distill_alpha = nn.sigmoid(alpha_raw)
        # Merit momentum in [0.8, 0.99]
        mom_raw = nn.Dense(1)(comb)
        merit_momentum = 0.8 + 0.19 * nn.sigmoid(mom_raw)
        return {
            'bias_logits': bias_logits.squeeze(),
            'temperature': temperature.squeeze(),
            'distill_alpha': distill_alpha.squeeze(),
            'merit_momentum': merit_momentum.squeeze(),
            'emotions': emo,
        }
