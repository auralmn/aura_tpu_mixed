#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np


class MeritBoard:
    """Simple merit system to bias routing based on historical utility.
    Maintains an exponential moving average of per-expert rewards.
    """

    def __init__(self, num_experts: int, momentum: float = 0.9, scale: float = 1.0, eps: float = 1e-6):
        self.num_experts = int(num_experts)
        self.momentum = float(momentum)
        self.scale = float(scale)
        self.eps = float(eps)
        self.merit = np.zeros((self.num_experts,), dtype=np.float32)
        # Bandit statistics
        self.counts = np.zeros((self.num_experts,), dtype=np.int32)
        self.values = np.zeros((self.num_experts,), dtype=np.float32)
        self.total_steps = 0

    def update(self, weights: np.ndarray, reward: float, momentum: float | None = None,
               bandit_policy: str | None = None, ucb_c: float = 1.0, softmax_temp: float = 0.5) -> None:
        """Update merits using expert participation weights and a scalar reward.
        weights: shape [num_experts]
        reward: scalar (e.g., batch accuracy or 1 - loss)
        """
        w = np.asarray(weights, dtype=np.float32)
        r = float(reward)
        if momentum is not None:
            self.momentum = float(momentum)
        contrib = w * r
        self.merit = self.momentum * self.merit + (1.0 - self.momentum) * contrib
        # Bandit update: pick dominant expert and update its value estimate
        chosen = int(np.argmax(w))
        self.total_steps += 1
        self.counts[chosen] += 1
        n = max(1, self.counts[chosen])
        self.values[chosen] += (r - self.values[chosen]) / n

    def bias(self, bandit_policy: str | None = None, ucb_c: float = 1.0, softmax_temp: float = 0.5) -> np.ndarray:
        """Return bias for gating logits combining EMA merit and (optional) bandit score."""
        # Merit z-score
        m = self.merit
        mz = m - m.mean()
        ms = m.std()
        if ms < self.eps:
            ms = 1.0
        mz = mz / ms
        # Bandit score
        if bandit_policy == 'ucb':
            # UCB score with exploration
            counts = np.maximum(1, self.counts)
            expl = ucb_c * np.sqrt(np.log(max(2, self.total_steps)) / counts)
            score = self.values + expl
        elif bandit_policy == 'softmax':
            # Preference proportional to softmax over values
            v = self.values - self.values.mean()
            ex = np.exp(v / max(self.eps, softmax_temp))
            p = ex / max(self.eps, ex.sum())
            score = p
        else:
            score = np.zeros_like(self.values)
        # Normalize bandit score
        sz = score - score.mean()
        ss = score.std()
        if ss < self.eps:
            ss = 1.0
        sz = sz / ss
        z = mz + sz
        return (self.scale * z).astype(np.float32)

    def reset(self) -> None:
        self.merit[:] = 0.0

    def set_momentum(self, momentum: float) -> None:
        self.momentum = float(momentum)
