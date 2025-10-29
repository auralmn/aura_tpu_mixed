#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate last-dim pairs: (x_even, x_odd) -> (-x_odd, x_even)."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot_even = -x_odd
    x_rot_odd = x_even
    # Interleave even/odd back along last dim
    out = jnp.empty_like(x)
    out = out.at[..., ::2].set(x_rot_even)
    out = out.at[..., 1::2].set(x_rot_odd)
    return out


def build_rope_cache(max_len: int, dim: int, base: float = 10000.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute RoPE cos/sin caches.
    Returns (cos: [max_len, dim], sin: [max_len, dim]).
    dim must be even.
    """
    assert dim % 2 == 0, "RoPE dimension must be even"
    half = dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half, 1.0) / half))  # [half]
    positions = jnp.arange(max_len)[:, None]  # [max_len, 1]
    freqs = positions * inv_freq[None, :]  # [max_len, half]
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    # Repeat each pair across even/odd dims
    cos = jnp.repeat(cos, 2, axis=-1)  # [max_len, dim]
    sin = jnp.repeat(sin, 2, axis=-1)  # [max_len, dim]
    return cos.astype(jnp.float32), sin.astype(jnp.float32)


def apply_rope(x: jnp.ndarray, cos_t: jnp.ndarray, sin_t: jnp.ndarray) -> jnp.ndarray:
    """Apply RoPE at a single position t using cos_t/sin_t of shape [dim].
    x: [..., dim]
    """
    # Broadcast cos/sin to x shape
    while cos_t.ndim < x.ndim:
        cos_t = cos_t[None, ...]
        sin_t = sin_t[None, ...]
    return (x * cos_t) + (rotate_half(x) * sin_t)
