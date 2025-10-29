from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union
import asyncio
import threading

from numpy import ndarray
import numpy as np

@dataclass
class SpikingAttention:
    """k‑WTA spiking attention over a token sequence (no backprop)."""
    decay: float = 0.7       # leak factor in [0,1): v <- decay*v + I
    theta: float = 1.0       # spiking threshold
    k_winners: int = 5       # number of winners (k‑WTA)
    gain_up: float = 1.5     # LR multiplier for winners
    gain_down: float = 0.6   # LR multiplier for non‑winners that appeared
    
    def compute_gains(self, token_seq: List[int], vocab_size: int) -> Optional[np.ndarray]:
        if not token_seq:
            return None
        v: Dict[int, float] = {}
        spikes: Dict[int, int] = {}
        for j in token_seq:
            # Fix: Ensure j is a scalar integer
            j = int(j) if np.isscalar(j) else int(j.item())
            vj = self.decay * v.get(j, 0.0) + 1.0
            if vj >= self.theta:
                spikes[j] = spikes.get(j, 0) + 1
                vj -= self.theta  # soft reset
            v[j] = vj
        # rank by spike count then membrane residue
        ranked = sorted(spikes.items(),
                        key=lambda kv: (-kv[1], -v.get(kv[0], 0.0)))
        winners = set(j for j,_ in ranked[:max(1, self.k_winners)])
        gains = np.ones(vocab_size, dtype=np.float64)
        seen = set(spikes.keys()) | set(v.keys())
        for j in seen:
            # Fix: Ensure j is within bounds
            if 0 <= j < vocab_size:
                gains[j] = self.gain_up if j in winners else self.gain_down
        return gains

@dataclass
class Head(ABC):
    # Weight vector; subclasses are responsible for sizing/initialization
    w: ndarray = field(init=False, default_factory=lambda: np.zeros((0,), dtype=np.float64))

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> float:
        pass

    @abstractmethod
    def attach(self, *args: Any, **kwargs: Any):
        pass


class NLMSHead(Head):
    mu_bias: float = 0.4
    mu_tok: float = 0.3
    mu_pos: float = 0.3
    mu_realm: float = 0.9
    mu_phase: float = 0.9
    mu_ctop: float = 0.5
    mu_pros: float = 0.0   # we don't learn from prosody feature; it's used for gating
    l2: float = 0.0
    clamp: tuple | None = None
    learn_bias: bool = True
    guard_sign: bool = False
    arousal_band: tuple | None = None
    error_thresh: float = 0.08
    prosody_gate_strength: float = 0.5  # scales down tok/POS LR when prosody is high
    tok_slice: slice = slice(0, 0)
    pos_slice: slice | None = None
    realm_slice: slice = slice(0, 0)
    phase_slice: slice = slice(0, 0)
    ctop_idx: int | None = None
    pros_idx: int | None = None
    enable_attention: bool = False
    attention_config: Dict | None = None
    vocab_size: int = 50000
    

    def __init__(self, n_features: int, n_outputs: int, mu: float = 0.8, 
             l2_decay: float = 0.0, clip01: bool = True, nonneg_z: bool = True, 
             normalize: bool = True, seed: int = 7, tok_slice: slice = slice(0, 0), 
             pos_slice: slice | None = None, realm_slice: slice = slice(0, 0), 
             phase_slice: slice = slice(0, 0), ctop_idx: int | None = None, 
             prosody_idx: int | None = None,
             # NEW: Add attention parameters
             enable_attention: bool = False,
             attention_config: Dict | None = None,
             vocab_size: int = 50000):        
        # Store configuration and initialize matrix weights
        # Use provided dimensions (do not override)
        if hasattr(n_features, 'item'):  # numpy scalar
            self.n_features = int(n_features[0])
        elif hasattr(n_features, '__iter__'):  # array-like
            self.n_features = int(n_features[0]) if len(n_features) > 0 else 1
        else:
            self.n_features = int(n_features)

        if hasattr(n_outputs, 'item'):  # numpy scalar
            self.n_outputs = int(n_outputs.item())
        elif hasattr(n_outputs, '__iter__'):  # array-like
            self.n_outputs = int(n_outputs[0]) if len(n_outputs) > 0 else 1
        else:
            self.n_outputs = int(n_outputs)

        self.mu = float(mu)
        self.l2_decay = float(l2_decay)
        self.clip01 = bool(clip01)
        self.nonneg_z = bool(nonneg_z)
        self.normalize = bool(normalize)
        self.seed = int(seed)

        # Fix: Ensure slices are properly constructed
        self.tok_slice = slice(tok_slice.start, tok_slice.stop, tok_slice.step)
        self.pos_slice = slice(int(pos_slice.start), int(pos_slice.stop), pos_slice.step) if pos_slice else None
        self.realm_slice = slice(int(realm_slice.start), int(realm_slice.stop), realm_slice.step)
        self.phase_slice = slice(int(phase_slice.start), int(phase_slice.stop), phase_slice.step)

        # Fix: Ensure indices are integers or None
        self.ctop_idx = int(ctop_idx) if ctop_idx is not None else None
        self.pros_idx = int(prosody_idx) if prosody_idx is not None else None

        # --- FIX: actually store attention flags/config so attention paths can run ---
        self.enable_attention = bool(enable_attention)
        self.attention_config = dict(attention_config) if attention_config else None
        self.vocab_size = int(vocab_size)

        # Add async lock for thread safety
        self._lock = asyncio.Lock()

        # Initialize weight matrix so step() can run before attach()
        self.w = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)
        self.spiking_attention = SpikingAttention(**self.attention_config) if self.enable_attention else None
     
    async def attach(self, base_w: np.ndarray,
               tok_slice: slice, realm_slice: slice, phase_slice: slice,
               pos_slice: slice | None = None,
               ctop_idx: int | None = None,
               prosody_idx: int | None = None):
        async with self._lock:
            W = np.array(base_w, dtype=np.float64, copy=True)
            # accept vector or matrix, normalize to (D, C)
            if W.ndim == 1:
                W = W.reshape(-1, 1)
            assert W.shape[0] == self.n_features, f"base_w rows {W.shape[0]} != n_features {self.n_features}"
            assert W.shape[1] == self.n_outputs,  f"base_w cols {W.shape[1]} != n_outputs {self.n_outputs}"
            self.w = W
            
            # Fix: Ensure slices are properly constructed from parameters
            self.tok_slice = slice(int(tok_slice.start), int(tok_slice.stop), tok_slice.step)
            self.pos_slice = slice(int(pos_slice.start), int(pos_slice.stop), pos_slice.step) if pos_slice else None
            self.realm_slice = slice(int(realm_slice.start), int(realm_slice.stop), realm_slice.step)
            self.phase_slice = slice(int(phase_slice.start), int(phase_slice.stop), phase_slice.step)
            self.ctop_idx = int(ctop_idx) if ctop_idx is not None else None
            self.pros_idx = int(prosody_idx) if prosody_idx is not None else None
        

    async def step(self, x: np.ndarray, y_true: np.ndarray | float) -> np.ndarray:
        async with self._lock:
            # Fix: Ensure proper array handling and type conversion
            x = np.asarray(x, dtype=np.float64)
            if x.ndim > 1:
                x = x.flatten()
            
            # Ensure x has the right length
            if len(x) != self.n_features:
                raise ValueError(f"Input features {len(x)} != expected {self.n_features}")
            
            y_hat = x @ self.w                      # (C,)
            if self.clamp is None:
                y_out = y_hat
            else:
                lo, hi = self.clamp
                y_out = np.clip(y_hat, lo, hi)
                
            # Fix: Handle y_true conversion more carefully
            if np.isscalar(y_true):
                y_true_vec = np.array([float(y_true)], dtype=np.float64)
            else:
                y_true_vec = np.asarray(y_true, dtype=np.float64).flatten()
                
            if len(y_true_vec) != self.n_outputs:
                raise ValueError(f"Target outputs {len(y_true_vec)} != expected {self.n_outputs}")
                
            e = y_true_vec - y_out                   # (C,)

            if self.guard_sign and self.n_outputs == 1:
                if (y_out[0] > 0) != (y_true_vec[0] > 0) and abs(e[0]) < self.error_thresh:
                    return y_out

            if self.arousal_band is not None:
                if self.n_outputs == 1:
                    lo, hi = self.arousal_band
                    if (lo <= y_true_vec[0] <= hi) and (abs(e[0]) < self.error_thresh):
                        return y_out

            x_upd = x.astype(np.float64).copy()
            if not self.learn_bias:
                x_upd[0] = 0.0

            # Fix: Create mu_vec more carefully with bounds checking
            mu_vec = np.zeros_like(x_upd, dtype=np.float64)
            
            # Bias term
            if len(mu_vec) > 0:
                mu_vec[0] = float(self.mu_bias)
            
            # Token slice - fix bounds checking
            tok_start, tok_stop = self.tok_slice.start, self.tok_slice.stop
            if 0 <= tok_start < len(mu_vec) and tok_start < tok_stop <= len(mu_vec):
                mu_vec[self.tok_slice] = float(self.mu_tok)
            
            # POS slice - fix bounds checking
            if self.pos_slice is not None:
                pos_start, pos_stop = self.pos_slice.start, self.pos_slice.stop
                if 0 <= pos_start < len(mu_vec) and pos_start < pos_stop <= len(mu_vec):
                    mu_vec[self.pos_slice] = float(self.mu_pos)
            
            # Realm slice - fix bounds checking
            realm_start, realm_stop = self.realm_slice.start, self.realm_slice.stop
            if 0 <= realm_start < len(mu_vec) and realm_start < realm_stop <= len(mu_vec):
                mu_vec[self.realm_slice] = float(self.mu_realm)
            
            # Phase slice - fix bounds checking
            phase_start, phase_stop = self.phase_slice.start, self.phase_slice.stop
            if 0 <= phase_start < len(mu_vec) and phase_start < phase_stop <= len(mu_vec):
                mu_vec[self.phase_slice] = float(self.mu_phase)
            
            # Individual indices - fix bounds checking
            if self.ctop_idx is not None and 0 <= self.ctop_idx < len(mu_vec):
                mu_vec[self.ctop_idx] = float(self.mu_ctop)
                
            if self.pros_idx is not None and 0 <= self.pros_idx < len(mu_vec):
                mu_vec[self.pros_idx] = float(self.mu_pros)
                # gate token/POS learning by prosody
                pb = float(x[self.pros_idx])
                gate = max(0.0, 1.0 - self.prosody_gate_strength * pb)
                
                # Apply gating with bounds checking
                if 0 <= tok_start < len(mu_vec) and tok_start < tok_stop <= len(mu_vec):
                    mu_vec[self.tok_slice] *= gate
                if self.pos_slice is not None:
                    pos_start, pos_stop = self.pos_slice.start, self.pos_slice.stop
                    if 0 <= pos_start < len(mu_vec) and pos_start < pos_stop <= len(mu_vec):
                        mu_vec[self.pos_slice] *= gate

            denom = 1e-8 + float(x_upd @ x_upd)
            if denom <= 0.0:  # degenerate
                return y_out

            # NLMS update for each output: w[:,c] += mu_vec * (e[c] * x_upd)/denom
            grad = (x_upd / denom)[:, None] * e[None, :]
            self.w = (1.0 - float(self.l2_decay)) * self.w + (mu_vec[:, None] * grad)
            return y_out
        
    async def step_with_attention(self, x: np.ndarray, y_true: Union[np.ndarray, float], 
                            token_sequence: Optional[List[int]] = None,
                            multi_channel_attention: Optional[Dict[str, object]] = None) -> float:
        """NLMS step with optional attention modulation (supports both old and new attention systems)"""
        if self.enable_attention and token_sequence and self.spiking_attention:
            # Legacy single-channel attention
            attention_gains = self.spiking_attention.compute_gains(token_sequence, self.vocab_size)
            
            if attention_gains is not None:
                # Fix: Better token filtering and bounds checking
                valid_tokens = [token for token in token_sequence 
                              if isinstance(token, (int, np.integer)) and 0 <= int(token) < len(attention_gains)]
                if valid_tokens:
                    avg_attention = np.mean([attention_gains[int(token)] for token in valid_tokens])
                    scale = float(avg_attention)
                else:
                    scale = 1.0
            else:
                scale = 1.0
        elif multi_channel_attention is not None:
            # New multi-channel attention system
            scale = float(multi_channel_attention.get("mu_scalar", 1.0))
        else:
            scale = 1.0
        
        # Temporarily scale token LR only (not global μ)
        old_mu_tok = self.mu_tok
        self.mu_tok = float(self.mu_tok) * scale
        try:
            return await self.step(x, y_true)
        finally:
            self.mu_tok = old_mu_tok

    async def step_with_multi_channel_attention(
        self, 
        x: np.ndarray, 
        y_true: Union[np.ndarray, float],
        token_ids: List[int],
        amp: np.ndarray,
        pitch: np.ndarray,
        boundary: np.ndarray,
        multi_channel_attention,
        token_to_feature: Optional[Dict[int, int]] = None
    ) -> Union[np.ndarray, float]:
        """NLMS step with multi-channel spiking attention modulation"""
        if not self.enable_attention:
            return await self.step(x, y_true)
        
        # Compute multi-channel attention
        attn_res = multi_channel_attention.compute(
            token_ids=token_ids,
            amp=amp,
            pitch=pitch,
            boundary=boundary,
            feature_size=(self.tok_slice.stop - self.tok_slice.start),
            token_to_feature=token_to_feature
        )
        
        # Apply attention modulation to mu_vec
        x_upd = x.astype(np.float64).copy()
        if not self.learn_bias:
            x_upd[0] = 0.0

        mu_vec = np.zeros_like(x_upd, dtype=np.float64)
        mu_vec[0] = float(self.mu_bias)
        
        # Fix: Better slice handling with bounds checking
        tok_start, tok_stop = self.tok_slice.start, self.tok_slice.stop
        if 0 <= tok_start < len(mu_vec) and tok_start < tok_stop <= len(mu_vec):
            mu_vec[self.tok_slice] = float(self.mu_tok)
            
        if self.pos_slice is not None:
            pos_start, pos_stop = self.pos_slice.start, self.pos_slice.stop
            if 0 <= pos_start < len(mu_vec) and pos_start < pos_stop <= len(mu_vec):
                mu_vec[self.pos_slice] = float(self.mu_pos)
                
        realm_start, realm_stop = self.realm_slice.start, self.realm_slice.stop
        if 0 <= realm_start < len(mu_vec) and realm_start < realm_stop <= len(mu_vec):
            mu_vec[self.realm_slice] = float(self.mu_realm)
            
        phase_start, phase_stop = self.phase_slice.start, self.phase_slice.stop
        if 0 <= phase_start < len(mu_vec) and phase_start < phase_stop <= len(mu_vec):
            mu_vec[self.phase_slice] = float(self.mu_phase)
            
        if self.ctop_idx is not None and 0 <= self.ctop_idx < len(mu_vec):
            mu_vec[self.ctop_idx] = float(self.mu_ctop)
            
        if self.pros_idx is not None and 0 <= self.pros_idx < len(mu_vec):
            mu_vec[self.pros_idx] = float(self.mu_pros)
            # gate token/POS learning by prosody
            pb = float(x[self.pros_idx])
            gate = max(0.0, 1.0 - self.prosody_gate_strength * pb)
            if 0 <= tok_start < len(mu_vec) and tok_start < tok_stop <= len(mu_vec):
                mu_vec[self.tok_slice] *= gate
            if self.pos_slice is not None:
                pos_start, pos_stop = self.pos_slice.start, self.pos_slice.stop
                if 0 <= pos_start < len(mu_vec) and pos_start < pos_stop <= len(mu_vec):
                    mu_vec[self.pos_slice] *= gate

        # Apply multi-channel attention modulation with bounds checking
        if attn_res.get("per_feature_gains") is not None:
            g = attn_res["per_feature_gains"]
            L = tok_stop - tok_start
            if isinstance(g, np.ndarray) and g.shape[0] == L and L > 0:
                mu_vec[self.tok_slice] *= g
            else:
                # dimension mismatch → fall back to scalar
                mu_vec[self.tok_slice] *= float(attn_res["mu_scalar"])
        else:
            if 0 <= tok_start < len(mu_vec) and tok_start < tok_stop <= len(mu_vec):
                mu_vec[self.tok_slice] *= float(attn_res["mu_scalar"])

        # Continue with NLMS update using modified mu_vec
        if np.isscalar(y_true):
            y_true_vec = np.array([float(y_true)], dtype=np.float64)
        else:
            y_true_vec = np.asarray(y_true, dtype=np.float64).flatten()
            
        if len(y_true_vec) != self.n_outputs:
            raise ValueError(f"Target outputs {len(y_true_vec)} != expected {self.n_outputs}")
        
        y_hat = x @ self.w
        if self.clamp is None:
            y_out = y_hat
        else:
            lo, hi = self.clamp
            y_out = np.clip(y_hat, lo, hi)
        
        e = y_true_vec - y_out

        # Guard conditions
        if self.guard_sign and self.n_outputs == 1:
            if (y_out[0] > 0) != (y_true_vec[0] > 0) and abs(e[0]) < self.error_thresh:
                return float(y_out[0]) if self.n_outputs == 1 else y_out

        if self.arousal_band is not None:
            if self.n_outputs == 1:
                lo, hi = self.arousal_band
                if (lo <= y_true_vec[0] <= hi) and (abs(e[0]) < self.error_thresh):
                    return float(y_out[0]) if self.n_outputs == 1 else y_out

        denom = 1e-8 + float(x_upd @ x_upd)
        if denom <= 0.0:
            return float(y_out[0]) if self.n_outputs == 1 else y_out

        # NLMS update for each output: w[:,c] += mu_vec * (e[c] * x_upd)/denom
        grad = (x_upd / denom)[:, None] * e[None, :]
        self.w = (1.0 - float(self.l2_decay)) * self.w + (mu_vec[:, None] * grad)
        return float(y_out[0]) if self.n_outputs == 1 else y_out

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Fix: Handle input array properly
        x = np.asarray(X, dtype=np.float64)
        if x.ndim > 1:
            x = x.flatten()
        if len(x) != self.n_features:
            raise ValueError(f"Input features {len(x)} != expected {self.n_features}")
            
        y = x @ self.w                       # (C,)
        if self.clamp is None:
            return y
        lo, hi = self.clamp
        return np.clip(y, lo, hi)

    # convenience for codepaths that used a sync API
    def update(self, X: np.ndarray, Y: Union[np.ndarray, float]) -> np.ndarray:
        # one-step synchronous shim
        return asyncio.run(self.step(X, Y))