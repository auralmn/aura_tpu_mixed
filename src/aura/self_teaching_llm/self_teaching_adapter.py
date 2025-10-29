#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import json
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import List, Optional, Tuple, Callable

from .spiking_retrieval_core import SpikingRetrievalCore
from .spiking_language_core import SpikingLanguageCore
from .token_decoder import TokenDecoder
from .token_embedding import TokenEmbedding
from .generation_loop import generate_text_with_consciousness
from .tokenizer_spm import SPMTokenizer
from aura.bio_inspired.expert_io import load_params


class SelfTeachingAdapter:
    """
    Self-teaching adapter for LLM that integrates consciousness context.
    Coordinates between spiking retrieval, language core, and token decoding.
    """
    
    def __init__(self, 
                 embed_dim: int = 768,
                 hidden_dim: int = 512,
                 vocab_size: int = 32000,
                 num_experts: int = 16,
                 lang_backend: str = 'lif',
                 use_rope: bool = False,
                 rope_max_len: int = 2048,
                 rope_base: float = 10000.0,
                 spm_model_path: Optional[str] = None,
                 dtype: str = 'f32'):
        """
        Initialize self-teaching adapter.
        
        Args:
            embed_dim: Embedding dimension
            hidden_dim: Hidden state dimension for spiking cores
            vocab_size: Vocabulary size for token decoding
            num_experts: Number of experts in Liquid-MoE retrieval core
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.lang_backend = lang_backend
        self.use_rope = use_rope
        self.rope_max_len = rope_max_len
        self.rope_base = rope_base
        self.dtype = dtype  # 'f32' or 'bf16'
        # Tokenizer (SentencePiece) optional
        self.spm_tokenizer = None
        if spm_model_path is not None:
            try:
                self.spm_tokenizer = SPMTokenizer(spm_model_path)
            except Exception:
                self.spm_tokenizer = None
        
        # Initialize components
        self.retrieval_core = SpikingRetrievalCore(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_dim=embed_dim
        )
        
        self.lang_core = SpikingLanguageCore(
            hidden_dim=hidden_dim,
            backend=self.lang_backend,
            use_rope=self.use_rope,
            rope_base=self.rope_base
        )
        
        self.token_decoder = TokenDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size
        )
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
        # Component parameters (will be initialized during first use)
        self.retrieval_params = None
        self.lang_params = None
        self.decoder_params = None
        self.embed_params = None
        # Optimizers for training embeddings/decoder
        self.embed_tx = None
        self.embed_opt_state = None
        self.decoder_tx = None
        self.decoder_opt_state = None
        
        # Random key for initialization
        self.key = jax.random.PRNGKey(0)
        # Zone heads (optional)
        self._thalamus_centroids = None  # jnp.ndarray [K, embed_dim]
        self._hypothalamus_W = None      # jnp.ndarray [embed_dim, 2]
        self._hypothalamus_b = None      # jnp.ndarray [2]
        self._amygdala_W = None          # optional
        self._amygdala_b = None

    @property
    def compute_dtype(self):
        return jnp.bfloat16 if str(self.dtype).lower() in ('bf16','bfloat16') else jnp.float32
    
    def initialize_parameters(self, batch_size: int = 1):
        """
        Initialize all component parameters.
        
        Args:
            batch_size: Batch size for parameter initialization
        """
        # Create dummy inputs for initialization
        dummy_embeddings = jnp.ones((batch_size, self.embed_dim))
        dummy_rates = jnp.ones((batch_size, self.hidden_dim))
        
        # Initialize parameters
        self.retrieval_params = self.retrieval_core.init(self.key, dummy_embeddings.astype(self.compute_dtype))
        self.lang_params = self.lang_core.init(self.key, dummy_rates, 
                                             self.lang_core.initialize_state(batch_size))
        self.decoder_params = self.token_decoder.init(self.key, dummy_rates)
        self.embed_params = self.token_embedding.init(self.key, jnp.ones((batch_size,), dtype=jnp.int32))
        # Initialize optimizers
        if self.embed_tx is None:
            self.embed_tx = optax.adam(1e-3)
            self.embed_opt_state = self.embed_tx.init(self.embed_params)
        if self.decoder_tx is None:
            self.decoder_tx = optax.adam(1e-3)
            self.decoder_opt_state = self.decoder_tx.init(self.decoder_params)
    
    def generate_with_consciousness(self,
                                  prompt_embeddings: jnp.ndarray,
                                  consciousness_system: Optional[Callable] = None,
                                  max_len: int = 50,
                                  temperature: float = 1.0) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Generate text with consciousness system integration.
        
        Args:
            prompt_embeddings: Prompt embeddings [batch, embed_dim]
            consciousness_system: Consciousness system for context biasing (optional)
            max_len: Maximum generation length
            temperature: Sampling temperature for token generation
            
        Returns:
            Tuple of (generated_token_ids [batch, max_len], all_rates [List of [batch, hidden_dim]])
        """
        # Initialize parameters if not already done
        if self.retrieval_params is None:
            self.initialize_parameters(prompt_embeddings.shape[0])
        
        # Create component functions with parameters applied
        # Compute optional thalamic bias and temperature from zone heads using prompt embedding mean
        def _controls_from_emb(emb: jnp.ndarray):
            gate_bias = None
            temp = None
            if self._thalamus_centroids is not None:
                c = self._thalamus_centroids  # [K,D]
                e = jnp.mean(emb, axis=0)     # [D]
                d2 = jnp.sum((c - e[None, :]) ** 2, axis=-1)  # [K]
                gate_bias = -d2
            if self._hypothalamus_W is not None and self._hypothalamus_b is not None:
                e = jnp.mean(emb, axis=0)
                y = jnp.matmul(e, self._hypothalamus_W) + self._hypothalamus_b  # [2]
                temp = jnp.clip(y[0], 0.5, 2.0)
            return gate_bias, temp

        th_bias, hyp_temp = _controls_from_emb(prompt_embeddings)

        def retrieval_fn(query_embedding):
            return self.retrieval_core.apply(self.retrieval_params, query_embedding, temperature=hyp_temp)
        
        # Add the retrieve_context method to the function
        retrieval_fn.retrieve_context = retrieval_fn
        
        def lang_fn(input_state, prev_state):
            return self.lang_core.apply(self.lang_params, input_state, prev_state)
        
        def decoder_fn(rate_vector):
            return self.token_decoder.apply(self.decoder_params, rate_vector)
        
        # Add required methods to functions
        lang_fn.initialize_state = lambda batch_size: self.lang_core.initialize_state(batch_size)
        
        # Generate text with consciousness
        def embed_fn(token_ids):
            return self.token_embedding.apply(self.embed_params, token_ids)
        generated_tokens, all_rates = generate_text_with_consciousness(
            prompt_embeddings,
            retrieval_fn,
            lang_fn,
            decoder_fn,
            consciousness_system=consciousness_system,
            max_len=max_len,
            temperature=float(temperature),
            use_rope=self.use_rope,
            rope_max_len=self.rope_max_len,
            rope_base=self.rope_base,
            embed_fn=embed_fn
        )
        
        return generated_tokens, all_rates

    # Zone heads loader
    def load_zone_heads(self, root: str = "checkpoints") -> bool:
        ok = False
        # Thalamus centroids
        th_path = os.path.join(root, 'thalamus', 'gate_head.msgpack')
        try:
            # Try pickle first
            with open(th_path, 'rb') as f:
                th = pickle.load(f)
            C = th.get('centroids', None)
            if C is None:
                raise ValueError('not pickle centroids')
        except Exception:
            try:
                th = load_params({}, th_path)
                C = th.get('centroids', None)
            except Exception:
                C = None
        if C is not None:
            C_arr = jnp.array(C)
            # Ensure centroids match number of experts
            K, D = C_arr.shape
            Ne = int(self.num_experts)
            if K == Ne:
                self._thalamus_centroids = C_arr
            elif K > Ne:
                self._thalamus_centroids = C_arr[:Ne]
            else:
                pad = jnp.zeros((Ne - K, D), dtype=C_arr.dtype)
                self._thalamus_centroids = jnp.concatenate([C_arr, pad], axis=0)
            ok = True
        # Hypothalamus control head
        hy_path = os.path.join(root, 'hypothalamus', 'control_head.msgpack')
        try:
            with open(hy_path, 'rb') as f:
                hy = pickle.load(f)
            W = hy.get('W', None); b = hy.get('b', None)
        except Exception:
            try:
                hy = load_params({}, hy_path)
                W = hy.get('W', None); b = hy.get('b', None)
            except Exception:
                W = None; b = None
        if W is not None and b is not None:
            self._hypothalamus_W = jnp.array(W)
            self._hypothalamus_b = jnp.array(b)
            ok = True
        # Amygdala (currently unused in retrieval)
        am_path = os.path.join(root, 'amygdala', 'bias_head.msgpack')
        try:
            with open(am_path, 'rb') as f:
                am = pickle.load(f)
            W = am.get('W', None); b = am.get('b', None)
        except Exception:
            try:
                am = load_params({}, am_path)
                W = am.get('W', None); b = am.get('b', None)
            except Exception:
                W = None; b = None
        if W is not None and b is not None:
            self._amygdala_W = jnp.array(W)
            self._amygdala_b = jnp.array(b)
            ok = True
        return ok
    
    def teach_self(self, generated_text: str, generated_embeddings: jnp.ndarray):
        """
        Teach the model from its own generated text.
        
        Args:
            generated_text: Text generated by the model
            generated_embeddings: Embeddings of the generated text
        """
        # In a full implementation, this would:
        # 1. Add generated text to memory fragments
        # 2. Update consciousness system with new knowledge
        # 3. Potentially update component parameters through learning
        pass

    # --- Embedding training ---
    def train_embeddings(self, token_batch: jnp.ndarray, steps: int = 100, lr: float = 1e-3):
        """Train embeddings (and decoder) on next-token prediction.
        token_batch: [batch, seq_len] integer ids
        """
        if self.retrieval_params is None:
            self.initialize_parameters(token_batch.shape[0])
        # (Re)create optimizers with desired LR
        self.embed_tx = optax.adam(lr)
        self.embed_opt_state = self.embed_tx.init(self.embed_params)
        self.decoder_tx = optax.adam(lr)
        self.decoder_opt_state = self.decoder_tx.init(self.decoder_params)

        def loss_fn(embed_params, decoder_params, token_batch):
            batch_size, seq_len = token_batch.shape
            state = self.lang_core.initialize_state(batch_size)
            emb0 = self.token_embedding.apply(embed_params, token_batch[:, 0])
            h0 = self.retrieval_core.apply(self.retrieval_params, emb0.astype(self.compute_dtype))

            def step(carry, t_inputs):
                h_t, st = carry
                tgt = t_inputs  # [batch]
                rates, st2 = self.lang_core.apply(self.lang_params, h_t, st)
                # Cast activations if needed
                rates_c = rates.astype(self.compute_dtype)
                logits = self.token_decoder.apply(decoder_params, rates_c)
                ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt)
                emb_next = self.token_embedding.apply(embed_params, tgt)
                h_next = self.retrieval_core.apply(self.retrieval_params, emb_next.astype(self.compute_dtype))
                return (h_next, st2), ce

            (_, _), ce_seq = jax.lax.scan(step, (h0, state), token_batch[:, 1:])
            return jnp.mean(ce_seq)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
        for _ in range(steps):
            (loss_val), (g_emb, g_dec) = grad_fn(self.embed_params, self.decoder_params, token_batch)
            # Update embeddings
            updates_e, self.embed_opt_state = self.embed_tx.update(g_emb, self.embed_opt_state, self.embed_params)
            self.embed_params = optax.apply_updates(self.embed_params, updates_e)
            # Update decoder
            updates_d, self.decoder_opt_state = self.decoder_tx.update(g_dec, self.decoder_opt_state, self.decoder_params)
            self.decoder_params = optax.apply_updates(self.decoder_params, updates_d)
        return True

    def train_embeddings_masked(self, token_batch, loss_mask, steps: int = 100, lr: float = 1e-3, batch_size: int = 64, accumulate_steps: int = 1):
        if self.retrieval_params is None:
            # Initialize with a small batch to keep memory low
            self.initialize_parameters(max(1, int(batch_size)))
        self.embed_tx = optax.adam(lr)
        self.embed_opt_state = self.embed_tx.init(self.embed_params)
        self.decoder_tx = optax.adam(lr)
        self.decoder_opt_state = self.decoder_tx.init(self.decoder_params)

        def loss_fn(embed_params, decoder_params, token_batch, loss_mask):
            batch_size, seq_len = token_batch.shape
            state = self.lang_core.initialize_state(batch_size)
            emb0 = self.token_embedding.apply(embed_params, token_batch[:, 0])
            h0 = self.retrieval_core.apply(self.retrieval_params, emb0.astype(self.compute_dtype))

            def step(carry, t_inputs):
                h_t, st = carry
                tgt, m = t_inputs  # [batch], [batch]
                rates, st2 = self.lang_core.apply(self.lang_params, h_t, st)
                rates_c = rates.astype(self.compute_dtype)
                logits = self.token_decoder.apply(decoder_params, rates_c)
                ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt)
                num = jnp.sum(ce * m)
                den = jnp.sum(m)
                preds = jnp.argmax(logits, axis=-1)
                corr = jnp.sum((preds == tgt) * m)
                emb_next = self.token_embedding.apply(embed_params, tgt)
                h_next = self.retrieval_core.apply(self.retrieval_params, emb_next.astype(self.compute_dtype))
                return (h_next, st2), (num, den, corr)

            (_, _), out = jax.lax.scan(step, (h0, state), (token_batch[:, 1:], loss_mask[:, 1:]))
            num = jnp.sum(out[0])
            den = jnp.sum(out[1])
            corr = jnp.sum(out[2])
            loss = num / (den + 1e-9)
            return loss, (num, den, corr)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))
        loss_sum = 0.0
        num_sum = 0.0
        den_sum = 0.0
        corr_sum = 0.0
        # Keep dataset on CPU to limit GPU memory
        toks = np.asarray(token_batch)
        msk = np.asarray(loss_mask)
        N = int(toks.shape[0])
        L = int(toks.shape[1])
        bs = max(1, int(min(batch_size, N)))
        # Prepare an index permutation
        perm = np.random.permutation(N)
        pos = 0
        if jax.process_index() == 0:
            print(f"[instruct] dataset N={N}, L={L}, batch_size={bs}, steps={steps}, accumulate_steps={accumulate_steps}")
        compiled = False
        for si in range(steps):
            # Reset accumulators for this optimizer step
            acc_g_emb = None
            acc_g_dec = None
            step_num = 0.0
            step_den = 0.0
            step_corr = 0.0
            for mi in range(max(1, int(accumulate_steps))):
                if pos + bs > N:
                    # reshuffle for next epoch
                    rem = N - pos
                    idx = np.concatenate([perm[pos:], np.random.permutation(N)[:(bs - rem)]]) if rem > 0 else np.random.permutation(N)[:bs]
                    perm = np.random.permutation(N)
                    pos = (bs - rem) % N
                else:
                    idx = perm[pos:pos+bs]
                    pos += bs
                batch_tokens = jnp.array(toks[idx], dtype=jnp.int32)
                batch_mask = jnp.array(msk[idx], dtype=jnp.float32)
                if not compiled:
                    if jax.process_index() == 0:
                        print("[instruct] compiling JAX function (first micro-step)...")
                    t0 = time.time()
                    (loss_val, (num_val, den_val, corr_val)), (g_emb, g_dec) = grad_fn(self.embed_params, self.decoder_params, batch_tokens, batch_mask)
                    if jax.process_index() == 0:
                        print(f"[instruct] compile took {time.time()-t0:.2f}s")
                    compiled = True
                else:
                    (loss_val, (num_val, den_val, corr_val)), (g_emb, g_dec) = grad_fn(self.embed_params, self.decoder_params, batch_tokens, batch_mask)
                # Accumulate grads
                if acc_g_emb is None:
                    acc_g_emb = jax.tree_util.tree_map(lambda x: x / max(1, int(accumulate_steps)), g_emb)
                    acc_g_dec = jax.tree_util.tree_map(lambda x: x / max(1, int(accumulate_steps)), g_dec)
                else:
                    acc_g_emb = jax.tree_util.tree_map(lambda a, b: a + b / max(1, int(accumulate_steps)), acc_g_emb, g_emb)
                    acc_g_dec = jax.tree_util.tree_map(lambda a, b: a + b / max(1, int(accumulate_steps)), acc_g_dec, g_dec)
                step_num += float(num_val)
                step_den += float(den_val)
                step_corr += float(corr_val)
            # Apply accumulated grads once per macro step
            updates_e, self.embed_opt_state = self.embed_tx.update(acc_g_emb, self.embed_opt_state, self.embed_params)
            self.embed_params = optax.apply_updates(self.embed_params, updates_e)
            updates_d, self.decoder_opt_state = self.decoder_tx.update(acc_g_dec, self.decoder_opt_state, self.decoder_params)
            self.decoder_params = optax.apply_updates(self.decoder_params, updates_d)
            loss_sum += float(step_num / max(1e-9, step_den))
            num_sum += float(step_num)
            den_sum += float(step_den)
            corr_sum += float(step_corr)
            step_acc = (float(step_corr) / max(1e-9, float(step_den))) if float(step_den) > 0 else 0.0
            if jax.process_index() == 0:
                print(f"[instruct] step {si+1}/{steps}: loss={float(step_num/max(1e-9,step_den)):.4f} acc={step_acc:.4f} masked={float(step_den):.0f} (accum {accumulate_steps}x)")
        mean_loss = loss_sum / max(1, steps)
        mean_acc = (corr_sum / max(1e-9, den_sum)) if den_sum > 0 else 0.0
        return {
            'mean_loss': float(mean_loss),
            'mean_acc': float(mean_acc),
            'steps': int(steps),
            'masked_tokens': float(den_sum),
        }

    def _train_embeddings_masked_pmap(self, token_batch, loss_mask, steps: int, lr: float, per_device_batch: int, accumulate_steps: int = 1):
        if self.retrieval_params is None:
            self.initialize_parameters(max(1, int(per_device_batch)))
        tx_e = optax.adam(lr)
        tx_d = optax.adam(lr)
        opt_state_e = tx_e.init(self.embed_params)
        opt_state_d = tx_d.init(self.decoder_params)

        def loss_fn(embed_params, decoder_params, token_batch, loss_mask):
            batch_size, seq_len = token_batch.shape
            state = self.lang_core.initialize_state(batch_size)
            emb0 = self.token_embedding.apply(embed_params, token_batch[:, 0])
            h0 = self.retrieval_core.apply(self.retrieval_params, emb0.astype(self.compute_dtype))
            def step(carry, t_inputs):
                h_t, st = carry
                tgt, m = t_inputs
                rates, st2 = self.lang_core.apply(self.lang_params, h_t, st)
                logits = self.token_decoder.apply(decoder_params, rates.astype(self.compute_dtype))
                ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt)
                num = jnp.sum(ce * m)
                den = jnp.sum(m)
                preds = jnp.argmax(logits, axis=-1)
                corr = jnp.sum((preds == tgt) * m)
                emb_next = self.token_embedding.apply(embed_params, tgt)
                h_next = self.retrieval_core.apply(self.retrieval_params, emb_next.astype(self.compute_dtype))
                return (h_next, st2), (num, den, corr)
            (_, _), out = jax.lax.scan(step, (h0, state), (token_batch[:, 1:], loss_mask[:, 1:]))
            num = jnp.sum(out[0]); den = jnp.sum(out[1]); corr = jnp.sum(out[2])
            loss = num / (den + 1e-9)
            return loss, (num, den, corr)

        def pmapped_step(embed_params, decoder_params, token_shard, mask_shard):
            (loss, (num, den, corr)), (g_emb, g_dec) = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)(embed_params, decoder_params, token_shard, mask_shard)
            g_emb = jax.lax.pmean(g_emb, axis_name='data')
            g_dec = jax.lax.pmean(g_dec, axis_name='data')
            num = jax.lax.psum(num, axis_name='data')
            den = jax.lax.psum(den, axis_name='data')
            corr = jax.lax.psum(corr, axis_name='data')
            return (loss, (num, den, corr)), (g_emb, g_dec)

        pstep = jax.pmap(pmapped_step, axis_name='data', in_axes=(None, None, 0, 0))

        toks = np.asarray(token_batch)
        msk = np.asarray(loss_mask)
        N, L = toks.shape
        dcount = jax.local_device_count()
        bpd = max(1, int(min(per_device_batch, max(1, N // dcount))))
        # Prepare replicated params on devices
        rep_eparams = jax.device_put_replicated(self.embed_params, jax.local_devices())
        rep_dparams = jax.device_put_replicated(self.decoder_params, jax.local_devices())

        loss_sum = num_sum = den_sum = corr_sum = 0.0
        perm = np.random.permutation(N)
        pos = 0
        if jax.process_index() == 0:
            print(f"[instruct/pmap] devices={jax.device_count()}, local={dcount}, per_device_batch={bpd}, steps={steps}, accum={accumulate_steps}")
        for si in range(steps):
            step_num = step_den = step_corr = 0.0
            for mi in range(max(1, int(accumulate_steps))):
                need = dcount * bpd
                if pos + need > N:
                    rem = N - pos
                    extra = need - rem
                    idx = np.concatenate([perm[pos:], np.random.permutation(N)[:extra]]) if rem > 0 else np.random.permutation(N)[:need]
                    perm = np.random.permutation(N)
                    pos = extra % N
                else:
                    idx = perm[pos:pos+need]
                    pos += need
                batch_tokens = toks[idx].reshape(dcount, bpd, L)
                batch_mask = msk[idx].reshape(dcount, bpd, L)
                (loss_val, (num_val, den_val, corr_val)), (g_emb, g_dec) = pstep(rep_eparams, rep_dparams, jnp.array(batch_tokens, jnp.int32), jnp.array(batch_mask, jnp.float32))
                # Bring averaged grads to host from first replica
                g_emb_host = jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), g_emb)
                g_dec_host = jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), g_dec)
                updates_e, opt_state_e = tx_e.update(g_emb_host, opt_state_e, self.embed_params)
                self.embed_params = optax.apply_updates(self.embed_params, updates_e)
                updates_d, opt_state_d = tx_d.update(g_dec_host, opt_state_d, self.decoder_params)
                self.decoder_params = optax.apply_updates(self.decoder_params, updates_d)
                # Refresh replicated params
                rep_eparams = jax.device_put_replicated(self.embed_params, jax.local_devices())
                rep_dparams = jax.device_put_replicated(self.decoder_params, jax.local_devices())
                step_num += float(jax.device_get(num_val[0]))
                step_den += float(jax.device_get(den_val[0]))
                step_corr += float(jax.device_get(corr_val[0]))
            loss_sum += step_num / max(1e-9, step_den)
            num_sum += step_num; den_sum += step_den; corr_sum += step_corr
            if jax.process_index() == 0:
                step_acc = (step_corr / max(1e-9, step_den)) if step_den > 0 else 0.0
                print(f"[instruct/pmap] step {si+1}/{steps}: loss={step_num/max(1e-9,step_den):.4f} acc={step_acc:.4f} masked={step_den:.0f}")
        mean_loss = loss_sum / max(1, steps)
        mean_acc = (corr_sum / max(1e-9, den_sum)) if den_sum > 0 else 0.0
        return {
            'mean_loss': float(mean_loss),
            'mean_acc': float(mean_acc),
            'steps': int(steps),
            'masked_tokens': float(den_sum),
        }
    
    # --- SentencePiece utilities ---
    def load_spm(self, model_path: str) -> bool:
        try:
            self.spm_tokenizer = SPMTokenizer(model_path)
            return True
        except Exception:
            self.spm_tokenizer = None
            return False

    def encode_texts_spm(self, texts: List[str], max_len: int = 64, pad_to: int = 64) -> jnp.ndarray:
        if self.spm_tokenizer is None:
            raise RuntimeError("SPM tokenizer not loaded. Provide spm_model_path or call load_spm().")
        ids_list, _ = self.spm_tokenizer.encode_batch(texts, max_len=max_len, pad_to=pad_to)
        return jnp.array(ids_list, dtype=jnp.int32)

    def train_embeddings_from_texts(self, texts: List[str], steps: int = 200, lr: float = 1e-3, max_len: int = 64, pad_to: int = 64) -> bool:
        tokens = self.encode_texts_spm(texts, max_len=max_len, pad_to=pad_to)
        return self.train_embeddings(tokens, steps=steps, lr=lr)

    def encode_instruct_pairs_spm(self, pairs: List[Tuple[str, str]], max_len: int = 256, pad_to: int = 256, return_numpy: bool = True) -> Tuple[object, object]:
        if self.spm_tokenizer is None:
            raise RuntimeError("SPM tokenizer not loaded. Provide spm_model_path or call load_spm().")
        batch_ids: List[List[int]] = []
        batch_mask: List[List[float]] = []
        pad_id = 0
        if self.spm_tokenizer.proc is not None and self.spm_tokenizer.proc.pad_id() >= 0:
            pad_id = int(self.spm_tokenizer.proc.pad_id())
        proc = self.spm_tokenizer.proc
        p2i = proc.piece_to_id if proc is not None else None
        inst_id = p2i('<INST>') if p2i is not None else -1
        inp_id = p2i('<INP>') if p2i is not None else -1
        resp_id = p2i('<RESP>') if p2i is not None else -1
        sep_id = p2i('<SEP>') if p2i is not None else -1
        bos_id = proc.bos_id() if proc is not None else -1
        eos_id = proc.eos_id() if proc is not None else -1
        for prompt, response in pairs:
            use_ctrl = (inst_id >= 0) or (inp_id >= 0) or (resp_id >= 0) or (sep_id >= 0)
            if use_ctrl:
                ids = []
                if bos_id is not None and bos_id >= 0:
                    ids.append(int(bos_id))
                if inst_id >= 0:
                    ids.append(int(inst_id))
                p_ids = self.spm_tokenizer.encode(prompt, add_bos=False, add_eos=False)
                ids.extend(p_ids)
                if inp_id >= 0 and ('Input:' in prompt or True):
                    ids.append(int(inp_id))
                elif sep_id >= 0:
                    ids.append(int(sep_id))
                # Mark the start of response content after optional <RESP>
                if resp_id >= 0:
                    ids.append(int(resp_id))
                resp_start = len(ids)
                r_ids = self.spm_tokenizer.encode(response, add_bos=False, add_eos=False)
                ids.extend(r_ids)
                if eos_id is not None and eos_id >= 0:
                    ids.append(int(eos_id))
                # Build mask to exactly match ids length, 1.0 only over response content tokens
                mask = [0.0] * len(ids)
                end = min(len(ids), resp_start + len(r_ids))
                for k in range(resp_start, end):
                    mask[k] = 1.0
            else:
                p_ids = self.spm_tokenizer.encode(prompt, add_bos=True, add_eos=False)
                r_ids = self.spm_tokenizer.encode(response, add_bos=False, add_eos=True)
                ids = p_ids + r_ids
                # Response starts after prompt tokens
                resp_start = len(p_ids)
                mask = [0.0] * len(ids)
                end = min(len(ids), resp_start + len(r_ids))
                for k in range(resp_start, end):
                    mask[k] = 1.0
            if len(ids) > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
            if pad_to is not None and len(ids) < pad_to:
                need = pad_to - len(ids)
                ids = ids + [pad_id] * need
                mask = mask + [0.0] * need
            batch_ids.append(ids)
            batch_mask.append(mask)
        if return_numpy:
            return np.array(batch_ids, dtype=np.int32), np.array(batch_mask, dtype=np.float32)
        else:
            return jnp.array(batch_ids, dtype=jnp.int32), jnp.array(batch_mask, dtype=jnp.float32)

    def train_instruct_from_jsonl(self, jsonl_path: str, steps: int = 500, lr: float = 1e-3, max_len: int = 256, pad_to: int = 256, limit: Optional[int] = None, batch_size: int = 64, accumulate_steps: int = 1, pmap: bool = False, per_device_batch: int = 8) -> bool:
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(jsonl_path)
        pairs: List[Tuple[str, str]] = []
        line_count = 0
        def add_pair(p: str, r: str):
            if p is None: p = ''
            if r is None: return
            rs = r.strip()
            if not rs:
                return
            pairs.append((p, rs))

        def extract_conv(messages):
            # messages: list of {role: user/assistant/system, content: str}
            if not isinstance(messages, list):
                return
            dialog = []
            for m in messages:
                role = (m.get('role') or m.get('from') or '').lower()
                content = m.get('content') or m.get('value') or ''
                if not isinstance(content, str):
                    continue
                dialog.append((role, content))
            # Build pairs from user->assistant turns with running context
            ctx = []
            for role, content in dialog:
                if role in ('system','context'):
                    ctx.append(f"System:\n{content}\n")
                elif role in ('user','human'):
                    ctx.append(f"User:\n{content}\n")
                elif role in ('assistant','gpt','bot'):
                    prompt = ''.join(ctx) + "Response:\n"
                    add_pair(prompt, content)
                    if limit is not None and len(pairs) >= int(limit):
                        return

        with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                line_count += 1
                if limit is not None and len(pairs) >= int(limit):
                    break
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                # Schema 1: Alpaca-style
                instr = obj.get('instruction') or obj.get('prompt') or obj.get('task') or ''
                inp = obj.get('input') or ''
                out = obj.get('output') or obj.get('response') or obj.get('answer') or obj.get('completion') or obj.get('target') or ''
                if out:
                    prompt = f"Instruction:\n{instr}\n"
                    if inp:
                        prompt += f"Input:\n{inp}\n"
                    prompt += "Response:\n"
                    add_pair(prompt, out)
                    continue
                # Schema 2: QA
                q = obj.get('question'); a = obj.get('answer')
                if a:
                    prompt = f"Question:\n{q or ''}\nResponse:\n"
                    add_pair(prompt, a)
                    continue
                # Schema 3: Conversations/messages
                conv = obj.get('conversations') or obj.get('messages') or obj.get('dialog') or obj.get('chat')
                if conv:
                    extract_conv(conv)
                    continue
                # Schema 4: generic text -> target
                txt = obj.get('text') or obj.get('source')
                tgt = obj.get('target') or obj.get('label') or obj.get('output_text') or obj.get('response_text')
                if tgt:
                    prompt = f"Instruction:\n{(instr or '')}\nInput:\n{txt or ''}\nResponse:\n"
                    add_pair(prompt, tgt)
                    continue
                # Schema 5: single-field text with <human>: ... <bot>: ... markers
                if isinstance(txt, str) and '<bot>:' in txt:
                    try:
                        # Keep everything before first <bot>: as prompt context, everything after as response
                        left, right = txt.split('<bot>:', 1)
                        # Normalize markers for prompt
                        left_norm = left.replace('<human>:', 'User:\n').strip()
                        prompt = f"{left_norm}\nResponse:\n"
                        response = right.strip()
                        # If multiple pairs exist, we take the first response block
                        add_pair(prompt, response)
                        continue
                    except Exception:
                        pass
                if line_count % 5000 == 0:
                    print(f"[instruct] parsed lines={line_count}, pairs_collected={len(pairs)}")
        if not pairs:
            return False
        print(f"[instruct] total pairs={len(pairs)} (limit={limit if limit is not None else 'none'})")
        tokens, mask = self.encode_instruct_pairs_spm(pairs, max_len=max_len, pad_to=pad_to, return_numpy=True)
        print(f"[instruct] encoded tokens shape={tuple(tokens.shape)}, mask ones={int(jnp.sum(mask).item())}")
        if pmap and jax.device_count() > 1:
            return self._train_embeddings_masked_pmap(tokens, mask, steps=steps, lr=lr, per_device_batch=per_device_batch, accumulate_steps=accumulate_steps)
        else:
            return self.train_embeddings_masked(tokens, mask, steps=steps, lr=lr, batch_size=batch_size, accumulate_steps=accumulate_steps)

    def save_checkpoint(self, path: str) -> bool:
        try:
            obj = {
                'retrieval_params': self.retrieval_params,
                'lang_params': self.lang_params,
                'decoder_params': self.decoder_params,
                'embed_params': self.embed_params,
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'num_experts': self.num_experts,
                'spm_model_path': getattr(self.spm_tokenizer, 'model_path', None),
            }
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            return True
        except Exception:
            return False

    def load_checkpoint(self, path: str) -> bool:
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            self.retrieval_params = obj.get('retrieval_params')
            self.lang_params = obj.get('lang_params')
            self.decoder_params = obj.get('decoder_params')
            self.embed_params = obj.get('embed_params')
            return True
        except Exception:
            return False
    
    def get_component_status(self) -> dict:
        """
        Get status of all self-teaching components.
        
        Returns:
            Dictionary with component status information
        """
        return {
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'vocab_size': self.vocab_size,
            'num_experts': self.num_experts,
            'parameters_initialized': self.retrieval_params is not None,
            'spm_loaded': self.spm_tokenizer is not None
        }

    def fit_lm_spm_streaming(self, text_iter, steps: int = 1000, lr: float = 1e-3, seq_len: int = 256, batch_size: int = 128) -> dict:
        """
        Pretrain embeddings/decoder for next-token LM using a streaming text iterator and SentencePiece.
        Shapes are fixed by (batch_size, seq_len) for stable JIT.
        """
        if self.spm_tokenizer is None:
            raise RuntimeError("SPM tokenizer not loaded. Provide spm_model_path or call load_spm().")
        if self.retrieval_params is None:
            self.initialize_parameters(max(1, int(batch_size)))
        # Optimizers
        self.embed_tx = optax.adam(lr)
        self.embed_opt_state = self.embed_tx.init(self.embed_params)
        self.decoder_tx = optax.adam(lr)
        self.decoder_opt_state = self.decoder_tx.init(self.decoder_params)

        pad_id = 0
        if self.spm_tokenizer.proc is not None and self.spm_tokenizer.proc.pad_id() >= 0:
            pad_id = int(self.spm_tokenizer.proc.pad_id())

        def make_batch():
            # Construct [batch, seq_len] token ids by concatenating texts
            toks = []
            got = 0
            while got < batch_size:
                try:
                    t = next(text_iter)
                except StopIteration:
                    break
                if not isinstance(t, str) or not t.strip():
                    continue
                ids = self.spm_tokenizer.encode(t, add_bos=True, add_eos=True)
                if len(ids) < seq_len:
                    ids = ids + [pad_id] * (seq_len - len(ids))
                else:
                    ids = ids[:seq_len]
                toks.append(ids)
                got += 1
            if not toks:
                raise StopIteration
            # If short batch, pad with pad rows to keep static shape
            while len(toks) < batch_size:
                toks.append([pad_id] * seq_len)
            return jnp.array(toks, dtype=jnp.int32)

        def loss_fn(embed_params, decoder_params, token_batch):
            B, L = token_batch.shape
            state = self.lang_core.initialize_state(B)
            emb0 = self.token_embedding.apply(embed_params, token_batch[:, 0])
            h0 = self.retrieval_core.apply(self.retrieval_params, emb0.astype(self.compute_dtype))
            def step(carry, tgt):
                h_t, st = carry
                rates, st2 = self.lang_core.apply(self.lang_params, h_t, st)
                logits = self.token_decoder.apply(decoder_params, rates.astype(self.compute_dtype))
                ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt)
                emb_next = self.token_embedding.apply(embed_params, tgt)
                h_next = self.retrieval_core.apply(self.retrieval_params, emb_next.astype(self.compute_dtype))
                return (h_next, st2), ce
            (_, _), ce_seq = jax.lax.scan(step, (h0, state), token_batch[:, 1:])
            return jnp.mean(ce_seq)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
        loss_sum = 0.0
        compiled = False
        for si in range(steps):
            try:
                batch_tokens = make_batch()
            except StopIteration:
                break
            if not compiled:
                print("[corpus] compiling JAX LM step...")
                t0 = time.time()
                (loss_val), (g_emb, g_dec) = grad_fn(self.embed_params, self.decoder_params, batch_tokens)
                print(f"[corpus] compile took {time.time()-t0:.2f}s")
                compiled = True
            else:
                (loss_val), (g_emb, g_dec) = grad_fn(self.embed_params, self.decoder_params, batch_tokens)
            updates_e, self.embed_opt_state = self.embed_tx.update(g_emb, self.embed_opt_state, self.embed_params)
            self.embed_params = optax.apply_updates(self.embed_params, updates_e)
            updates_d, self.decoder_opt_state = self.decoder_tx.update(g_dec, self.decoder_opt_state, self.decoder_params)
            self.decoder_params = optax.apply_updates(self.decoder_params, updates_d)
            loss_sum += float(loss_val)
            if (si + 1) % 1 == 0:
                print(f"[corpus] step {si+1}/{steps}: loss={float(loss_val):.4f}")
        return {
            'steps': int(steps),
            'mean_loss': float(loss_sum / max(1, steps)),
        }
