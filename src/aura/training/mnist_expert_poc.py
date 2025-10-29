#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import json
import sys
import logging
import time
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from flax.core.frozen_dict import freeze, unfreeze

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from aura.bio_inspired.phasor_bank import PhasorBankJAX
from aura.bio_inspired.spiking_attention import SpikingAttentionJAX
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore
from aura.bio_inspired.expert_registry import build_core_kwargs_for_zone, expert_ckpt_path
from aura.bio_inspired.expert_io import save_params, load_params
from aura.bio_inspired.merit_board import MeritBoard
from aura.bio_inspired.personality_jax import PersonalityModulator
from aura.bio_inspired.personality_engine import PersonalityEngineJAX


class MNISTExpertPOC(nn.Module):
    hidden_dim: int = 768
    num_classes: int = 10
    freeze_experts: bool = False
    personality_traits: tuple = (0.6, 0.6, 0.5, 0.5, 0.4)  # O, C, E, A, N
    thalamic_scale: float = 1.0
    predictive_weight: float = 0.0
    top_k_route: int = 2

    def setup(self):
        self.input_projection = nn.Dense(self.hidden_dim)
        self.phasor_bank = PhasorBankJAX(delta0=7.0, H=32)
        self.phasor_projection = nn.Dense(self.hidden_dim)
        self.attention = SpikingAttentionJAX(decay=0.7, theta=1.0, k_winners=5)
        # Use expert registry for hippocampus zone (memory experts), default to 6 experts
        core_kwargs = build_core_kwargs_for_zone(
            zone="hippocampus", hidden_dim=self.hidden_dim, freeze_experts=self.freeze_experts
        )
        # Enable hierarchical gating (3 groups) for POC
        core_kwargs.setdefault('group_count', 3)
        # Enable soft mixture routing (top-k)
        core_kwargs.setdefault('top_k_route', int(self.top_k_route))
        # Predictive gating head weight
        core_kwargs.setdefault('predictive_weight', self.predictive_weight)
        self.retrieval_core = EnhancedSpikingRetrievalCore(**core_kwargs)
        # Personality bias modules sized to number of experts
        num_exp = core_kwargs["num_experts"]
        self.personality_mod = PersonalityModulator(num_experts=num_exp)
        self.personality_engine = PersonalityEngineJAX(num_experts=num_exp, hidden_dim=self.hidden_dim)
        # Thalamic head to produce global routing bias
        self.thalamic_head = nn.Dense(num_exp)
        self.output_projection = nn.Dense(self.num_classes)

    def __call__(self, x, active_experts: int = None, freeze_mask: jnp.ndarray = None, merit_bias: jnp.ndarray = None, temperature: jnp.ndarray = None, inactive_mask: jnp.ndarray = None):
        x_flat = x.reshape((x.shape[0], -1))
        projected_x = self.input_projection(x_flat)
        x_mean = jnp.mean(projected_x, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean)
        temporal_mapped = self.phasor_projection(temporal_features)
        enhanced_x = projected_x + temporal_mapped
        K = min(32, enhanced_x.shape[-1])
        topk_idx = jax.lax.top_k(enhanced_x, K)[1]
        vocab_size = int(enhanced_x.shape[-1])
        attention_gains = jax.vmap(self.attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        attended_x = enhanced_x * attention_gains
        # Initialize personality engine params (no-op output) using current stimulus
        _ = self.personality_engine(jnp.asarray(self.personality_traits, dtype=jnp.float32), jnp.mean(attended_x, axis=0))
        # Compute personality bias and combine with merit bias
        pbias = self.personality_mod(jnp.asarray(self.personality_traits, dtype=jnp.float32))
        comb_bias = pbias if merit_bias is None else (merit_bias + pbias)
        # Thalamic bias from summarized state
        thal = self.thalamic_head(jnp.mean(attended_x, axis=0)) * self.thalamic_scale
        context = self.retrieval_core(attended_x, active_experts=active_experts, freeze_mask=freeze_mask, merit_bias=comb_bias, temperature=temperature, inactive_mask=inactive_mask, thalamic_bias=thal)
        logits = self.output_projection(context)
        return logits

    def compute_gate_weights(self, x, active_experts: int = None, merit_bias: jnp.ndarray = None, inactive_mask: jnp.ndarray = None):
        """Compute routing weights over experts for a given batch x."""
        x_flat = x.reshape((x.shape[0], -1))
        projected_x = self.input_projection(x_flat)
        x_mean = jnp.mean(projected_x, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean)
        temporal_mapped = self.phasor_projection(temporal_features)
        enhanced_x = projected_x + temporal_mapped
        K = min(32, enhanced_x.shape[-1])
        topk_idx = jax.lax.top_k(enhanced_x, K)[1]
        vocab_size = int(enhanced_x.shape[-1])
        attention_gains = jax.vmap(self.attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        attended_x = enhanced_x * attention_gains
        pbias = self.personality_mod(jnp.asarray(self.personality_traits, dtype=jnp.float32))
        # Personality engine controls using a summarized stimulus (mean over hidden)
        stim = jnp.mean(attended_x, axis=0)
        eng = self.personality_engine(jnp.asarray(self.personality_traits, dtype=jnp.float32), stim)
        pbias_eng = eng['bias_logits']
        temperature = eng['temperature']
        comb_bias = pbias + pbias_eng if merit_bias is None else (merit_bias + pbias + pbias_eng)
        thal = self.thalamic_head(jnp.mean(attended_x, axis=0)) * self.thalamic_scale
        gate_w = self.retrieval_core.compute_gate_weights(attended_x, active_experts, comb_bias, temperature, inactive_mask, thalamic_bias=thal)
        return gate_w

    def compute_thalamic_bias(self, x):
        x_flat = x.reshape((x.shape[0], -1))
        projected_x = self.input_projection(x_flat)
        x_mean = jnp.mean(projected_x, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean)
        temporal_mapped = self.phasor_projection(temporal_features)
        enhanced_x = projected_x + temporal_mapped
        K = min(32, enhanced_x.shape[-1])
        topk_idx = jax.lax.top_k(enhanced_x, K)[1]
        vocab_size = int(enhanced_x.shape[-1])
        attention_gains = jax.vmap(self.attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        attended_x = enhanced_x * attention_gains
        return self.thalamic_head(jnp.mean(attended_x, axis=0)) * self.thalamic_scale

    def compute_expert_outputs(self, x):
        """Return [batch, num_experts, hidden_dim] expert outputs before mixing."""
        x_flat = x.reshape((x.shape[0], -1))
        projected_x = self.input_projection(x_flat)
        x_mean = jnp.mean(projected_x, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean)
        temporal_mapped = self.phasor_projection(temporal_features)
        enhanced_x = projected_x + temporal_mapped
        K = min(32, enhanced_x.shape[-1])
        topk_idx = jax.lax.top_k(enhanced_x, K)[1]
        vocab_size = int(enhanced_x.shape[-1])
        attention_gains = jax.vmap(self.attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        attended_x = enhanced_x * attention_gains
        return self.retrieval_core.expert_outputs(attended_x)

    def compute_controls(self, x):
        """Return (pbias_total, temperature, distill_alpha, merit_momentum) for current batch x."""
        x_flat = x.reshape((x.shape[0], -1))
        projected_x = self.input_projection(x_flat)
        x_mean = jnp.mean(projected_x, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean)
        temporal_mapped = self.phasor_projection(temporal_features)
        enhanced_x = projected_x + temporal_mapped
        K = min(32, enhanced_x.shape[-1])
        topk_idx = jax.lax.top_k(enhanced_x, K)[1]
        vocab_size = int(enhanced_x.shape[-1])
        attention_gains = jax.vmap(self.attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        attended_x = enhanced_x * attention_gains
        # Stimulus for engine: mean over batch
        stim = jnp.mean(attended_x, axis=0)
        eng = self.personality_engine(jnp.asarray(self.personality_traits, dtype=jnp.float32), stim)
        pbias_mod = self.personality_mod(jnp.asarray(self.personality_traits, dtype=jnp.float32))
        pbias_total = pbias_mod + eng['bias_logits']
        return pbias_total, eng['temperature'], eng['distill_alpha'], eng['merit_momentum']

    def compute_distill_loss(self, x, teacher_idx: int, student_idx: int, teacher_weights: jnp.ndarray = None):
        """Compute teacher->student distillation loss for batch x."""
        x_flat = x.reshape((x.shape[0], -1))
        projected_x = self.input_projection(x_flat)
        x_mean = jnp.mean(projected_x, axis=-1)
        temporal_features = jax.vmap(self.phasor_bank)(x_mean)
        temporal_mapped = self.phasor_projection(temporal_features)
        enhanced_x = projected_x + temporal_mapped
        K = min(32, enhanced_x.shape[-1])
        topk_idx = jax.lax.top_k(enhanced_x, K)[1]
        vocab_size = int(enhanced_x.shape[-1])
        attention_gains = jax.vmap(self.attention, in_axes=(0, None))(topk_idx.astype(jnp.int32), vocab_size)
        attended_x = enhanced_x * attention_gains
        return self.retrieval_core.distill_loss(attended_x, teacher_idx, student_idx, teacher_weights)


def load_csv_mnist():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    csv_train_path = os.path.join(project_root, 'data', 'MNIST', 'raw', 'mnist_train.csv')
    csv_test_path = os.path.join(project_root, 'data', 'MNIST', 'raw', 'mnist_test.csv')
    try:
        train_data = np.loadtxt(csv_train_path, delimiter=',', dtype=np.float32)
        y_train = train_data[:, 0].astype(np.int32)
        X_train = (train_data[:, 1:] / 255.0).reshape((-1, 28, 28)).astype(np.float32)
        if os.path.exists(csv_test_path):
            test_data = np.loadtxt(csv_test_path, delimiter=',', dtype=np.float32)
            y_test = test_data[:, 0].astype(np.int32)
            X_test = (test_data[:, 1:] / 255.0).reshape((-1, 28, 28)).astype(np.float32)
        else:
            split = max(5000, int(0.1 * X_train.shape[0]))
            X_test, y_test = X_train[-split:], y_train[-split:]
            X_train, y_train = X_train[:-split], y_train[:-split]
        return X_train, y_train, X_test, y_test
    except Exception:
        n_train = 10000
        n_test = 2000
        X_train = np.random.rand(n_train, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(n_train,), dtype=np.int32)
        X_test = np.random.rand(n_test, 28, 28).astype(np.float32)
        y_test = np.random.randint(0, 10, size=(n_test,), dtype=np.int32)
        return X_train, y_train, X_test, y_test


def create_train_state(rng, model):
    params = model.init(rng, jnp.ones((1, 28, 28)))
    tx = optax.adam(1e-3)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, x, y, active_experts: int, freeze_mask: jnp.ndarray, teacher_idx: int, student_idx: int, alpha_distill: float, merit_bias: jnp.ndarray, temperature: jnp.ndarray, teacher_weights: jnp.ndarray, inactive_mask: jnp.ndarray, emc_lambda: float):
    def loss_fn(params):
        logits = state.apply_fn(params, x, active_experts, freeze_mask, merit_bias, temperature, inactive_mask)
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        d_loss = jax.lax.cond(
            jnp.greater(alpha_distill, 0.0),
            lambda _: state.apply_fn(
                params, x, teacher_idx, student_idx, teacher_weights,
                method=MNISTExpertPOC.compute_distill_loss),
            lambda _: jnp.array(0.0, dtype=jnp.float32),
            operand=None,
        )
        # EMC-style soft freeze penalty for non-selected experts (output-space)
        lambda_emc = emc_lambda
        outs = state.apply_fn(params, x, method=MNISTExpertPOC.compute_expert_outputs)  # [batch, num_exp, hidden]
        outs_sq = jnp.sum(outs * outs, axis=-1)  # [batch, num_exp]
        fm = jnp.asarray(freeze_mask, dtype=jnp.float32)[None, :]
        emc_pen = lambda_emc * jnp.mean(outs_sq * fm)
        loss = ce_loss + alpha_distill * d_loss + emc_pen
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return state, loss, acc


@jax.jit
def eval_step(state, x, y, active_experts: int, merit_bias: jnp.ndarray):
    logits = state.apply_fn(state.params, x, active_experts, None, merit_bias)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y)
    return acc


def run_training(target_acc=0.95, max_epochs=10, batch_size=256, hidden_dim=128, freeze_experts=False,
                 enable_neurogenesis: bool = True, entropy_thr_scale: float = 0.8,
                 spawn_teach_batches: int = 30, bandit_policy: str = 'ucb',
                 emc_lambda: float = 1e-4, predictive_weight: float = 0.0, thalamic_scale: float = 1.0,
                 top_k_route: int = 2, enable_merging: bool = False, merge_sim_thr: float = 0.995, merge_util_thr: float = 0.01,
                 results_json: str = ""):
    X_train, y_train, X_test, y_test = load_csv_mnist()
    model = MNISTExpertPOC(hidden_dim=hidden_dim, num_classes=10, freeze_experts=freeze_experts, thalamic_scale=thalamic_scale, predictive_weight=predictive_weight, top_k_route=top_k_route)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model)
    # Attempt to load expert checkpoints for hippocampus zone
    def load_expert_ckpts(state):
        params = unfreeze(state.params)
        core = params.get('params', {}).get('enhanced_spiking_retrieval_core', None)
        if core is None:
            return state
        keys = list(core.keys())
        exp_keys = [k for k in keys if k.startswith('expert_')]
        for k in exp_keys:
            idx = int(k.split('_')[-1]) if '_' in k else None
            if idx is None:
                continue
            p = expert_ckpt_path('hippocampus', idx)
            try:
                loaded = load_params(core[k], p)
                core[k] = loaded
            except Exception:
                pass
        params['params']['enhanced_spiking_retrieval_core'] = core
        return state.replace(params=freeze(params))
    state = load_expert_ckpts(state)
    # predictive_weight applied via model init
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    # Merit board for live-RAG-like routing bias
    # Initialized to number of experts from registry (hippocampus default: 6)
    merit_board = MeritBoard(num_experts=len(build_core_kwargs_for_zone('hippocampus', hidden_dim)['expert_types']))
    # Neurogenesis and bandit settings (from args)
    enable_neurogenesis = bool(enable_neurogenesis)
    entropy_thr_scale = float(entropy_thr_scale)
    spawn_teach_batches = int(spawn_teach_batches)
    bandit_policy = str(bandit_policy)
    # Pruning state
    n_exp_total = len(build_core_kwargs_for_zone('hippocampus', hidden_dim)['expert_types'])
    pruned_mask = np.zeros((n_exp_total,), dtype=bool)
    prune_after = 6
    prune_util_thr = 0.005
    # Routed accuracy accumulators
    routed_counts = np.zeros((n_exp_total,), dtype=np.float32)
    routed_correct = np.zeros((n_exp_total,), dtype=np.float32)
    # Specialization metric: per-expert label histogram
    label_hist = np.zeros((n_exp_total, 10), dtype=np.int64)

    # Track thalamic magnitude
    thal_mag_sum = 0.0
    thal_mag_cnt = 0
    for epoch in range(max_epochs):
        t0 = time.time()
        perm = np.random.permutation(n_train)
        X_train_shuf = X_train[perm]
        y_train_shuf = y_train[perm]
        # Growth schedule: 1 -> 3 -> 6 experts
        active_experts_base = 1 if epoch < 2 else (3 if epoch < 4 else 6)
        active_experts_dyn = active_experts_base
        spawn_batches_left = 0
        spawned_expert_idx = None
        entropy_sum = 0.0
        entropy_count = 0
        prev_active = 1 if epoch == 2 else (3 if epoch == 4 else active_experts_base)
        clone_phase = (epoch == 2) or (epoch == 4)
        # Track per-expert utilization by accumulating mean gate weights per batch
        util_sums = None
        num_batches = n_train // batch_size
        epoch_loss = 0.0
        for i in range(num_batches):
            s = i * batch_size
            e = s + batch_size
            xb = jnp.array(X_train_shuf[s:e])
            yb = jnp.array(y_train_shuf[s:e])
            # Compute personality controls
            pbias_total, temp, eng_alpha, eng_mom = model.apply(state.params, xb, method=MNISTExpertPOC.compute_controls)
            # Current merit bias with bandit policy
            merit_bias = jnp.array(merit_board.bias(bandit_policy=bandit_policy, ucb_c=1.0)) + pbias_total
            # Compute routing weights with current params and pick the dominant expert for this batch
            gate_w = model.apply(state.params, xb, active_experts_dyn, merit_bias, jnp.array(pruned_mask), method=model.compute_gate_weights)  # [batch, num_experts]
            mean_w = jnp.mean(gate_w, axis=0)  # [num_experts]
            top_expert = int(jnp.argmax(mean_w))
            n_exp = int(gate_w.shape[-1])
            if util_sums is None:
                util_sums = jnp.zeros((n_exp,), dtype=jnp.float32)
            util_sums = util_sums + mean_w
            # Gate entropy (based on mean weights)
            eps = 1e-8
            entropy = float(-jnp.sum(mean_w * jnp.log(jnp.clip(mean_w, eps))))
            entropy_sum += entropy
            entropy_count += 1
            # Committee teacher weights (top-2 by default)
            k = min(2, active_experts_dyn)
            if k > 1:
                vals, idx = jax.lax.top_k(mean_w, k)
                tw = jnp.zeros_like(mean_w)
                tw = tw.at[idx].set(vals)
                tw = tw / (jnp.sum(tw) + 1e-9)
            else:
                tw = mean_w
            # Neurogenesis trigger: high entropy and capacity available
            if enable_neurogenesis and spawn_batches_left == 0 and active_experts_dyn < n_exp and i > 10:
                max_h = float(jnp.log(active_experts_dyn)) if active_experts_dyn > 0 else 0.0
                if entropy > entropy_thr_scale * max_h:
                    spawned_expert_idx = int(active_experts_dyn)
                    active_experts_dyn = active_experts_dyn + 1
                    spawn_batches_left = spawn_teach_batches
                # Confidence-loss mismatch: low entropy but high loss
                elif entropy < 0.3 * max_h:
                    # peek a small forward pass loss threshold from previous loop; use running epoch loss density
                    if i > 5 and (epoch_loss / max(1, i)) > 0.7:
                        spawned_expert_idx = int(active_experts_dyn)
                        active_experts_dyn = active_experts_dyn + 1
                        spawn_batches_left = spawn_teach_batches
            if clone_phase and i < 20:
                student_idx = prev_active if prev_active < n_exp else top_expert
                freeze_mask = jnp.array([i != student_idx for i in range(n_exp)])
                alpha = 1.0
            else:
                if spawn_batches_left > 0 and spawned_expert_idx is not None:
                    # Teach new expert from the current top expert
                    student_idx = spawned_expert_idx
                    freeze_mask = jnp.array([i != student_idx for i in range(n_exp)])
                    alpha = 1.0
                    spawn_batches_left -= 1
                else:
                    freeze_mask = jnp.array([i != top_expert for i in range(n_exp)])
                    if active_experts_dyn > 1:
                        active_mask = jnp.array([i < active_experts_dyn for i in range(n_exp)])
                        util_active = jnp.where(active_mask, util_sums, jnp.inf)
                        util_active = util_active.at[top_expert].set(jnp.inf)
                        student_idx = int(jnp.argmin(util_active))
                        alpha = float(eng_alpha)
                    else:
                        student_idx = top_expert
                        alpha = 0.0
            state, loss, batch_acc = train_step(state, xb, yb, active_experts_dyn, freeze_mask, top_expert, student_idx, alpha, merit_bias, temp, tw, jnp.array(pruned_mask), float(emc_lambda))
            epoch_loss += float(loss)
            # Update merit board with mean routing weights and batch reward
            merit_board.update(np.array(mean_w), float(batch_acc), momentum=float(eng_mom))
            # Routed accuracy accumulation for top expert
            routed_counts[top_expert] += 1.0
            routed_correct[top_expert] += float(batch_acc)
            # Label histogram for specialization (top-1 routed expert)
            y_np = np.array(y_train_shuf[s:e])
            for cls in range(10):
                label_hist[top_expert, cls] += int(np.sum(y_np == cls))
        test_bs = 512
        num_test_batches = n_test // test_bs
        acc_sum = 0.0
        for i in range(num_test_batches):
            s = i * test_bs
            e = s + test_bs
            acc_sum += float(eval_step(state, jnp.array(X_test[s:e]), jnp.array(y_test[s:e]), active_experts_dyn, jnp.array(merit_board.bias(bandit_policy=bandit_policy, ucb_c=1.0))))
        test_acc = acc_sum / max(1, num_test_batches)
        # Log utilization distribution if available
        if util_sums is not None:
            util_dist = (util_sums / jnp.sum(util_sums)).tolist()
            logger.info(f"Epoch {epoch}: experts_active={active_experts_dyn}, util={util_dist}")
        avg_entropy = (entropy_sum / max(1, entropy_count)) if entropy_count > 0 else 0.0
        # Prune experts with very low utilization after warmup
        if epoch >= prune_after and util_sums is not None:
            util_dist_np = np.array((util_sums / jnp.sum(util_sums)).tolist(), dtype=np.float32)
            newly_pruned = (util_dist_np < prune_util_thr)
            pruned_mask = np.logical_or(pruned_mask, newly_pruned)
        # Routed accuracy per expert (top-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            routed_acc = np.where(routed_counts > 0, routed_correct / np.maximum(1e-9, routed_counts), 0.0)
        # Expert merging based on similarity and low utilization
        merges = []
        if enable_merging:
            # Flatten expert params and compute cosine similarity
            params = unfreeze(state.params)
            core = params.get('params', {}).get('enhanced_spiking_retrieval_core', None)
            def flatten_tree(tree):
                out = []
                if isinstance(tree, dict):
                    for v in tree.values():
                        out.extend(flatten_tree(v))
                else:
                    try:
                        arr = np.asarray(tree)
                        if arr.dtype.kind in ['f','i']:
                            out.append(arr.ravel())
                    except Exception:
                        pass
                return out
            vecs = []
            for i in range(n_exp_total):
                k = f'expert_{i}'
                if core is None or k not in core or pruned_mask[i]:
                    vecs.append(None)
                    continue
                parts = flatten_tree(core[k])
                if not parts:
                    vecs.append(None)
                    continue
                v = np.concatenate(parts).astype(np.float32)
                n = np.linalg.norm(v) + 1e-9
                vecs.append(v / n)
            util = None
            if 'util_sums' in locals() and util_sums is not None:
                util = (np.array((util_sums / jnp.sum(util_sums)).tolist(), dtype=np.float32))
            for i in range(n_exp_total):
                if pruned_mask[i] or vecs[i] is None:
                    continue
                for j in range(i+1, n_exp_total):
                    if pruned_mask[j] or vecs[j] is None:
                        continue
                    sim = float(np.dot(vecs[i], vecs[j]))
                    if sim >= float(merge_sim_thr):
                        ui = float(util[i]) if util is not None else 0.0
                        uj = float(util[j]) if util is not None else 0.0
                        # Merge lower-util expert if below threshold
                        if min(ui, uj) < float(merge_util_thr):
                            drop = i if ui <= uj else j
                            pruned_mask[drop] = True
                            merges.append((i, j, sim, ui, uj, drop))
            if merges:
                logger.info(f"Merges: {merges}")
        thal_mean = (thal_mag_sum / max(1, thal_mag_cnt)) if thal_mag_cnt > 0 else 0.0
        logger.info(f"Epoch {epoch}: loss={(epoch_loss/max(1,num_batches)):.4f}, acc={test_acc:.4f}, H={avg_entropy:.4f}, thal={thal_mean:.4f}, pruned={pruned_mask.tolist()}, routed_acc={routed_acc.tolist()}, time={(time.time()-t0):.1f}s")
        # Results JSON logging
        if results_json:
            rec = {
                'epoch': int(epoch),
                'loss': float(epoch_loss/max(1,num_batches)),
                'test_acc': float(test_acc),
                'entropy': float(avg_entropy),
                'thalamic': float(thal_mean),
                'pruned_mask': pruned_mask.tolist(),
                'routed_acc': [float(x) for x in routed_acc.tolist()],
                'label_hist': label_hist.astype(int).tolist(),
                'util': ((util_sums / jnp.sum(util_sums)).tolist() if util_sums is not None else None),
                'merges': merges if merges else [],
            }
            try:
                # Append one JSON object per line
                with open(results_json, 'a') as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
        if test_acc >= target_acc:
            logger.info(f"Target accuracy {target_acc*100:.1f}% reached at epoch {epoch}")
            # Save expert checkpoints on success
            params = unfreeze(state.params)
            core = params.get('params', {}).get('enhanced_spiking_retrieval_core', None)
            if core is not None:
                for k in core:
                    if k.startswith('expert_'):
                        idx = int(k.split('_')[-1]) if '_' in k else None
                        if idx is not None:
                            p = expert_ckpt_path('hippocampus', idx)
                            try:
                                save_params(core[k], p)
                            except Exception:
                                pass
            return True
    logger.info("Training completed")
    # Save expert checkpoints at epoch end
    params = unfreeze(state.params)
    core = params.get('params', {}).get('enhanced_spiking_retrieval_core', None)
    if core is not None:
        for k in core:
            if k.startswith('expert_'):
                idx = int(k.split('_')[-1]) if '_' in k else None
                if idx is not None:
                    p = expert_ckpt_path('hippocampus', idx)
                    try:
                        save_params(core[k], p)
                    except Exception:
                        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="MNIST Expert POC")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--max-epochs', type=int, default=12)
    parser.add_argument('--target-acc', type=float, default=0.95)
    parser.add_argument('--freeze-experts', dest='freeze_experts', action='store_true', default=True)
    parser.add_argument('--no-freeze-experts', dest='freeze_experts', action='store_false')
    parser.add_argument('--bandit-policy', type=str, default='ucb', choices=['ucb','softmax','none'])
    parser.add_argument('--entropy-thr-scale', type=float, default=0.8)
    parser.add_argument('--spawn-teach-batches', type=int, default=30)
    parser.add_argument('--enable-neurogenesis', dest='enable_neurogenesis', action='store_true', default=True)
    parser.add_argument('--disable-neurogenesis', dest='enable_neurogenesis', action='store_false')
    parser.add_argument('--emc-lambda', type=float, default=1e-4)
    parser.add_argument('--predictive-weight', type=float, default=0.0)
    parser.add_argument('--thalamic-scale', type=float, default=1.0)
    parser.add_argument('--top-k-route', type=int, default=2)
    parser.add_argument('--enable-merging', dest='enable_merging', action='store_true', default=False)
    parser.add_argument('--disable-merging', dest='enable_merging', action='store_false')
    parser.add_argument('--merge-sim-thr', type=float, default=0.995)
    parser.add_argument('--merge-util-thr', type=float, default=0.01)
    parser.add_argument('--results-json', type=str, default="")
    args = parser.parse_args()

    ok = run_training(target_acc=args.target_acc, max_epochs=args.max_epochs, batch_size=args.batch_size,
                      hidden_dim=args.hidden_dim, freeze_experts=args.freeze_experts,
                      enable_neurogenesis=args.enable_neurogenesis, entropy_thr_scale=args.entropy_thr_scale,
                      spawn_teach_batches=args.spawn_teach_batches, bandit_policy=('' if args.bandit_policy=='none' else args.bandit_policy),
                      emc_lambda=args.emc_lambda, predictive_weight=args.predictive_weight, thalamic_scale=args.thalamic_scale,
                      top_k_route=args.top_k_route, enable_merging=args.enable_merging, merge_sim_thr=args.merge_sim_thr,
                      merge_util_thr=args.merge_util_thr, results_json=args.results_json)
    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    main()
