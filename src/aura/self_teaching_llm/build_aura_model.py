#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import json
import time
import pickle
import numpy as np

import jax
import jax.numpy as jnp

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aura.self_teaching_llm.tokenizer_spm import SPMTokenizer
from aura.self_teaching_llm.self_teaching_adapter import SelfTeachingAdapter
from aura.training.mnist_expert_poc import run_training as run_mnist_training
from aura.ingestion.txt_loader import load_text_corpus_all
from aura.data.hf_stream import stream_hf_text, stream_local_jsonl_text
from aura.data.hf_ultrachat import stream_ultrachat_sharegpt
from aura.data.hf_conversations import stream_conversations
from aura.data.hf_templategsm import stream_templategsm
from aura.tools.registry import load_registry, save_registry, register_tool, list_tools, get_tool, registry_default_path
from aura.bio_inspired.expert_registry import get_zone_expert_types
from aura.bio_inspired.expert_io import save_params


def _log_root_devices():
    if jax.process_index() == 0:
        print(f"[aura] backend={jax.default_backend()} devices={jax.device_count()} local_devices={jax.local_device_count()} proc_index={jax.process_index()} proc_count={jax.process_count()}")


def run_tokenizer_train(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, 'spiece')
    model_path = SPMTokenizer.train_from_dir(
        input_dir=args.input_dir,
        model_prefix=prefix,
        vocab_size=int(args.vocab_size),
        model_type=args.model_type,
        character_coverage=float(args.character_coverage),
        pad_id=int(args.pad_id),
        user_defined_symbols=[s for s in (args.user_symbols.split(',') if args.user_symbols else []) if s],
        max_sentence_length=int(args.max_sentence_length),
        hard_vocab_limit=bool(args.hard_vocab_limit),
        byte_fallback=bool(args.byte_fallback),
        clean_controls=bool(args.clean_controls),
        normalize_spaces=bool(args.normalize_spaces),
        use_iterator=bool(args.use_iterator),
        ascii_only=bool(args.ascii_only),
    )
    print(model_path)
    return 0


def _make_adapter(args, piece_size: int) -> SelfTeachingAdapter:
    return SelfTeachingAdapter(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=piece_size,
        num_experts=args.num_experts,
        lang_backend=args.lang_backend,
        use_rope=bool(args.use_rope),
        rope_max_len=args.rope_max_len,
        rope_base=args.rope_base,
        spm_model_path=args.spm_model,
        dtype=args.dtype,
    )


def run_pretrain_instruct(args) -> int:
    if args.init_distributed:
        try:
            jax.distributed.initialize(
                coordinator_address=(args.coordinator or None),
                num_processes=(args.process_count or None),
                process_id=(args.process_index or None),
            )
        except Exception as e:
            print(f"[aura] distributed init skipped/failed: {e}")
    _log_root_devices()

    proc = None
    try:
        import sentencepiece as sp
        proc = sp.SentencePieceProcessor(); proc.load(args.spm_model)
        piece_size = int(proc.get_piece_size())
    except Exception as e:
        print(f"[aura] sentencepiece load failed: {e}")
        return 1

    adapter = _make_adapter(args, piece_size)

    metrics = adapter.train_instruct_from_jsonl(
        args.jsonl,
        steps=args.steps,
        lr=args.lr,
        max_len=args.max_len,
        pad_to=args.pad_to,
        limit=args.limit,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate_steps,
        pmap=bool(args.pmap),
        per_device_batch=args.per_device_batch,
    )
    if not metrics:
        print("[aura] training returned no metrics (likely no pairs)")
        return 1
    if jax.process_index() == 0:
        print(json.dumps(metrics))
        if args.ckpt_out:
            os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
            ok = adapter.save_checkpoint(args.ckpt_out)
            print(f"[aura] saved checkpoint to {args.ckpt_out}: {ok}")
    return 0


def _encode_prompt_embedding(adapter: SelfTeachingAdapter, prompt: str, max_len: int) -> jnp.ndarray:
    ids = adapter.spm_tokenizer.encode(prompt, add_bos=True, add_eos=False)
    ids = ids[:max_len]
    ids_arr = jnp.array(ids, dtype=jnp.int32)
    embs = adapter.token_embedding.apply(adapter.embed_params, ids_arr)  # [seq, embed]
    emb = jnp.mean(embs, axis=0, keepdims=True)  # [1, embed]
    return emb


def run_chat(args) -> int:
    _log_root_devices()
    try:
        import sentencepiece as sp
        proc = sp.SentencePieceProcessor(); proc.load(args.spm_model)
        piece_size = int(proc.get_piece_size())
    except Exception as e:
        print(f"[aura] sentencepiece load failed: {e}")
        return 1
    adapter = _make_adapter(args, piece_size)
    if args.ckpt and os.path.exists(args.ckpt):
        adapter.load_checkpoint(args.ckpt)
    else:
        print(f"[aura] checkpoint not found: {args.ckpt}")
        return 1
    # Load zone heads if available
    try:
        root = getattr(args, 'ckpt_root', 'checkpoints')
    except Exception:
        root = 'checkpoints'
    loaded = adapter.load_zone_heads(root)
    if jax.process_index() == 0:
        print(f"[aura] loaded zone heads from {root}: {loaded}")
    prompt_emb = _encode_prompt_embedding(adapter, args.prompt, args.max_len)
    gen_ids, _ = adapter.generate_with_consciousness(
        prompt_embeddings=prompt_emb,
        consciousness_system=None,
        max_len=args.gen_len,
        temperature=args.temperature,
    )
    toks = [int(x) for x in list(jnp.array(gen_ids[0]).tolist())]
    try:
        out = adapter.spm_tokenizer.decode(toks)
    except Exception:
        out = str(toks)
    print(out)
    return 0


def run_pretrain_zones(args) -> int:
    # Zones: hippocampus (MNIST POC), language (instruct), others: future hooks
    zones = [z.strip().lower() for z in (args.zones.split(',') if args.zones else []) if z.strip()]
    if not zones or zones == ['all']:
        zones = ['hippocampus', 'language']
    rc = 0
    if 'hippocampus' in zones:
        print('[aura] Pretraining hippocampus (MNIST POC)...')
        ok = run_mnist_training(
            target_acc=args.mnist_target_acc,
            max_epochs=args.mnist_max_epochs,
            batch_size=args.mnist_batch_size,
            hidden_dim=args.mnist_hidden_dim,
            freeze_experts=True,
            enable_neurogenesis=True,
            entropy_thr_scale=0.8,
            spawn_teach_batches=30,
            bandit_policy='ucb',
            emc_lambda=1e-4,
            predictive_weight=0.0,
            thalamic_scale=1.0,
            top_k_route=2,
            enable_merging=False,
            results_json=args.mnist_results_json,
        )
        print(f"[aura] hippocampus done ok={ok}")
        if not ok:
            rc = 1
    if 'language' in zones:
        print('[aura] Pretraining language (instruction tuning)...')
        rc2 = run_pretrain_instruct(args)
        rc = rc or rc2
    if 'amygdala' in zones:
        print('[aura] Pretraining amygdala (affect bias head)...')
        rc3 = run_pretrain_amygdala(args)
        rc = rc or rc3
    if 'thalamus' in zones:
        print('[aura] Pretraining thalamus (routing head via clusters)...')
        rc4 = run_pretrain_thalamus(args)
        rc = rc or rc4
    if 'hypothalamus' in zones:
        print('[aura] Pretraining hypothalamus (control head)...')
        rc5 = run_pretrain_hypothalamus(args)
        rc = rc or rc5
    for z in zones:
        if z not in ('hippocampus','language'):
            print(f"[aura] Zone '{z}' pretraining not yet implemented")
    return rc


def run_pretrain_hf(args) -> int:
    # Streaming HF dataset (e.g., c4, config 'en')
    if args.init_distributed:
        try:
            jax.distributed.initialize(
                coordinator_address=(args.coordinator or None),
                num_processes=(args.process_count or None),
                process_id=(args.process_index or None),
            )
        except Exception as e:
            print(f"[aura] distributed init skipped/failed: {e}")
    _log_root_devices()
    try:
        import sentencepiece as sp
        proc = sp.SentencePieceProcessor(); proc.load(args.spm_model)
        piece_size = int(proc.get_piece_size())
    except Exception as e:
        print(f"[aura] sentencepiece load failed: {e}")
        return 1
    adapter = _make_adapter(args, piece_size)
    # Build primary iterator (dataset 1)
    def make_it1():
        if args.local_jsonl:
            return stream_local_jsonl_text(args.local_jsonl, text_key=args.text_key)
        ds1 = str(args.dataset).strip().lower()
        cfg1 = (args.config or None)
        if (cfg1 in ('', 'en', None)) and ds1 != 'allenai/c4':
            cfg1 = None
        if ds1 == 'openchat/ultrachat-sharegpt':
            return stream_ultrachat_sharegpt(split=args.split, streaming=True, shuffle=True, seed=0)
        if ds1 == 'agent-ark/toucan-1.5m':
            return stream_conversations('Agent-Ark/Toucan-1.5M', config=cfg1, split=args.split, streaming=True, shuffle=True, seed=0)
        if ds1 == 'math-ai/templategsm':
            return stream_templategsm(split=args.split, streaming=True, shuffle=True, seed=0, include_answer=True)
        if ds1 in (
            'freedomintelligence/medical-r1-distill-data',
            'freedomintelligence/rag-instruct',
            'salesforce/xlam-function-calling-60k',
            'mlabonne/finetome-100k',
            'mlabonne/natural_reasoning-formatted',
            'mlabonne/toolace',
            'mlabonne/lmsys-arena-human-preference-filtered-19k',
        ):
            return stream_conversations(args.dataset, config=cfg1, split=args.split, streaming=True, shuffle=True, seed=0)
        if ds1 == 'huggingfacefw/finewiki':
            return stream_hf_text(args.dataset, split=args.split, text_key='text', config=None, streaming=True, shuffle=True, seed=0)
        return stream_hf_text(args.dataset, split=args.split, text_key=args.text_key, config=cfg1, streaming=True, shuffle=True, seed=0)
    # Optional dataset 2 for mixing
    def make_it2():
        if getattr(args, 'local_jsonl2', ''):
            return stream_local_jsonl_text(args.local_jsonl2, text_key=(args.text_key2 or args.text_key))
        if getattr(args, 'dataset2', ''):
            ds2 = str(args.dataset2).strip().lower()
            cfg2 = (getattr(args, 'config2', None) or None)
            if (cfg2 in ('', 'en', None)) and ds2 != 'allenai/c4':
                cfg2 = None
            if ds2 == 'openchat/ultrachat-sharegpt':
                return stream_ultrachat_sharegpt(split=(args.split2 or args.split), streaming=True, shuffle=True, seed=1)
            if ds2 == 'agent-ark/toucan-1.5m':
                return stream_conversations('Agent-Ark/Toucan-1.5M', config=cfg2, split=(args.split2 or args.split), streaming=True, shuffle=True, seed=1)
            if ds2 == 'math-ai/templategsm':
                return stream_templategsm(split=(args.split2 or args.split), streaming=True, shuffle=True, seed=1, include_answer=True)
            if ds2 in (
                'freedomintelligence/medical-r1-distill-data',
                'freedomintelligence/rag-instruct',
                'salesforce/xlam-function-calling-60k',
                'mlabonne/finetome-100k',
                'mlabonne/natural_reasoning-formatted',
                'mlabonne/toolace',
                'mlabonne/lmsys-arena-human-preference-filtered-19k',
            ):
                return stream_conversations(args.dataset2, config=cfg2, split=(args.split2 or args.split), streaming=True, shuffle=True, seed=1)
            if ds2 == 'huggingfacefw/finewiki':
                return stream_hf_text(args.dataset2, split=(args.split2 or args.split), text_key=(args.text_key2 or 'text'), config=None, streaming=True, shuffle=True, seed=1)
            return stream_hf_text(args.dataset2, split=(args.split2 or args.split), text_key=(args.text_key2 or args.text_key), config=cfg2, streaming=True, shuffle=True, seed=1)
        return None
    it1 = make_it1()
    it2 = make_it2()
    mix_p2 = float(getattr(args, 'mix_p2', 0.0) or 0.0)
    if it2 is None or mix_p2 <= 0.0:
        mixed = it1
    else:
        import random
        def mixed_gen():
            g1 = iter(it1)
            g2 = iter(it2)
            while True:
                use2 = random.random() < mix_p2
                try:
                    if use2:
                        yield next(g2)
                    else:
                        yield next(g1)
                except StopIteration:
                    # Recreate the exhausted generator
                    if use2:
                        g2 = iter(make_it2())
                    else:
                        g1 = iter(make_it1())
                    continue
        mixed = mixed_gen()
    metrics = adapter.fit_lm_spm_streaming(iter(mixed), steps=args.steps, lr=args.lr, seq_len=args.seq_len, batch_size=args.batch_size)
    if jax.process_index() == 0:
        print(json.dumps(metrics))
        if args.ckpt_out:
            os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
            ok = adapter.save_checkpoint(args.ckpt_out)
            print(f"[aura] saved checkpoint to {args.ckpt_out}: {ok}")
    return 0


def run_tools(args) -> int:
    act = args.action
    if act == 'list':
        names = list_tools(registry_default_path())
        print("\n".join(names))
        return 0
    elif act == 'register':
        spec = {
            'module': args.module,
            'callable': args.callable,
            'description': args.description,
            'tags': [s for s in (args.tags.split(',') if args.tags else []) if s],
        }
        ok = register_tool(args.name, spec, registry_default_path())
        print(f"registered {args.name}: {ok}")
        return 0
    elif act == 'build':
        # Create a skeleton tool file and register it
        from aura.tools.builder import create_skeleton_tool
        mod_path = create_skeleton_tool(args.name, description=args.description)
        spec = {
            'module': mod_path.replace('/', '.').replace('\\', '.').replace('.py',''),
            'callable': 'main',
            'description': args.description,
            'tags': [s for s in (args.tags.split(',') if args.tags else []) if s],
        }
        ok = register_tool(args.name, spec, registry_default_path())
        print(f"built {args.name} at {mod_path}, registered={ok}")
        return 0
    elif act == 'run':
        from aura.tools.runtime import run_registered_tool
        kwargs = {}
        if args.kwargs:
            try:
                kwargs = json.loads(args.kwargs)
            except Exception:
                pass
        out = run_registered_tool(args.name, kwargs)
        if out is not None:
            print(json.dumps(out))
        return 0
    else:
        print('unknown tools action')
        return 1


def _pool_embeddings(adapter: SelfTeachingAdapter, texts, max_len: int = 128, batch: int = 64):
    # Encode tokens with padding, average non-pad embeddings
    ids_list, pad_id = adapter.spm_tokenizer.encode_batch(list(texts), max_len=max_len, pad_to=max_len, add_bos=True, add_eos=False)
    ids_np = np.array(ids_list, dtype=np.int32)
    N, L = ids_np.shape
    embs = []
    for i in range(0, N, batch):
        chunk = ids_np[i:i+batch]
        # Accumulate mean over non-pad tokens
        mean_vecs = []
        for row in chunk:
            # Compute embedding per position then mean
            vecs = []
            for t in range(L):
                if row[t] == pad_id:
                    continue
                v = adapter.token_embedding.apply(adapter.embed_params, jnp.array([int(row[t])], dtype=jnp.int32))[0]
                vecs.append(np.array(v))
            if vecs:
                mean_vecs.append(np.mean(np.stack(vecs, axis=0), axis=0))
            else:
                mean_vecs.append(np.zeros((adapter.embed_dim,), dtype=np.float32))
        embs.append(np.stack(mean_vecs, axis=0))
    return np.concatenate(embs, axis=0)


def run_pretrain_amygdala(args) -> int:
    # Train a linear head to predict 8-d affect vector from pooled embeddings
    items = load_text_corpus_all(args.txt_dir, args.json_dir)
    if not items:
        print('[aura] no text corpus for amygdala')
        return 1
    texts = [t for (t, _) in items]
    labels = np.stack([v for (_, v) in items], axis=0)  # [N,8]
    if args.limit and len(texts) > args.limit:
        texts = texts[:args.limit]
        labels = labels[:args.limit]
    # Build adapter to get embeddings
    try:
        import sentencepiece as sp
        proc = sp.SentencePieceProcessor(); proc.load(args.spm_model)
        piece_size = int(proc.get_piece_size())
    except Exception as e:
        print(f"[aura] sentencepiece load failed: {e}")
        return 1
    adapter = _make_adapter(args, piece_size)
    # Initialize adapter params
    adapter.initialize_parameters(batch_size=1)
    # Load instruction-pretrained adapter if available
    if args.ckpt_out and os.path.exists(args.ckpt_out):
        try:
            adapter.load_checkpoint(args.ckpt_out)
            print(f"[aura] loaded adapter ckpt for amygdala: {args.ckpt_out}")
        except Exception:
            pass
    X = _pool_embeddings(adapter, texts, max_len=args.max_len, batch=max(1, args.batch_size))  # [N,D]
    N, D = X.shape
    Y = labels.astype(np.float32)
    # Ridge regression closed form
    lam = 1e-3
    XtX = X.T @ X + lam * np.eye(D, dtype=np.float32)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)  # [D,8]
    b = (Y - X @ W).mean(axis=0)   # [8]
    params = {'W': W.astype(np.float32), 'b': b.astype(np.float32)}
    out_path = os.path.join(args.ckpt_root, 'amygdala', 'bias_head.msgpack')
    save_params(params, out_path)
    print(f"[aura] saved amygdala head to {out_path}")
    return 0


def _kmeans(X: np.ndarray, K: int, iters: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    idx = rng.choice(N, K, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        # Assign
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)  # [N,K]
        labels = d2.argmin(axis=1)
        # Update
        for k in range(K):
            sel = (labels == k)
            if np.any(sel):
                C[k] = X[sel].mean(axis=0)
    return C, labels


def run_pretrain_thalamus(args) -> int:
    # Learn centroids for routing head (nearest-centroid logits)
    items = load_text_corpus_all(args.txt_dir, args.json_dir)
    if not items:
        print('[aura] no text corpus for thalamus')
        return 1
    texts = [t for (t, _) in items]
    if args.limit and len(texts) > args.limit:
        texts = texts[:args.limit]
    try:
        import sentencepiece as sp
        proc = sp.SentencePieceProcessor(); proc.load(args.spm_model)
        piece_size = int(proc.get_piece_size())
    except Exception as e:
        print(f"[aura] sentencepiece load failed: {e}")
        return 1
    adapter = _make_adapter(args, piece_size)
    adapter.initialize_parameters(batch_size=1)
    if args.ckpt_out and os.path.exists(args.ckpt_out):
        try:
            adapter.load_checkpoint(args.ckpt_out)
            print(f"[aura] loaded adapter ckpt for thalamus: {args.ckpt_out}")
        except Exception:
            pass
    X = _pool_embeddings(adapter, texts, max_len=args.max_len, batch=max(1, args.batch_size))  # [N,D]
    K = len(get_zone_expert_types('thalamus'))
    C, labels = _kmeans(X, K, iters=10, seed=0)
    params = {'centroids': C.astype(np.float32)}
    out_path = os.path.join(args.ckpt_root, 'thalamus', 'gate_head.msgpack')
    save_params(params, out_path)
    print(f"[aura] saved thalamus centroids to {out_path}")
    return 0


def run_pretrain_hypothalamus(args) -> int:
    # Learn control head mapping embedding -> (temperature, merit_momentum) using heuristics from length
    items = load_text_corpus_all(args.txt_dir, args.json_dir)
    if not items:
        print('[aura] no text corpus for hypothalamus')
        return 1
    texts = [t for (t, _) in items]
    if args.limit and len(texts) > args.limit:
        texts = texts[:args.limit]
    lens = np.array([max(1, len(t.split())) for t in texts], dtype=np.float32)
    lmean = float(np.mean(lens)); lstd = float(np.std(lens) + 1e-6)
    # Targets
    temp = 0.7 + 0.6 / (1.0 + np.exp(-(lens - lmean) / (lstd + 1e-6)))  # 0.7..1.3
    mom = 0.5 + 0.4 / (1.0 + np.exp((lens - lmean) / (lstd + 1e-6)))    # 0.5..0.9 (shorter -> higher momentum)
    Y = np.stack([temp, mom], axis=-1).astype(np.float32)  # [N,2]
    try:
        import sentencepiece as sp
        proc = sp.SentencePieceProcessor(); proc.load(args.spm_model)
        piece_size = int(proc.get_piece_size())
    except Exception as e:
        print(f"[aura] sentencepiece load failed: {e}")
        return 1
    adapter = _make_adapter(args, piece_size)
    adapter.initialize_parameters(batch_size=1)
    X = _pool_embeddings(adapter, texts, max_len=args.max_len, batch=max(1, args.batch_size))  # [N,D]
    N, D = X.shape
    # Ridge regression
    lam = 1e-3
    XtX = X.T @ X + lam * np.eye(D, dtype=np.float32)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)  # [D,2]
    b = (Y - X @ W).mean(axis=0)   # [2]
    params = {'W': W.astype(np.float32), 'b': b.astype(np.float32)}
    out_path = os.path.join(args.ckpt_root, 'hypothalamus', 'control_head.msgpack')
    save_params(params, out_path)
    print(f"[aura] saved hypothalamus control head to {out_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Build/Run Aura model: tokenizer, pretrain, live")
    sub = p.add_subparsers(dest='cmd', required=True)

    ptok = sub.add_parser('tokenizer', help='Train SentencePiece tokenizer from data/txt')
    ptok.add_argument('--input_dir', type=str, default='data/txt')
    ptok.add_argument('--out_dir', type=str, default='models/spm')
    ptok.add_argument('--vocab_size', type=int, default=2000)
    ptok.add_argument('--model_type', type=str, default='unigram', choices=['unigram','bpe'])
    ptok.add_argument('--character_coverage', type=float, default=0.9995)
    ptok.add_argument('--pad_id', type=int, default=3)
    ptok.add_argument('--max_sentence_length', type=int, default=1000000)
    ptok.add_argument('--hard_vocab_limit', type=int, default=0)
    ptok.add_argument('--byte_fallback', type=int, default=0)
    ptok.add_argument('--clean_controls', type=int, default=1)
    ptok.add_argument('--normalize_spaces', type=int, default=1)
    ptok.add_argument('--use_iterator', type=int, default=1)
    ptok.add_argument('--ascii_only', type=int, default=1)
    ptok.add_argument('--user_symbols', type=str, default='<INST>,<INP>,<RESP>,<SEP>')

    ppre = sub.add_parser('pretrain', help='Instruction pretraining (JSONL) on TPU/GPU/CPU')
    ppre.add_argument('--jsonl', type=str, default='data/json/instruct_55k.jsonl')
    ppre.add_argument('--spm_model', type=str, default='models/spm/spiece.model')
    ppre.add_argument('--embed_dim', type=int, default=768)
    ppre.add_argument('--hidden_dim', type=int, default=512)
    ppre.add_argument('--num_experts', type=int, default=8)
    ppre.add_argument('--lang_backend', type=str, default='lif', choices=['lif','srwkv'])
    ppre.add_argument('--use_rope', type=int, default=1)
    ppre.add_argument('--rope_max_len', type=int, default=2048)
    ppre.add_argument('--rope_base', type=float, default=10000.0)
    ppre.add_argument('--steps', type=int, default=1000)
    ppre.add_argument('--lr', type=float, default=1e-3)
    ppre.add_argument('--max_len', type=int, default=256)
    ppre.add_argument('--pad_to', type=int, default=256)
    ppre.add_argument('--limit', type=int, default=4096)
    ppre.add_argument('--batch_size', type=int, default=128)
    ppre.add_argument('--accumulate_steps', type=int, default=4)
    ppre.add_argument('--dtype', type=str, default='bf16', choices=['f32','bf16'])
    ppre.add_argument('--init_distributed', action='store_true', default=False)
    ppre.add_argument('--coordinator', type=str, default='')
    ppre.add_argument('--process_count', type=int, default=0)
    ppre.add_argument('--process_index', type=int, default=0)
    ppre.add_argument('--pmap', action='store_true', default=False)
    ppre.add_argument('--per_device_batch', type=int, default=8)
    ppre.add_argument('--ckpt_out', type=str, default='models/aura/adapter_ckpt.pkl')

    pchat = sub.add_parser('chat', help='Load checkpoint and generate text from a prompt')
    pchat.add_argument('--spm_model', type=str, default='models/spm/spiece.model')
    pchat.add_argument('--embed_dim', type=int, default=768)
    pchat.add_argument('--hidden_dim', type=int, default=512)
    pchat.add_argument('--num_experts', type=int, default=8)
    pchat.add_argument('--lang_backend', type=str, default='lif', choices=['lif','srwkv'])
    pchat.add_argument('--use_rope', type=int, default=1)
    pchat.add_argument('--rope_max_len', type=int, default=2048)
    pchat.add_argument('--rope_base', type=float, default=10000.0)
    pchat.add_argument('--dtype', type=str, default='bf16', choices=['f32','bf16'])
    pchat.add_argument('--ckpt', type=str, default='models/aura/adapter_ckpt.pkl')
    pchat.add_argument('--prompt', type=str, default='Hello!')
    pchat.add_argument('--max_len', type=int, default=256)
    pchat.add_argument('--gen_len', type=int, default=64)
    pchat.add_argument('--temperature', type=float, default=1.0)
    pchat.add_argument('--ckpt_root', type=str, default='checkpoints')

    phf = sub.add_parser('pretrain_hf', help='Streaming HF dataset LM pretraining (supports mixing two sources)')
    phf.add_argument('--dataset', type=str, default='allenai/c4')
    phf.add_argument('--config', type=str, default='en')
    phf.add_argument('--split', type=str, default='train')
    phf.add_argument('--text_key', type=str, default='text')
    phf.add_argument('--local_jsonl', type=str, default='')
    # Optional second dataset for mixing (e.g., WikiText or Nemotron code JSONL)
    phf.add_argument('--dataset2', type=str, default='')
    phf.add_argument('--config2', type=str, default='')
    phf.add_argument('--split2', type=str, default='')
    phf.add_argument('--text_key2', type=str, default='')
    phf.add_argument('--local_jsonl2', type=str, default='')
    phf.add_argument('--mix_p2', type=float, default=0.3, help='Probability of sampling from dataset2')
    phf.add_argument('--spm_model', type=str, default='models/spm/spiece.model')
    phf.add_argument('--embed_dim', type=int, default=768)
    phf.add_argument('--hidden_dim', type=int, default=512)
    phf.add_argument('--num_experts', type=int, default=8)
    phf.add_argument('--lang_backend', type=str, default='lif', choices=['lif','srwkv'])
    phf.add_argument('--use_rope', type=int, default=1)
    phf.add_argument('--rope_max_len', type=int, default=2048)
    phf.add_argument('--rope_base', type=float, default=10000.0)
    phf.add_argument('--steps', type=int, default=1000)
    phf.add_argument('--lr', type=float, default=1e-3)
    phf.add_argument('--seq_len', type=int, default=256)
    phf.add_argument('--batch_size', type=int, default=128)
    phf.add_argument('--dtype', type=str, default='bf16', choices=['f32','bf16'])
    phf.add_argument('--init_distributed', action='store_true', default=False)
    phf.add_argument('--coordinator', type=str, default='')
    phf.add_argument('--process_count', type=int, default=0)
    phf.add_argument('--process_index', type=int, default=0)
    phf.add_argument('--pmap', action='store_true', default=False)
    phf.add_argument('--per_device_batch', type=int, default=8)
    phf.add_argument('--ckpt_out', type=str, default='models/aura/adapter_ckpt.pkl')

    ptools = sub.add_parser('tools', help='Tool registry')
    ptools.add_argument('--action', type=str, required=True, choices=['list','register','build','run'])
    ptools.add_argument('--name', type=str, default='')
    ptools.add_argument('--module', type=str, default='')
    ptools.add_argument('--callable', type=str, default='main')
    ptools.add_argument('--description', type=str, default='')
    ptools.add_argument('--tags', type=str, default='')
    ptools.add_argument('--kwargs', type=str, default='')

    args = p.parse_args()

    if args.cmd == 'tokenizer':
        return run_tokenizer_train(args)
    elif args.cmd == 'pretrain':
        return run_pretrain_instruct(args)
    elif args.cmd == 'pretrain_hf':
        return run_pretrain_hf(args)
    elif args.cmd == 'chat':
        return run_chat(args)
    elif args.cmd == 'pretrain_zones':
        return run_pretrain_zones(args)
    elif args.cmd == 'tools':
        return run_tools(args)
    else:
        print('unknown command')
        return 1


if __name__ == '__main__':
    sys.exit(main())
