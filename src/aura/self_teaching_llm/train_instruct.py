#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import logging

import jax
import jax.numpy as jnp

try:
    import sentencepiece as sp
except Exception as e:
    sp = None

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aura.self_teaching_llm.self_teaching_adapter import SelfTeachingAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_instruct")


def main():
    parser = argparse.ArgumentParser(description="Instruction tuning with SelfTeachingAdapter on JSONL dataset")
    parser.add_argument('--jsonl', type=str, default='data/json/instruct_55k.jsonl')
    parser.add_argument('--spm-model', type=str, default='models/spm/spiece.model')
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--lang-backend', type=str, default='lif', choices=['lif','srwkv'])
    parser.add_argument('--use-rope', type=int, default=1)
    parser.add_argument('--rope-max-len', type=int, default=2048)
    parser.add_argument('--rope-base', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--pad-to', type=int, default=256)
    parser.add_argument('--limit', type=int, default=4096, help='Number of pairs to use (to limit memory)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--accumulate-steps', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='f32', choices=['f32','bf16'])
    parser.add_argument('--init-distributed', action='store_true', default=False)
    parser.add_argument('--coordinator', type=str, default='')
    parser.add_argument('--process-count', type=int, default=0)
    parser.add_argument('--process-index', type=int, default=0)
    parser.add_argument('--pmap', action='store_true', default=False)
    parser.add_argument('--per-device-batch', type=int, default=8)
    args = parser.parse_args()

    if sp is None:
        logger.error("sentencepiece is required. pip install sentencepiece")
        return 1
    if not os.path.exists(args.spm_model):
        logger.error(f"SPM model not found: {args.spm_model}")
        return 1
    if not os.path.exists(args.jsonl):
        logger.error(f"JSONL file not found: {args.jsonl}")
        return 1

    # Align vocab size to the tokenizer piece size
    proc = sp.SentencePieceProcessor()
    proc.load(args.spm_model)
    piece_size = int(proc.get_piece_size())
    logger.info(f"Loaded SPM model with piece_size={piece_size}")

    if args.init_distributed:
        try:
            jax.distributed.initialize(
                coordinator_address=(args.coordinator or None),
                num_processes=(args.process_count or None),
                process_id=(args.process_index or None),
            )
        except Exception as e:
            logger.warning(f"Distributed init skipped/failed: {e}")

    if jax.process_index() == 0:
        logger.info(f"backend={jax.default_backend()} devices={jax.device_count()} local_devices={jax.local_device_count()} process_index={jax.process_index()} process_count={jax.process_count()}")

    adapter = SelfTeachingAdapter(
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
        logger.error("Training failed or no pairs found.")
        return 1
    logger.info(f"Instruction tuning completed. metrics={metrics}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
