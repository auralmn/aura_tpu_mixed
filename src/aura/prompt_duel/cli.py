#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import json
import random

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aura.prompt_duel.dueler import (
    load_candidates,
    load_data,
    PromptDueler,
    OpenSearchRetriever,
)


def parse_retriever(spec: str, topk: int) -> OpenSearchRetriever | None:
    if not spec or spec.lower() in ("none", "null"):
        return None
    if spec.startswith("opensearch://"):
        rest = spec[len("opensearch://"):]
        if "/" not in rest:
            return None
        host, index = rest.split("/", 1)
        base = host
        if not base.startswith("http://") and not base.startswith("https://"):
            base = "http://" + base
        return OpenSearchRetriever(base, index, topk=topk)
    return None


def main():
    ap = argparse.ArgumentParser(description="Prompt Duel Optimizer (Double Thompson Sampling)")
    ap.add_argument("--candidates", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="prompt_duel_out.json")
    ap.add_argument("--best_out", type=str, default="best_prompt.txt")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--duels_per_round", type=int, default=50)
    ap.add_argument("--mutate_period", type=int, default=10)
    ap.add_argument("--no_mutate", action="store_true", default=False)
    ap.add_argument("--label_fraction", type=float, default=0.0)
    ap.add_argument("--judge_provider", type=str, default="openai", choices=["openai", "echo", "hybrid"])
    ap.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--gen_provider", type=str, default="openai", choices=["openai", "echo"])
    ap.add_argument("--gen_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--retriever", type=str, default="")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    prompts = load_candidates(args.candidates)
    data = load_data(args.data)
    if not prompts:
        raise SystemExit("no prompts loaded from --candidates")
    if not data:
        raise SystemExit("no data loaded from --data")

    retr = parse_retriever(args.retriever, args.topk)

    dueler = PromptDueler(
        prompts=prompts,
        data=data,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        gen_provider=args.gen_provider,
        gen_model=args.gen_model,
        label_fraction=args.label_fraction,
        mutate_period=args.mutate_period,
        retriever=retr,
    )
    res = dueler.run(rounds=args.rounds, duels_per_round=args.duels_per_round, mutate=(not args.no_mutate))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "best_index": res["best_index"],
            "W": res["W"],
            "N": res["N"],
            "logs": res["logs"],
        }, f)
    with open(args.best_out, "w", encoding="utf-8") as f:
        f.write(res["best_prompt"])  # noqa

    print(f"[prompt_duel] best_index={res['best_index']} best_prompt_file={args.best_out} out={args.out}")


if __name__ == "__main__":
    main()
