#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from typing import Iterator, Optional


def stream_templategsm(split: str = "train",
                        streaming: bool = True,
                        shuffle: bool = True,
                        seed: int = 0,
                        include_answer: bool = True) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("pip install datasets to use HF streaming")

    ds = load_dataset("math-ai/TemplateGSM", split=split, streaming=streaming)
    if shuffle:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=10000)
        except Exception:
            pass

    for ex in ds:
        q = None
        a = None
        # Common keys
        for k in ("question", "prompt", "problem", "query", "text"):
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                q = v.strip()
                break
        for k in ("answer", "final", "solution", "response", "label"):
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                a = v.strip()
                break
        if q and include_answer and a:
            yield f"User:\n{q}\n\nAssistant:\n{a}"
        elif q:
            yield q
        elif a:
            yield a
