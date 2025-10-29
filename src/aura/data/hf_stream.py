#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Iterator, Optional


def stream_hf_text(dataset_name: str,
                   split: str = "train",
                   text_key: str = "text",
                   config: Optional[str] = None,
                   streaming: bool = True,
                   shuffle: bool = True,
                   seed: int = 0) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("pip install datasets to use HF streaming")
    ds = load_dataset(dataset_name, config, split=split, streaming=streaming)
    if shuffle:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=10000)
        except Exception:
            pass
    for ex in ds:
        t = ex.get(text_key, None)
        if isinstance(t, str) and t.strip():
            yield t


def stream_local_jsonl_text(path: str, text_key: str = "text") -> Iterator[str]:
    import os, json
    if os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, "**/*.jsonl"), recursive=True))
    else:
        files = [path]
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    t = obj.get(text_key)
                    if isinstance(t, str) and t.strip():
                        yield t
        except Exception:
            continue
