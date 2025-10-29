#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from typing import Iterator, Optional


def stream_ultrachat_sharegpt(split: str = "train",
                               include_system: bool = True,
                               streaming: bool = True,
                               shuffle: bool = True,
                               seed: int = 0) -> Iterator[str]:
    """
    Stream conversational transcripts from openchat/ultrachat-sharegpt as plain text.
    Joins turns with role headers (User/Assistant[/System]).
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("pip install datasets to use HF streaming")

    ds = load_dataset("openchat/ultrachat-sharegpt", split=split, streaming=streaming)
    if shuffle:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=10000)
        except Exception:
            pass

    def _join_messages(items):
        out_lines = []
        for m in items:
            role = str(m.get("role") or m.get("from") or "unknown").lower()
            content = m.get("content") if "content" in m else m.get("value")
            if not isinstance(content, str) or not content.strip():
                continue
            if role in ("user", "human"):
                out_lines.append("User:\n" + content.strip())
            elif role in ("assistant", "gpt", "bot"):
                out_lines.append("Assistant:\n" + content.strip())
            elif role == "system":
                if include_system:
                    out_lines.append("System:\n" + content.strip())
            else:
                out_lines.append(f"{role.capitalize()}:\n" + content.strip())
        return "\n\n".join(out_lines)

    for ex in ds:
        # Try messages schema
        if isinstance(ex, dict) and "messages" in ex and isinstance(ex["messages"], list):
            txt = _join_messages(ex["messages"])
            if txt:
                yield txt
                continue
        # Try conversations schema
        if isinstance(ex, dict) and "conversations" in ex and isinstance(ex["conversations"], list):
            txt = _join_messages(ex["conversations"])
            if txt:
                yield txt
                continue
        # Try prompt/response
        if isinstance(ex, dict) and ("prompt" in ex or "response" in ex):
            p = ex.get("prompt", "").strip()
            r = ex.get("response", "").strip()
            if p or r:
                out = []
                if p:
                    out.append("User:\n" + p)
                if r:
                    out.append("Assistant:\n" + r)
                yield "\n\n".join(out)
                continue
        # Fallback: text/content
        for key in ("text", "content"):
            v = ex.get(key)
            if isinstance(v, str) and v.strip():
                yield v.strip()
                break
