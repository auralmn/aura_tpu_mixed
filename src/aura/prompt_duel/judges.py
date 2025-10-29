#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import json
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import requests


@dataclass
class JudgeDecision:
    winner: str  # 'A', 'B', or 'tie'
    reason: str
    meta: Dict[str, Any]


def _openai_chat(messages, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for OpenAI judge/generator")
    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def _simple_pairwise_template(inp: str, a: str, b: str) -> Tuple[str, str, int]:
    order = ["A", "B"]
    random.shuffle(order)
    if order[0] == "A":
        ra, rb = a, b
        a_pos = 0
    else:
        ra, rb = b, a
        a_pos = 1
    sys = "You are a strict judge. Compare two responses for the same input and choose the better one. Reply with just 'A' or 'B' and nothing else."
    user = (
        "Input:\n" + inp + "\n\n" +
        f"Response A:\n{ra}\n\nResponse B:\n{rb}\n\n" +
        "Which response is better overall given accuracy, relevance, completeness, and clarity? Reply with A or B."
    )
    return sys, user, a_pos


def llm_pairwise_judge(inp: str, out_a: str, out_b: str, provider: str = "openai", model: str = "gpt-4o-mini") -> JudgeDecision:
    if provider == "echo":
        # Heuristic: prefer longer but penalize excessive length
        la, lb = len(out_a), len(out_b)
        score_a = la - abs(la - 800)
        score_b = lb - abs(lb - 800)
        if abs(score_a - score_b) < 5:
            return JudgeDecision("tie", "echo-judge: tie", {"provider": provider})
        return JudgeDecision("A" if score_a > score_b else "B", "echo-judge heuristic", {"provider": provider})
    sys, user, a_pos = _simple_pairwise_template(inp, out_a, out_b)
    content = _openai_chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ], model=model)
    raw = content.strip().upper().split()
    pick = "A" if (raw and raw[0].startswith("A")) else ("B" if (raw and raw[0].startswith("B")) else "A")
    # Map back if we swapped A/B order
    if a_pos == 1:
        # our displayed A was original B
        pick = "B" if pick == "A" else "A"
    return JudgeDecision(pick, "llm judge", {"provider": provider, "raw": content})


def oracle_compare(inp: str, out_a: str, out_b: str, label: Optional[str]) -> JudgeDecision:
    if label is None or str(label).strip() == "":
        return JudgeDecision("tie", "no-label", {})
    label_s = str(label).strip().lower()
    sa = label_s in str(out_a).strip().lower()
    sb = label_s in str(out_b).strip().lower()
    if sa and not sb:
        return JudgeDecision("A", "oracle contains label", {})
    if sb and not sa:
        return JudgeDecision("B", "oracle contains label", {})
    return JudgeDecision("tie", "oracle tie", {})


def hybrid_judge(inp: str, out_a: str, out_b: str, label: Optional[str], label_available: bool,
                 llm_provider: str = "openai", llm_model: str = "gpt-4o-mini") -> JudgeDecision:
    if label_available and label is not None:
        dec = oracle_compare(inp, out_a, out_b, label)
        if dec.winner != "tie":
            return dec
    return llm_pairwise_judge(inp, out_a, out_b, provider=llm_provider, model=llm_model)
