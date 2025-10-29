#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import json
import math
import random
import time
from typing import List, Dict, Any, Optional, Tuple

import requests

from .judges import JudgeDecision, hybrid_judge, llm_pairwise_judge


def _openai_chat(messages, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for OpenAI generation")
    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


class OpenSearchRetriever:
    def __init__(self, url: str, index: str, topk: int = 5):
        self.base = url.rstrip("/")
        self.index = index
        self.topk = topk

    def search(self, q: str) -> List[str]:
        try:
            body = {"size": self.topk, "query": {"match": {"text": q}}}
            r = requests.get(f"{self.base}/{self.index}/_search", headers={"Content-Type": "application/json"}, data=json.dumps(body), timeout=30)
            r.raise_for_status()
            js = r.json()
            hits = js.get("hits", {}).get("hits", [])
            out = []
            for h in hits:
                s = h.get("_source", {})
                t = s.get("text")
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
            return out
        except Exception:
            return []


class PromptDueler:
    def __init__(self,
                 prompts: List[str],
                 data: List[Dict[str, Any]],
                 judge_provider: str = "openai",
                 judge_model: str = "gpt-4o-mini",
                 gen_provider: str = "openai",
                 gen_model: str = "gpt-4o-mini",
                 label_fraction: float = 0.0,
                 mutate_period: int = 10,
                 alpha: float = 1.2,
                 retriever: Optional[OpenSearchRetriever] = None):
        self.prompts = list(prompts)
        self.data = list(data)
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.gen_provider = gen_provider
        self.gen_model = gen_model
        self.label_fraction = float(label_fraction)
        self.mutate_period = int(mutate_period)
        self.alpha = float(alpha)
        self.retriever = retriever
        k = len(self.prompts)
        self.W = [[0 for _ in range(k)] for _ in range(k)]
        self.N = [[0 for _ in range(k)] for _ in range(k)]
        self.cache: Dict[Tuple[int, int], str] = {}
        self.t = 1
        self.label_mask = self._make_label_mask()

    def _make_label_mask(self) -> List[bool]:
        n = len(self.data)
        idx = list(range(n))
        random.shuffle(idx)
        m = int(self.label_fraction * n)
        use = set(idx[:m])
        mask = [i in use and ("label" in self.data[i] and str(self.data[i]["label"]).strip() != "") for i in range(n)]
        return mask

    def _build_input(self, ex: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        label = ex.get("label")
        q = ex.get("query") or ex.get("input") or ""
        if self.retriever and q:
            ctxs = self.retriever.search(str(q))
            if ctxs:
                ctx = "\n".join(f"- {c}" for c in ctxs[: self.retriever.topk])
                inp = f"Question:\n{q}\n\nContext:\n{ctx}"
                return inp, label
        inp = ex.get("input") or q
        return str(inp or ""), label

    def _gen(self, prompt: str, inp: str) -> str:
        if self.gen_provider == "echo":
            return f"{prompt}\n\n{inp}\n\nAnswer: {inp[:256]}"
        sys = prompt
        user = inp
        return _openai_chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ], model=self.gen_model)

    def _get_output(self, p_idx: int, ex_idx: int) -> str:
        key = (p_idx, ex_idx)
        if key in self.cache:
            return self.cache[key]
        ex = self.data[ex_idx]
        inp, _ = self._build_input(ex)
        out = self._gen(self.prompts[p_idx], inp)
        self.cache[key] = out
        return out

    def _ucb_lcb(self, i: int, j: int) -> Tuple[float, float]:
        w = self.W[i][j]
        n = self.N[i][j]
        if n <= 0:
            return 1.0, 0.0
        p = w / n
        bonus = self.alpha * math.log(max(2, self.t)) / max(1, n)
        return min(1.0, p + bonus), max(0.0, p - bonus)

    def _optimistic_copeland(self) -> Tuple[List[int], List[int]]:
        k = len(self.prompts)
        scores = [0 for _ in range(k)]
        for i in range(k):
            s = 0
            for j in range(k):
                if i == j:
                    continue
                u, _ = self._ucb_lcb(i, j)
                if u >= 0.5:
                    s += 1
            scores[i] = s
        m = max(scores) if scores else 0
        cand = [i for i, s in enumerate(scores) if s == m]
        return cand, scores

    def _sample_beta(self, a: int, b: int) -> float:
        # Simple approximation via random.gammavariate
        x = random.gammavariate(a, 1.0)
        y = random.gammavariate(b, 1.0)
        return x / (x + y) if (x + y) > 0 else 0.5

    def _select_pair(self) -> Tuple[int, int]:
        k = len(self.prompts)
        cand, _ = self._optimistic_copeland()
        if not cand:
            cand = list(range(k))
        best_i = None
        best_s = -1
        for i in cand:
            s = 0
            for j in range(k):
                if i == j:
                    continue
                a = self.W[i][j] + 1
                b = self.W[j][i] + 1
                th = self._sample_beta(a, b)
                if th >= 0.5:
                    s += 1
            if s > best_s:
                best_s = s
                best_i = i
        i_star = best_i if best_i is not None else random.randrange(k)
        S = []
        for j in range(k):
            if j == i_star:
                continue
            _, l = self._ucb_lcb(i_star, j)
            if l <= 0.5:
                S.append(j)
        if not S:
            S = [j for j in range(k) if j != i_star]
        best_j = None
        best_theta = -1.0
        for j in S:
            a = self.W[j][i_star] + 1
            b = self.W[i_star][j] + 1
            th = self._sample_beta(a, b)
            if th > best_theta:
                best_theta = th
                best_j = j
        j_star = best_j if best_j is not None else random.choice(S)
        return i_star, j_star

    def _mutate(self) -> None:
        if not self.prompts:
            return
        i_best = self.best_index()
        base = self.prompts[i_best]
        tips = [
            "Be concise.",
            "Use step-by-step reasoning.",
            "Prefer factual statements and avoid speculation.",
            "Answer directly before providing details.",
            "Use bullet points when listing multiple items.",
        ]
        tip = random.choice(tips)
        newp = base + "\n\n" + tip
        self._add_prompt(newp)

    def _add_prompt(self, p: str) -> None:
        self.prompts.append(p)
        k = len(self.prompts)
        for r in self.W:
            r.append(0)
        for r in self.N:
            r.append(0)
        self.W.append([0 for _ in range(k)])
        self.N.append([0 for _ in range(k)])

    def best_index(self) -> int:
        k = len(self.prompts)
        scores = [0 for _ in range(k)]
        wins = [0.0 for _ in range(k)]
        for i in range(k):
            s = 0
            wsum = 0.0
            for j in range(k):
                if i == j:
                    continue
                n = max(1, self.N[i][j])
                p = self.W[i][j] / n
                if p > 0.5:
                    s += 1
                wsum += p
            scores[i] = s
            wins[i] = wsum
        m = max(scores)
        cand = [i for i, s in enumerate(scores) if s == m]
        if len(cand) == 1:
            return cand[0]
        return max(cand, key=lambda idx: wins[idx])

    def duel_once(self) -> Dict[str, Any]:
        i, j = self._select_pair()
        ex_idx = random.randrange(len(self.data))
        ex = self.data[ex_idx]
        inp, label = self._build_input(ex)
        out_i = self._get_output(i, ex_idx)
        out_j = self._get_output(j, ex_idx)
        use_label = self.label_mask[ex_idx]
        if self.judge_provider == "hybrid":
            dec = hybrid_judge(inp, out_i, out_j, label, use_label, llm_provider="openai", llm_model=self.judge_model)
        else:
            dec = llm_pairwise_judge(inp, out_i, out_j, provider=self.judge_provider, model=self.judge_model)
        if dec.winner == "A":
            self.W[i][j] += 1
        elif dec.winner == "B":
            self.W[j][i] += 1
        else:
            self.W[i][j] += 1
            self.W[j][i] += 1
        self.N[i][j] += 1
        self.N[j][i] += 1
        self.t += 1
        return {
            "i": i,
            "j": j,
            "ex_idx": ex_idx,
            "winner": dec.winner,
            "reason": dec.reason,
            "meta": dec.meta,
        }

    def run(self, rounds: int, duels_per_round: int, mutate: bool = True) -> Dict[str, Any]:
        logs = []
        for r in range(1, rounds + 1):
            for _ in range(duels_per_round):
                logs.append(self.duel_once())
            if mutate and self.mutate_period > 0 and (r % self.mutate_period == 0):
                self._mutate()
        bi = self.best_index()
        return {
            "best_index": bi,
            "best_prompt": self.prompts[bi],
            "W": self.W,
            "N": self.N,
            "logs": logs,
        }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                out.append(json.loads(line))
            else:
                out.append({"prompt": line})
    return out


def load_candidates(path: str) -> List[str]:
    rows = load_jsonl(path)
    prompts = []
    for r in rows:
        p = r.get("prompt") if isinstance(r, dict) else None
        if isinstance(p, str) and p.strip():
            prompts.append(p.strip())
    return prompts


def load_data(path: str) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    out = []
    for r in rows:
        inp = r.get("input") or r.get("question") or r.get("query") or r.get("text") or ""
        lbl = r.get("label") or r.get("answer") or None
        q = r.get("query") or None
        out.append({"input": str(inp), "label": (str(lbl) if lbl is not None else None), "query": (str(q) if q is not None else None)})
    return out
