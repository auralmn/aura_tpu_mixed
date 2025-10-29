#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import json
from typing import List, Tuple
import numpy as np

# Emotion order: [joy, sadness, anger, fear, disgust, surprise, trust, anticipation]
_LEXICON = {
    'joy': [
        'happy', 'joy', 'delight', 'smile', 'cheer', 'glad', 'pleased', 'excited', 'grateful', 'love'
    ],
    'sadness': [
        'sad', 'sorrow', 'grief', 'down', 'depressed', 'lonely', 'cry', 'tears'
    ],
    'anger': [
        'angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'hate'
    ],
    'fear': [
        'fear', 'scared', 'afraid', 'terrified', 'anxious', 'panic', 'worry'
    ],
    'disgust': [
        'disgust', 'gross', 'repulsed', 'nausea', 'sick', 'yuck'
    ],
    'surprise': [
        'surprised', 'astonished', 'amazed', 'wow', 'unexpected', 'suddenly'
    ],
    'trust': [
        'trust', 'rely', 'depend', 'faith', 'confidence', 'assure'
    ],
    'anticipation': [
        'anticipate', 'await', 'expect', 'hope', 'plan', 'prepare'
    ],
}
_ORDER = ['joy','sadness','anger','fear','disgust','surprise','trust','anticipation']


def _affect_vector(text: str) -> np.ndarray:
    t = text.lower()
    counts = []
    for key in _ORDER:
        words = _LEXICON[key]
        c = sum(t.count(w) for w in words)
        counts.append(c)
    vec = np.array(counts, dtype=np.float32)
    if vec.sum() > 0:
        vec = vec / (vec.sum() + 1e-6)
    return vec


def load_text_corpus(txt_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Load all .txt files in txt_dir and return list of (text, affect_vec[8])."""
    items: List[Tuple[str, np.ndarray]] = []
    if not os.path.isdir(txt_dir):
        return items
    for name in os.listdir(txt_dir):
        if not name.lower().endswith('.txt'):
            continue
        p = os.path.join(txt_dir, name)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                if not text:
                    continue
                vec = _affect_vector(text)
                items.append((text, vec))
        except Exception:
            continue
    return items


def load_json_corpus(json_dir: str, text_keys: List[str] = None) -> List[Tuple[str, np.ndarray]]:
    items: List[Tuple[str, np.ndarray]] = []
    if not os.path.isdir(json_dir):
        return items
    keys = text_keys or ['text', 'content', 'body', 'message']
    def _extract(obj):
        if isinstance(obj, dict):
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v
        if isinstance(obj, str) and obj.strip():
            return obj
        return None
    for name in os.listdir(json_dir):
        low = name.lower()
        if not (low.endswith('.json') or low.endswith('.jsonl')):
            continue
        p = os.path.join(json_dir, name)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                if low.endswith('.jsonl'):
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            continue
                        t = _extract(obj)
                        if t:
                            vec = _affect_vector(t)
                            items.append((t, vec))
                else:
                    try:
                        data = json.load(f)
                    except Exception:
                        continue
                    def visit(node):
                        if isinstance(node, list):
                            for x in node:
                                visit(x)
                        elif isinstance(node, dict):
                            t = _extract(node)
                            if t:
                                vec = _affect_vector(t)
                                items.append((t, vec))
                            else:
                                for v in node.values():
                                    if isinstance(v, (list, dict, str)):
                                        visit(v)
                        elif isinstance(node, str):
                            s = node.strip()
                            if s:
                                vec = _affect_vector(s)
                                items.append((s, vec))
                    visit(data)
        except Exception:
            continue
    return items


def load_text_corpus_all(txt_dir: str, json_dir: str) -> List[Tuple[str, np.ndarray]]:
    items = []
    items.extend(load_text_corpus(txt_dir))
    items.extend(load_json_corpus(json_dir))
    return items
