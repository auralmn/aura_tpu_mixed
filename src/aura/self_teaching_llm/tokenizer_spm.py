#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple
import argparse

try:
    import sentencepiece as spm
    _SPM_AVAILABLE = True
except Exception:
    spm = None
    _SPM_AVAILABLE = False


class SPMTokenizer:
    def __init__(self, model_path: Optional[str] = None):
        self.proc = None
        self.model_path = None
        if model_path is not None:
            self.load(model_path)

    @staticmethod
    def train_from_dir(
        input_dir: str,
        model_prefix: str,
        vocab_size: int = 2000,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        pad_id: int = 3,
        user_defined_symbols: Optional[List[str]] = None,
        max_sentence_length: int = 1000000,
        hard_vocab_limit: bool = False,
        byte_fallback: bool = False,
        train_extremely_large_corpus: bool = False,
        clean_controls: bool = True,
        normalize_spaces: bool = True,
        use_iterator: bool = False,
        ascii_only: bool = False,
    ) -> str:
        """Train a SentencePiece model from all .txt files in input_dir.
        Returns path to the generated model (.model).
        """
        if not _SPM_AVAILABLE:
            raise ImportError("sentencepiece is not installed. Please `pip install sentencepiece`. ")
        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
        # Prepare cleaners
        ctrl_re = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")  # keep \n and \t
        def _clean_line(s: str) -> str:
            if not s:
                return s
            # Remove nulls and other control chars (except \n/\t)
            if clean_controls:
                s = ctrl_re.sub(" ", s)
            # Keep ASCII printable + tab only (optional)
            if ascii_only:
                s = re.sub(r'[^\x09\x20-\x7E]', ' ', s)
            # Normalize internal whitespace to single space (preserve leading/trailing minimal)
            if normalize_spaces:
                s = " ".join(s.split())
            return s
        # Stream all text files into a single corpus, cleaning as we go
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input dir not found: {input_dir}")
        txt_paths = [os.path.join(input_dir, n) for n in os.listdir(input_dir) if n.lower().endswith('.txt')]
        if not txt_paths:
            raise ValueError(f"No .txt files found in {input_dir}")
        # Iterator to yield cleaned, chunked lines
        chunk = 2000
        def _iter_lines():
            for p in txt_paths:
                try:
                    with open(p, 'rb') as bf:
                        raw = bf.read()
                    t = raw.decode('utf-8', errors='ignore')
                except Exception:
                    continue
                if not t:
                    continue
                for line in t.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    s = _clean_line(s)
                    if not s:
                        continue
                    while len(s) > chunk:
                        yield s[:chunk]
                        s = s[chunk:]
                    if s:
                        yield s
        # Train either from iterator or from a written corpus file
        # Note: <pad>, <bos>, <eos>, <unk> are built-in control tokens
        # and should NOT be in user_defined_symbols
        if user_defined_symbols is None:
            uds = []
        else:
            # Filter out built-in control tokens and de-duplicate while preserving order
            built_in_tokens = {"<pad>", "<bos>", "<eos>", "<unk>", "<s>", "</s>"}
            seen = set()
            uds = []
            for s in list(user_defined_symbols):
                if s not in seen and s not in built_in_tokens:
                    seen.add(s)
                    uds.append(s)
        uds_arg = ",".join(uds) if uds else ""
        if use_iterator:
            spm.SentencePieceTrainer.train(
                sentence_iterator=_iter_lines(),
                model_prefix=model_prefix,
                vocab_size=int(vocab_size),
                model_type=model_type,
                character_coverage=float(character_coverage),
                pad_id=int(pad_id),
                user_defined_symbols=uds_arg,
                max_sentence_length=int(max_sentence_length),
                hard_vocab_limit=hard_vocab_limit,
                byte_fallback=byte_fallback,
                train_extremely_large_corpus=train_extremely_large_corpus,
            )
        else:
            corpus_path = model_prefix + ".corpus.txt"
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for s in _iter_lines():
                    f.write(s + "\n")
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=int(vocab_size),
                model_type=model_type,
                character_coverage=float(character_coverage),
                pad_id=int(pad_id),
                user_defined_symbols=uds_arg,
                max_sentence_length=int(max_sentence_length),
                hard_vocab_limit=hard_vocab_limit,
                byte_fallback=byte_fallback,
                train_extremely_large_corpus=train_extremely_large_corpus,
            )
        return model_prefix + ".model"

    def load(self, model_path: str) -> None:
        if not _SPM_AVAILABLE:
            raise ImportError("sentencepiece is not installed. Please `pip install sentencepiece`. ")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.proc = spm.SentencePieceProcessor()
        self.proc.load(model_path)
        self.model_path = model_path

    # Encoding/decoding
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = self.proc.encode(text, out_type=int)
        if add_bos and self.proc.bos_id() >= 0:
            ids = [self.proc.bos_id()] + ids
        if add_eos and self.proc.eos_id() >= 0:
            ids = ids + [self.proc.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.proc.decode(ids)

    def encode_batch(
        self, texts: List[str], max_len: int = 64, pad_to: Optional[int] = 64,
        add_bos: bool = True, add_eos: bool = True
    ) -> Tuple[List[List[int]], int]:
        batch: List[List[int]] = []
        for t in texts:
            ids = self.encode(t, add_bos=add_bos, add_eos=add_eos)
            ids = ids[:max_len]
            batch.append(ids)
        pad_id = self.proc.pad_id() if self.proc.pad_id() >= 0 else 0
        if pad_to is not None:
            padded: List[List[int]] = []
            for ids in batch:
                if len(ids) < pad_to:
                    ids = ids + [pad_id] * (pad_to - len(ids))
                padded.append(ids)
            batch = padded
        return batch, pad_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer from a text directory")
    parser.add_argument("--input_dir", type=str, default="data/txt", help="Directory of .txt files")
    parser.add_argument("--out_dir", type=str, default="models/spm", help="Output directory for model")
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram","bpe"]) 
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--pad_id", type=int, default=3)
    parser.add_argument("--max_sentence_length", type=int, default=1000000)
    parser.add_argument("--hard_vocab_limit", type=int, default=0, help="0=false, 1=true")
    parser.add_argument("--byte_fallback", type=int, default=0, help="0=false, 1=true")
    parser.add_argument("--clean_controls", type=int, default=1, help="Remove null/control characters")
    parser.add_argument("--normalize_spaces", type=int, default=1, help="Collapse internal whitespace")
    parser.add_argument("--use_iterator", type=int, default=0, help="Stream sentences via iterator instead of corpus file")
    parser.add_argument("--user_symbols", type=str, default="", help="Comma-separated user-defined symbols, e.g. <INST>,<INP>,<RESP>,<SEP>")
    parser.add_argument("--ascii_only", type=int, default=0, help="Filter to ASCII printable + tab only")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, "spiece")
    model_path = SPMTokenizer.train_from_dir(
        input_dir=args.input_dir,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=args.pad_id,
        max_sentence_length=args.max_sentence_length,
        hard_vocab_limit=bool(args.hard_vocab_limit),
        byte_fallback=bool(args.byte_fallback),
        clean_controls=bool(args.clean_controls),
        normalize_spaces=bool(args.normalize_spaces),
        use_iterator=bool(args.use_iterator),
        ascii_only=bool(args.ascii_only, default=True),
        user_defined_symbols=[s for s in (args.user_symbols.split(',') if args.user_symbols else []) if s],
    )
    print(f"Trained SentencePiece model saved to: {model_path}")
