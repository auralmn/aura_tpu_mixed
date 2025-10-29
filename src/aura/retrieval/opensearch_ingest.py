#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
import time
from typing import Iterator, Dict, Any

try:
    import requests  # lightweight HTTP client
except Exception as e:
    print("[ingest] Please `pip install requests`", file=sys.stderr)
    raise


def create_index(host: str, index: str, force: bool = False) -> None:
    # Delete if exists
    if force:
        try:
            requests.delete(f"{host}/{index}")
        except Exception:
            pass
    # Create single-node dev index with BM25
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "similarity": {
                    "default": {"type": "BM25", "k1": 0.9, "b": 0.4}
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                "qid": {"type": "keyword"},
                "pid": {"type": "keyword"},
                "url": {"type": "keyword"},
                "source": {"type": "keyword"}
            }
        }
    }
    r = requests.put(f"{host}/{index}", headers={"Content-Type": "application/json"}, data=json.dumps(settings))
    if r.status_code not in (200, 201):
        print(f"[ingest] index create status={r.status_code} body={r.text}", file=sys.stderr)
        # Continue if already exists


def rows_ms_marco_v21(dataset: str, config: str, split: str) -> Iterator[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception:
        print("[ingest] Please `pip install datasets`", file=sys.stderr)
        raise
    ds = load_dataset(dataset, config, split=split, streaming=True)
    for ex in ds:
        yield ex


def gen_docs_from_ms_marco(ex: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    qid = ex.get("query_id")
    passages = (ex.get("passages") or {})
    texts = passages.get("passage_text") or []
    urls = passages.get("url") or []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        doc = {
            "text": s,
            "qid": f"{qid}" if qid is not None else None,
            "pid": f"p{i}",
            "url": urls[i] if i < len(urls) else None,
            "source": "msmarco_v2.1"
        }
        yield doc


def bulk_index(host: str, index: str, docs: Iterator[Dict[str, Any]], batch: int = 1000, refresh: bool = False) -> int:
    url = f"{host}/{index}/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}
    cnt = 0
    buf = []
    def flush(buf):
        if not buf:
            return 0
        data = "".join(buf)
        r = requests.post(url, headers=headers, data=data)
        if r.status_code != 200:
            print(f"[ingest] bulk status={r.status_code} body={r.text[:1000]}", file=sys.stderr)
        else:
            resp = r.json()
            if resp.get("errors"):
                # Print first error
                for it in resp.get("items", []):
                    if it.get("index", {}).get("error"):
                        print(f"[ingest] bulk error: {it['index']['error']}", file=sys.stderr)
                        break
        return len(buf) // 2
    for d in docs:
        meta = {"index": {"_index": index}}
        buf.append(json.dumps(meta) + "\n")
        buf.append(json.dumps(d, ensure_ascii=False) + "\n")
        if len(buf) >= 2 * batch:
            cnt += flush(buf)
            buf = []
    if buf:
        cnt += flush(buf)
    if refresh:
        try:
            requests.post(f"{host}/{index}/_refresh")
        except Exception:
            pass
    return cnt


def main():
    ap = argparse.ArgumentParser(description="Ingest MS MARCO v2.1 passages into OpenSearch via bulk API")
    ap.add_argument("--host", type=str, default="http://127.0.0.1:9200")
    ap.add_argument("--index", type=str, default="msmarco-passages")
    ap.add_argument("--dataset", type=str, default="microsoft/ms_marco")
    ap.add_argument("--config", type=str, default="v2.1")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--max_rows", type=int, default=0, help="Max MS MARCO rows to pull (0=all)")
    ap.add_argument("--max_docs", type=int, default=0, help="Max documents to index (0=all)")
    ap.add_argument("--force", action="store_true", default=False, help="Delete index if exists")
    ap.add_argument("--refresh", action="store_true", default=False)
    args = ap.parse_args()

    print(f"[ingest] host={args.host} index={args.index}")
    create_index(args.host, args.index, force=args.force)

    row_iter = rows_ms_marco_v21(args.dataset, args.config, args.split)
    total_docs = 0
    total_rows = 0
    t0 = time.time()

    def doc_gen():
        nonlocal total_docs, total_rows
        for row in row_iter:
            total_rows += 1
            for d in gen_docs_from_ms_marco(row):
                yield d
                total_docs += 1
                if args.max_docs and total_docs >= args.max_docs:
                    return
            if args.max_rows and total_rows >= args.max_rows:
                return

    cnt = bulk_index(args.host, args.index, doc_gen(), batch=args.batch, refresh=args.refresh)
    dt = time.time() - t0
    print(f"[ingest] indexed_docs={cnt} rows_read={total_rows} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
