import time
import threading
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path

try:
    import faiss  # type: ignore
    _FAISS = True
except Exception:
    _FAISS = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST = True
except Exception:
    _ST = False

try:
    import requests  # type: ignore
    _REQ = True
except Exception:
    _REQ = False

try:
    import feedparser  # type: ignore
    _FEED = True
except Exception:
    _FEED = False


class NewsIngestor:
    def __init__(self, sources: Dict[str, str], vector_dim: int = 384, db_path: str = "aura_news.db"):
        self.sources = sources
        self.seen = set()
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') if _ST else None
        self.index = faiss.IndexFlatIP(vector_dim) if _FAISS else None
        self.metadata: List[Tuple[float, str, str, str]] = []
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._lock = threading.Lock()
        # Optional keyword categories from war_keywords.json
        self.keyword_categories: Dict[str, List[str]] = {}
        try:
            kw_path = Path("war_keywords.json")
            if kw_path.exists():
                with kw_path.open('r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Expect {category: [keywords, ...]}
                    self.keyword_categories = {str(k).lower(): [str(x).lower() for x in v] for k, v in data.items() if isinstance(v, list)}
                elif isinstance(data, list):
                    self.keyword_categories = {"conflict": [str(x).lower() for x in data]}
        except Exception:
            self.keyword_categories = {}

    def _init_db(self):
        c = self.db.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                source TEXT,
                category TEXT,
                url TEXT,
                title TEXT,
                summary TEXT,
                embedding BLOB
            )
        ''')
        self.db.commit()

    def _embed(self, text: str):
        if not self.embedder:
            return None
        return self.embedder.encode(text)

    def classify_category(self, text: str) -> str:
        tl = text.lower()
        # Keyword-driven categories from war_keywords.json (if available)
        for cat, keywords in self.keyword_categories.items():
            try:
                if any(kw in tl for kw in keywords):
                    return cat
            except Exception:
                continue
        if 'economy' in tl or 'market' in tl: return 'finance'
        if 'election' in tl or 'senate' in tl or 'president' in tl: return 'politics'
        if 'science' in tl or 'physics' in tl or 'biology' in tl: return 'science'
        if 'ai' in tl or 'tech' in tl or 'software' in tl: return 'technology'
        return 'general'

    def _upsert(self, ts: float, source: str, category: str, url: str, title: str, summary: str, emb) -> None:
        with self._lock:
            if url in self.seen:
                return
            self.seen.add(url)
            if self.index is not None and emb is not None:
                self.index.add(emb.reshape(1, -1))
                self.metadata.append((ts, source, category, url))
            c = self.db.cursor()
            c.execute(
                "INSERT INTO news(timestamp, source, category, url, title, summary, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, source, category, url, title, summary, emb.tobytes() if emb is not None else b'')
            )
            self.db.commit()

    def _ingest_rss_once(self) -> None:
        if not _FEED:
            return
        for name, url in self.sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    link = getattr(entry, 'link', None)
                    title = getattr(entry, 'title', '')
                    summary = getattr(entry, 'summary', '')
                    if not link or link in self.seen:
                        continue
                    text = f"{title}. {summary}".strip()
                    emb = self._embed(text)
                    category = self.classify_category(text)
                    self._upsert(time.time(), name, category, link, title, summary, emb)
            except Exception:
                continue

    def ingest_loop(self, poll_interval: float = 60.0):
        while True:
            try:
                self._ingest_rss_once()
            except Exception:
                pass
            time.sleep(poll_interval)

    def query_similar(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.embedder is None:
            return []
        q_emb = self.embedder.encode(query_text).reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.metadata):
                continue
            ts, src, cat, url = self.metadata[idx]
            results.append({'timestamp': ts, 'source': src, 'category': cat, 'url': url})
        return results


class BraveProIngestor:
    def __init__(self, api_key: str, news_ingestor: NewsIngestor):
        self.api_key = api_key
        self.news_ingestor = news_ingestor
        self.session = requests.Session() if _REQ else None
        if self.session is not None:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })

    def fetch_latest(self) -> None:
        if self.session is None:
            return
        url = "https://api.brave.ai/v1/news/latest"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("data", [])
        for article in items:
            link = article.get("url")
            if not link or link in self.news_ingestor.seen:
                continue
            title = article.get("title", "").strip()
            summary = article.get("summary", "").strip()
            text = f"{title}. {summary}"
            emb = self.news_ingestor._embed(text)
            category = article.get("category", "general")
            self.news_ingestor._upsert(time.time(), "BraveProAI", category, link, title, summary, emb)

    def ingest_loop(self, interval: int = 120):
        while True:
            try:
                self.fetch_latest()
            except Exception:
                pass
            time.sleep(interval)


