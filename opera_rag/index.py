"""Hybrid retrieval: BM25 (always on) + dense embeddings (when available),
fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .loaders import Chunk

_STOP = set("""a an the and or of to in on for with by is are was were be been this that it
as at from how what which who where when why do does did can could should would i we you
into use used using via its their there""".split())


def tokenize(text: str) -> list[str]:
    """Code-aware tokens: keeps snake_case identifiers whole AND split,
    splits camelCase, lowercases, drops stopwords."""
    toks = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", text):
        low = word.lower()
        parts = [p for p in re.split(r"_+", word) if p]
        sub = []
        for p in parts:
            sub += re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", p)
        sub = [s.lower() for s in sub]
        if len(sub) > 1:
            toks.append(low.strip("_"))
        toks += sub or [low]
    return [t for t in toks if t not in _STOP and len(t) > 1]


def index_text(c: Chunk) -> str:
    m = c.meta or {}
    extra = " ".join(str(m.get(k, "")) for k in ("signature", "docstring", "context"))
    return f"{c.path} {c.name} {c.kind} {extra}\n{c.text}"


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class Embedder:
    """Thin wrapper around sentence-transformers. Returns None if unavailable
    (no package, no GPU needed, but the model must be downloadable)."""
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)

    @classmethod
    def try_load(cls, model_name: str | None):
        if not model_name:
            return None
        try:
            return cls(model_name)
        except Exception as e:
            print(f"  (dense embeddings off — {type(e).__name__}: {str(e)[:120]}) → keyword search only")
            return None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], repo_map: str = "", embed_model: str | None = None,
                 embedder: Embedder | None = None):
        self.chunks = chunks
        self.repo_map = repo_map
        self.bm25 = BM25([tokenize(index_text(c)) for c in chunks])
        self.embed_model = embed_model
        self.embedder = embedder if embedder is not None else Embedder.try_load(embed_model)
        self.vectors = None
        if self.embedder is not None and chunks:
            self.vectors = self.embedder.encode([index_text(c)[:2000] for c in chunks])

    # --------------------------------------------------------------- search
    def search(self, query: str, k: int = 8, rrf_k: int = 60, path_filter: str | None = None):
        n = len(self.chunks)
        if n == 0:
            return []
        rankings = []
        bm = self.bm25.scores(tokenize(query))
        rankings.append(np.argsort(-bm))
        dense = None
        if self.vectors is not None:
            q = self.embedder.encode([query])[0]
            dense = self.vectors @ q
            rankings.append(np.argsort(-dense))
        fused = np.zeros(n)
        for order in rankings:
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (rrf_k + rank + 1)
        # BM25 rows with zero score shouldn't get credit from the lexical list
        if dense is None:
            fused[bm <= 0] = 0
        # direct symbol / filename mention → strong boost (e.g. "what does interpolate_coordinates do")
        ql = query.lower()
        for i, c in enumerate(self.chunks):
            base = c.name.split(" ")[0].lower()
            if len(base) > 3 and base in ql:
                fused[i] += 0.02
            if Path(c.path).stem.lower() in ql:
                fused[i] += 0.01
        order = np.argsort(-fused)
        out = []
        for i in order:
            if fused[i] <= 0 or len(out) >= k:
                break
            c = self.chunks[i]
            if path_filter and path_filter not in c.path:
                continue
            out.append((c, float(fused[i]),
                        {"bm25": float(bm[i]), "dense": float(dense[i]) if dense is not None else None}))
        return out

    # --------------------------------------------------------------- persistence
    def save(self, folder: str | Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (folder / "repo_map.txt").write_text(self.repo_map, encoding="utf-8")
        (folder / "config.json").write_text(json.dumps({"embed_model": self.embed_model if self.vectors is not None else None}))
        if self.vectors is not None:
            np.save(folder / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, folder: str | Path):
        folder = Path(folder)
        chunks = [Chunk(**json.loads(l)) for l in open(folder / "chunks.jsonl", encoding="utf-8")]
        cfg = json.loads((folder / "config.json").read_text())
        obj = cls.__new__(cls)
        obj.chunks = chunks
        obj.repo_map = (folder / "repo_map.txt").read_text(encoding="utf-8")
        obj.bm25 = BM25([tokenize(index_text(c)) for c in chunks])
        obj.embed_model = cfg.get("embed_model")
        obj.vectors, obj.embedder = None, None
        if obj.embed_model and (folder / "vectors.npy").exists():
            obj.embedder = Embedder.try_load(obj.embed_model)
            if obj.embedder is not None:
                obj.vectors = np.load(folder / "vectors.npy")
        return obj
