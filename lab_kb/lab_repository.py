from __future__ import annotations
import os, json, glob, re, io
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# --------- 小工具 ---------
def _read_text_file(path: str) -> str:
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf_file(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""

def _read_csv_file(path: str) -> str:
    # 簡單讀（若需要可改用 pandas）
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _clean_text(s: str, strip_whitespace=True, collapse_newlines=True) -> str:
    if strip_whitespace:
        s = s.strip()
    if collapse_newlines:
        s = re.sub(r"\n{2,}", "\n", s)
    return s

def _chunk(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i:i+size]
        if not chunk.strip():
            break
        chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks

# --------- 設定 ---------
@dataclass
class LabRepoConfig:
    raw_dir: str
    store_dir: str
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    normalize_embeddings: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_score: float = 0.30
    include_ext: Tuple[str, ...] = (".txt", ".md", ".pdf", ".csv")
    use_faiss: bool = True
    strip_whitespace: bool = True
    collapse_newlines: bool = True

# --------- 主類別 ---------
class LabRepository:
    """
    與 memory_repository 同級的「實驗室專屬知識庫」。
    - 以段落向量檢索（Dual-tower embedding）
    - 支援 FAISS 或 sklearn NN
    - 可持久化與增量更新
    介面重點：
    - build() / add_paths(paths)   : 建索引或增量加入
    - retrieve(query, top_k, min_score) -> List[dict]

    設計邊界：
    - 本類別是 LAB_KB 的底層 repository / indexing backend。
    - Agent runtime 與 feature-level retrieval policy 應統一收束到 features/labkb.py。
    - retrieve_structured() / as_context() 僅保留給維護、除錯或離線檢查使用，
      不建議上層直接依賴這兩個方法作為正式 feature API。
    """
    def __init__(self, cfg: LabRepoConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.store_dir, exist_ok=True)
        self.model = SentenceTransformer(self.cfg.embedding_model)
        self._embeds: Optional[np.ndarray] = None
        self._meta: List[Dict[str, Any]] = []   # 每個段的 {path, chunk_id, text}
        self._index = None
        self._use_faiss = (self.cfg.use_faiss and _HAS_FAISS)
        self._load_if_exists()

    # ---------- 檔案收集與分塊 ----------
    def _iter_raw_files(self) -> List[str]:
        files = []
        for ext in self.cfg.include_ext:
            files.extend(glob.glob(os.path.join(self.cfg.raw_dir, f"**/*{ext}"), recursive=True))
        return sorted(set(files))

    def iter_raw_files(self) -> List[str]:
        """Public helper for build/maintenance scripts. Runtime feature code should use features/labkb.py instead."""
        return self._iter_raw_files()

    def indexed_source_paths(self) -> set[str]:
        """Return raw source paths already present in the repository index."""
        return {str(m.get("path")) for m in self._meta if m.get("path")}

    def _read_file(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".txt", ".md"):
            return _read_text_file(path)
        if ext == ".pdf":
            return _read_pdf_file(path)
        if ext == ".csv":
            return _read_csv_file(path)
        return ""

    def _ingest_paths(self, paths: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        all_chunks, metas = [], []
        for p in paths:
            raw = self._read_file(p)
            if not raw: 
                continue
            raw = _clean_text(raw, self.cfg.strip_whitespace, self.cfg.collapse_newlines)
            chunks = _chunk(raw, self.cfg.chunk_size, self.cfg.chunk_overlap)
            for i, ch in enumerate(chunks):
                metas.append({"path": p, "chunk_id": i})
                all_chunks.append(ch)
        return all_chunks, metas

    # ---------- 向量與索引 ----------
    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=self.cfg.normalize_embeddings)
        return np.asarray(embs, dtype="float32")

    def _build_index(self, X: np.ndarray):
        if self._use_faiss:
            dim = X.shape[1]
            index = faiss.IndexFlatIP(dim)  # 內積 = cosine（已正規化）
            index.add(X)
            self._index = index
        else:
            nn = NearestNeighbors(n_neighbors=min(10, max(1, self.cfg.top_k*3)), metric="cosine")
            nn.fit(X)
            self._index = nn

    def _search(self, qvec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._use_faiss:
            sims, idx = self._index.search(qvec, top_k)
            return idx[0], sims[0]
        else:
            # sklearn NN 需要距離 -> 相似度 = 1 - 距離
            dist, idx = self._index.kneighbors(qvec, n_neighbors=top_k, return_distance=True)
            sims = 1.0 - dist
            return idx[0], sims[0]

    # ---------- Public API ----------
    def build(self):
        paths = self._iter_raw_files()
        chunks, metas = self._ingest_paths(paths)
        if not chunks:
            raise RuntimeError("lab_repository: 找不到可用原始檔，請放資料到 raw_dir")
        X = self._embed(chunks)
        self._embeds = X
        self._meta = [{"text": t, **m} for t, m in zip(chunks, metas)]
        self._build_index(X)
        self._persist()

    def add_paths(self, paths: List[str]):
        # 增量加入
        if self._embeds is None or self._index is None:
            return self.build()

        chunks, metas = self._ingest_paths(paths)
        if not chunks:
            return
        X_new = self._embed(chunks)
        self._embeds = np.concatenate([self._embeds, X_new], axis=0)
        self._meta.extend([{"text": t, **m} for t, m in zip(chunks, metas)])

        if self._use_faiss:
            self._index.add(X_new)
        else:
            # 重新fit (簡潔起見)
            nn = NearestNeighbors(n_neighbors=min(10, max(1, self.cfg.top_k*3)), metric="cosine")
            nn.fit(self._embeds)
            self._index = nn
        self._persist()

    def retrieve(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        if self._index is None or self._embeds is None:
            raise RuntimeError("lab_repository: 索引不存在，請先 build()")

        qv = self._embed([query])
        tk = top_k if top_k is not None else self.cfg.top_k
        ms = min_score if min_score is not None else self.cfg.min_score

        idx, sims = self._search(qv, tk)
        results = []
        for i, s in zip(idx.tolist(), sims.tolist()):
            if i < 0: 
                continue
            if s < ms:
                continue
            meta = self._meta[i]
            results.append({"score": float(s), **meta})
        return results
    
    def retrieve_structured(self, query: str, top_k: int = 4, min_score: float = None) -> dict:
        """Debug/maintenance helper.

        正式 runtime feature 請改走 features/labkb.py，
        以免繞過 source-level rerank、ownership hints 與 structured observation policy。
        """
        hits = self.retrieve(query, top_k=top_k, min_score=min_score)
        return {
            "query": query,
            "num_hits": len(hits),
            "top_score": hits[0]["score"] if hits else 0.0,
            "hits": hits,
            "enough": len(hits) >= 1,
            "context": self.as_context(query, top_k=top_k),
        }

    def as_context(self, query: str, sep: str="\n---\n", top_k: Optional[int]=None) -> str:
        """Debug/maintenance helper.

        正式 runtime feature 請改走 features/labkb.py，避免直接把底層 chunk context 當成最終 observation。
        """
        hits = self.retrieve(query, top_k=top_k)
        blocks = []
        for h in hits:
            head = f"[來源檔案] {os.path.relpath(h['path'], start=self.cfg.raw_dir)}  (score={h['score']:.3f}, chunk={h['chunk_id']})"
            blocks.append(head + "\n" + h["text"])
        return sep.join(blocks)

    # ---------- 持久化 ----------
    def _persist(self):
        store = self.cfg.store_dir
        np.save(os.path.join(store, "vectors.npy"), self._embeds)
        with open(os.path.join(store, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        if self._use_faiss:
            faiss.write_index(self._index, os.path.join(store, "faiss.index"))

    def _load_if_exists(self):
        store = self.cfg.store_dir
        v = os.path.join(store, "vectors.npy")
        m = os.path.join(store, "meta.json")
        fi = os.path.join(store, "faiss.index")
        if not (os.path.exists(v) and os.path.exists(m)):
            return
        self._embeds = np.load(v).astype("float32")
        with open(m, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        if self.cfg.use_faiss and _HAS_FAISS and os.path.exists(fi):
            index = faiss.read_index(fi)
            self._index = index
            self._use_faiss = True
        else:
            nn = NearestNeighbors(n_neighbors=min(10, max(1, self.cfg.top_k*3)), metric="cosine")
            nn.fit(self._embeds)
            self._index = nn
            self._use_faiss = False
