# features/labkb.py
import os
import re
import sys
from collections import defaultdict
from typing import Optional

# --- 保證專案根目錄在 sys.path 裡（避免工作目錄不同時找不到 lab_kb） ---
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lab_kb.lab_repository import LabRepository, LabRepoConfig

_LAB_REPO: Optional[LabRepository] = None


def augment_query(q: str) -> str:
    """根據常見意圖，自動附加較穩定的中英擴充詞，幫助 lab 檢索。"""
    q = q or ""
    ql = q.lower()
    extras = []

    if any(k in ql for k in ["空調", "冷氣", "溫控", "hvac", "空氣調節"]):
        extras += ["HVAC", "air conditioning", "temperature control", "thermal comfort", "PID"]

    if any(k in ql for k in ["虛擬教師", "虛擬導師", "virtual teacher", "virtual tutor", "tutor"]):
        extras += ["virtual teacher", "virtual tutor", "intelligent tutoring system", "ITS", "avatar"]

    if any(k in ql for k in ["手勢", "手勢辨識", "gesture", "gesture recognition", "imu"]):
        extras += ["gesture", "gesture recognition", "hand gesture", "imu", "edge ai"]

    if any(k in ql for k in ["研究成果", "論文", "系統", "介紹", "有哪些", "還有", "其他", "補充", "更多"]):
        extras += ["research", "paper", "publication", "system", "project"]

    if extras:
        return f"{q} {' '.join(sorted(set(extras)))}"
    return q


def _reference_signal_score(text: str) -> int:
    t = (text or "").lower()
    score = 0
    strong_terms = [
        "references", "bibliography", "accessed:", "available:",
        "doi:", "vol.", "no.", "pp.", "ieee", "acm", "proc.",
    ]
    score += sum(2 for term in strong_terms if term in t)
    score += len(re.findall(r"\[[0-9]{1,2}\]", t))
    score += len(re.findall(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", t)) * 2
    score += len(re.findall(r"\bet al\.\b", t))
    if "references" in t and len(re.findall(r"\[[0-9]{1,2}\]", t)) >= 3:
        score += 4
    return score


def _content_signal_score(text: str) -> int:
    t = (text or "").lower()
    terms = [
        "abstract", "introduction", "conclusion", "this paper", "this study",
        "we propose", "we present", "we introduce", "we develop", "framework",
        "workflow", "system", "prototype", "poc", "fig.", "figure",
        "dataset", "experiment", "evaluation", "accuracy", "method",
    ]
    return sum(1 for term in terms if term in t)


def _summary_chunk_score(text: str, path: str = "") -> int:
    t = (text or "").lower()
    score = 0
    if any(term in t for term in ["abstract", "introduction", "concluding remarks", "conclusion"]):
        score += 3
    if any(term in t for term in ["this paper", "we propose", "we present", "we introduce", "this study"]):
        score += 3
    if any(term in t for term in ["system framework", "workflow", "prototype", "poc", "proof-of-concept"]):
        score += 2
    if any(term in t for term in ["department of", "university", "corresponding authors"]):
        score += 1
    if any(term in t for term in ["model building", "optimizer", "loss selection", "training:", "input layer", "hidden layer", "output layer"]):
        score -= 2
    score -= min(3, _reference_signal_score(text) // 5)
    if (path or "").lower().endswith('.pdf'):
        score += 1
    return score


def _ownership_signal_score(text: str, path: str = "") -> int:
    t = (text or "").lower()
    p = (path or "").lower()
    score = 0

    if "abstract" in t:
        score += 2
    if any(term in t for term in ["this paper", "we propose", "we present", "we introduce", "we develop"]):
        score += 2
    if any(term in t for term in ["department of", "university", "corresponding authors"]):
        score += 2
    if any(term in t for term in ["tatung university", "taipei", "meta quest", "unity", "system framework", "workflow of", "poc"]):
        score += 2
    if re.search(r"[A-Z][a-zA-Z\-']+(?:\d+)?,\s*[A-Z][a-zA-Z\-']+", text or ""):
        score += 1
    if p.endswith(".pdf"):
        score += 1

    score -= min(3, _reference_signal_score(text) // 4)
    return score


def _ownership_hint(text: str, path: str = "") -> str:
    score = _ownership_signal_score(text, path)
    if score >= 5:
        return "likely_lab_authored"
    if score >= 3:
        return "possibly_lab_authored"
    return "unclear"


def _looks_reference_heavy(text: str) -> bool:
    ref_score = _reference_signal_score(text)
    content_score = _content_signal_score(text)
    return ref_score >= max(6, content_score + 3)


def _prepare_hit(h: dict) -> dict:
    item = dict(h)
    text = item.get("text", "") or ""
    path = item.get("path", "") or ""
    penalty = 0.18 if _looks_reference_heavy(text) else 0.0
    base_score = max(0.0, float(item.get("score", 0.0)) - penalty)
    ownership_score = _ownership_signal_score(text, path)
    item["score"] = base_score
    item["is_reference_heavy"] = penalty > 0
    item["ownership_signal_score"] = ownership_score
    item["ownership_hint"] = _ownership_hint(text, path)
    summary_score = _summary_chunk_score(text, path)
    item["summary_chunk_score"] = summary_score
    item["diversity_score"] = base_score + min(0.10, ownership_score * 0.01) + min(0.08, max(0, summary_score) * 0.01)
    return item


def _rerank_and_filter_hits(hits: list[dict], *, per_source_cap: int = 2) -> list[dict]:
    """
    先把 reference-heavy chunk 降權，再做來源多樣化。
    目的不是只看單一 chunk 分數，而是避免同一篇論文佔滿所有名額。
    """
    prepared = [_prepare_hit(h) for h in hits]
    kept = [h for h in prepared if not h.get("is_reference_heavy")]
    dropped = [h for h in prepared if h.get("is_reference_heavy")]
    candidates = kept or sorted(dropped, key=lambda x: x.get("score", 0.0), reverse=True)[:1]
    if not candidates:
        return []

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[item.get("path", "")].append(item)

    for path in grouped:
        grouped[path].sort(
            key=lambda x: (
                x.get("summary_chunk_score", 0),
                x.get("ownership_signal_score", 0),
                x.get("diversity_score", 0.0),
                x.get("score", 0.0),
            ),
            reverse=True,
        )

    source_order = sorted(
        grouped.keys(),
        key=lambda path: grouped[path][0].get("diversity_score", 0.0),
        reverse=True,
    )

    selected: list[dict] = []
    used_per_source: dict[str, int] = defaultdict(int)
    progress = True
    while progress:
        progress = False
        for path in source_order:
            bucket = grouped[path]
            idx = used_per_source[path]
            if idx >= min(len(bucket), per_source_cap):
                continue
            selected.append(bucket[idx])
            used_per_source[path] += 1
            progress = True

    return selected


def _build_context_blocks(repo, hits: list[dict]) -> list[str]:
    blocks = []
    for h in hits:
        rel_path = os.path.relpath(h["path"], start=repo.cfg.raw_dir)
        head = f"[來源] {rel_path} (score={h['score']:.3f}, chunk={h['chunk_id']})"
        meta = []
        if h.get("is_reference_heavy"):
            meta.append("reference-heavy")
        oh = h.get("ownership_hint")
        if oh and oh != "unclear":
            meta.append(f"ownership={oh}")
        if meta:
            head += " [" + ", ".join(meta) + "]"
        blocks.append(head + "\n" + h["text"])
    return blocks


def _score_summary(hits: list[dict]) -> dict:
    """
    回傳 selected hits 的分數摘要。

    注意：rerank 後的第一筆不一定是 embedding 分數最高的 chunk。
    因此 observation 同時保留：
    - first_score：rerank 後第一筆的分數，方便除錯排序結果。
    - best_score：selected hits 中最高分，較適合 evaluator 判斷是否至少有一筆強相關證據。
    - avg_score：selected hits 的平均分數，輔助判斷整體檢索品質。
    - min_score：selected hits 中最低分，輔助除錯。
    
    top_score 暫時應對齊 best_score，維持舊欄位相容，同時避免 evaluator
    因為第一筆 rerank 結果分數略低而誤判 observation 不可用。
    """
    if not hits:
        return {
            "first_score": 0.0,
            "best_score": 0.0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "top_score": 0.0,
        }

    scores = [float(h.get("score", 0.0) or 0.0) for h in hits]
    first_score = scores[0]
    best_score = max(scores)
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)

    return {
        "first_score": first_score,
        "best_score": best_score,
        "avg_score": avg_score,
        "min_score": min_score,
        # backward compatibility：既有 evaluator 若仍讀 top_score，先讓它代表 best_score。
        "top_score": best_score,
    }


def _build_source_candidates(repo, hits: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in hits:
        grouped[item.get("path", "")].append(item)

    candidates: list[dict] = []
    for path, bucket in grouped.items():
        bucket_sorted = sorted(
            bucket,
            key=lambda x: (
                x.get("summary_chunk_score", 0),
                x.get("ownership_signal_score", 0),
                x.get("diversity_score", 0.0),
                x.get("score", 0.0),
            ),
            reverse=True,
        )
        rep = bucket_sorted[0]
        rel_path = os.path.relpath(path, start=repo.cfg.raw_dir) if path else ""
        ownership_hint = rep.get("ownership_hint", "unclear")
        is_confident_candidate = ownership_hint in {"likely_lab_authored", "possibly_lab_authored"}
        candidates.append({
            "path": path,
            "rel_path": rel_path,
            "ownership_hint": ownership_hint,
            "best_score": rep.get("score", 0.0),
            "best_summary_chunk_score": rep.get("summary_chunk_score", 0),
            "ownership_signal_score": rep.get("ownership_signal_score", 0),
            "representative_chunk_id": rep.get("chunk_id"),
            "representative_text": rep.get("text", ""),
            "num_selected_hits": len(bucket_sorted),
            "is_confident_candidate": is_confident_candidate,
        })

    candidates.sort(
        key=lambda x: (
            x.get("is_confident_candidate", False),
            x.get("best_summary_chunk_score", 0),
            x.get("ownership_signal_score", 0),
            x.get("best_score", 0.0),
        ),
        reverse=True,
    )
    return candidates


class LabKB:
    """實驗室知識庫封裝：負責根據 query 取出文字 context。"""

    def __init__(self):
        global _LAB_REPO
        if _LAB_REPO is not None:
            self.repo = _LAB_REPO
            return

        cfg = LabRepoConfig(
            raw_dir="./lab_kb/raw",
            store_dir="./lab_kb/store",
            embedding_model="BAAI/bge-m3",
            normalize_embeddings=True,
            chunk_size=800,
            chunk_overlap=120,
            top_k=4,
            min_score=0.30,
            use_faiss=True,
        )
        _LAB_REPO = LabRepository(cfg)
        self.repo = _LAB_REPO

    def context(self, user_query: str, top_k: Optional[int] = None, debug: bool = False) -> str:
        tk = top_k or 4
        expanded_query = augment_query(user_query)
        try:
            raw_hits = self.repo.retrieve(expanded_query, top_k=max(tk, 8))
        except Exception:
            if debug:
                import traceback
                traceback.print_exc()
            return ""

        hits = _rerank_and_filter_hits(raw_hits, per_source_cap=2)[:tk]
        if not hits:
            return ""

        if debug:
            print(f"[lab_kb] hits={len(raw_hits)} raw / {len(hits)} filtered for q='{user_query[:60]}'")

        return "\n---\n".join(_build_context_blocks(self.repo, hits))

    def retrieve_structured(self, user_query: str, top_k: Optional[int] = None, debug: bool = False) -> dict:
        tk = top_k or 4
        expanded_query = augment_query(user_query)
        try:
            raw_hits = self.repo.retrieve(expanded_query, top_k=max(tk, 8))
        except Exception as e:
            if debug:
                import traceback
                traceback.print_exc()
            return {
                "tool": "lab_kb",
                "query": user_query,
                "query_used": expanded_query,
                "num_hits": 0,
                "top_score": 0.0,
                "first_score": 0.0,
                "best_score": 0.0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "hits": [],
                "context": "",
                "error": str(e),
                "content_type_hint": "lab_docs",
                "has_authorship_signal": False,
                "ownership_hints": [],
                "num_unique_sources": 0,
                "unique_sources": [],
                "source_candidates": [],
                "num_confident_source_candidates": 0,
                "num_tentative_source_candidates": 0,
            }

        hits = _rerank_and_filter_hits(raw_hits, per_source_cap=2)[:tk]

        if debug:
            print(f"[lab_kb] hits={len(raw_hits)} raw / {len(hits)} filtered for q='{user_query[:60]}'")

        unique_paths = []
        for h in hits:
            path = h.get("path")
            if path and path not in unique_paths:
                unique_paths.append(path)

        ownership_hints = [h.get("ownership_hint", "unclear") for h in hits]
        source_candidates = _build_source_candidates(self.repo, hits)
        num_confident = sum(1 for c in source_candidates if c.get("is_confident_candidate"))
        num_tentative = sum(1 for c in source_candidates if not c.get("is_confident_candidate"))
        score_summary = _score_summary(hits)

        return {
            "tool": "lab_kb",
            "query": user_query,
            "query_used": expanded_query,
            "num_hits": len(hits),
            "top_score": score_summary["top_score"],
            "first_score": score_summary["first_score"],
            "best_score": score_summary["best_score"],
            "avg_score": score_summary["avg_score"],
            "min_score": score_summary["min_score"],
            "hits": hits,
            "raw_num_hits": len(raw_hits),
            "context": "\n---\n".join(_build_context_blocks(self.repo, hits)),
            "content_type_hint": "lab_docs",
            "has_authorship_signal": any(h.get("ownership_hint") in {"likely_lab_authored", "possibly_lab_authored"} for h in hits),
            "ownership_hints": ownership_hints,
            "num_unique_sources": len(unique_paths),
            "unique_sources": unique_paths,
            "source_candidates": source_candidates,
            "num_confident_source_candidates": num_confident,
            "num_tentative_source_candidates": num_tentative,
        }
