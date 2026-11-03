# features/extkb.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

import os
import json
import datetime
import threading
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    # 推薦使用新版 ddgs（duckduckgo_search 已經被作者標記為改名）
    from ddgs import DDGS
    _HAS_DDGS = True
except Exception:
    try:
        # 若你環境裝的是舊版 duckduckgo_search，也支援一下
        from duckduckgo_search import DDGS  # type: ignore
        _HAS_DDGS = True
    except Exception:
        _HAS_DDGS = False


# ========= 設定區（你只要改這幾個常數） =========

# 全域開關：是否啟用外部檢索（Tavily + ddgs 一起控制）
EXT_SEARCH_ENABLED: bool = os.getenv("EXT_SEARCH_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}

# --- Tavily 設定（主力搜尋） ---
# 文件：https://docs.tavily.com/documentation/api-reference/endpoint/search
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_ENDPOINT: str = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search").strip()

# 搜尋深度： "basic" 比較省、"advanced" 會抓更相關的內容但比較貴（credits 消耗較多）
TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced").strip()  # "basic" 或 "advanced"
TAVILY_MAX_RESULTS_DEFAULT: int = 5

# 每天最多允許 Tavily 呼叫次數（保險用，防止超過免費額度）
_MAX_TAVILY_CALLS_PER_DAY: int = int(os.getenv("MAX_TAVILY_CALLS_PER_DAY", "80"))

# --- ddgs 設定（備援搜尋，走 DuckDuckGo） ---
DDGS_MAX_RESULTS_DEFAULT: int = 4

# 記錄 Tavily 使用狀態的檔案
_USAGE_FILE = os.getenv("EXT_SEARCH_USAGE_FILE", "./ext_search_usage.json")
_LOCK = threading.Lock()
_usage_state = {
    "date": None,          # "YYYY-MM-DD"
    "tavily_count": 0,     # 今日 Tavily 呼叫次數
    "tavily_disabled": False,  # True 表示暫時停用 Tavily（額度或錯誤）
}

# ===============================================


# ---------- Tavily 使用狀態工具 ----------

def _today_str() -> str:
    return datetime.date.today().isoformat()


def _load_usage() -> None:
    """從檔案載入 Tavily 使用狀態，若日期不同則重置每日計數與停用狀態。"""
    global _usage_state

    if not os.path.exists(_USAGE_FILE):
        _usage_state = {
            "date": _today_str(),
            "tavily_count": 0,
            "tavily_disabled": False,
        }
        return

    try:
        with open(_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _usage_state = {
            "date": _today_str(),
            "tavily_count": 0,
            "tavily_disabled": False,
        }
        return

    # 若是新的一天，就重置計數與停用狀態
    if data.get("date") != _today_str():
        _usage_state = {
            "date": _today_str(),
            "tavily_count": 0,
            "tavily_disabled": False,
        }
    else:
        _usage_state = {
            "date": data.get("date", _today_str()),
            "tavily_count": int(data.get("tavily_count", 0)),
            "tavily_disabled": bool(data.get("tavily_disabled", False)),
        }


def _save_usage() -> None:
    try:
        with open(_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(_usage_state, f, ensure_ascii=False, indent=2)
    except Exception:
        # 寫檔失敗就略過，不讓主流程掛掉
        pass


def _can_use_tavily() -> bool:
    """
    檢查 Tavily 是否允許使用：
    - 沒被標記停用
    - 未超過每日最大呼叫次數
    """
    with _LOCK:
        _load_usage()
        if _usage_state.get("tavily_disabled"):
            return False
        if _usage_state.get("tavily_count", 0) >= _MAX_TAVILY_CALLS_PER_DAY:
            return False
        return True


def _record_tavily_success_call() -> None:
    """成功呼叫一次 Tavily 後，累計今日使用次數。"""
    with _LOCK:
        _load_usage()
        _usage_state["tavily_count"] = int(_usage_state.get("tavily_count", 0)) + 1
        _usage_state["date"] = _today_str()
        _save_usage()


def _mark_tavily_disabled() -> None:
    """遇到額度/信用/限流相關錯誤時，標記 Tavily 暫停使用（直到隔天自動重置）。"""
    with _LOCK:
        _load_usage()
        _usage_state["tavily_disabled"] = True
        _usage_state["date"] = _today_str()
        _save_usage()


class ExternalKB:
    """
    外部檢索封裝：

    - 優先使用 Tavily（若有設定 API key 且未達保險條件）
    - Tavily 出錯 / 沒結果 / 被保險關閉時，如果有安裝 ddgs，則 fallback 到 DuckDuckGo 搜尋
    - 對外只暴露 context(user_query, top_k, debug) 這個介面
    """

    def __init__(self):
        self.enabled = EXT_SEARCH_ENABLED

        # 根據是否有填 API key 決定 Tavily 是否啟用
        self._use_tavily = bool(TAVILY_API_KEY.strip())
        self._has_ddgs = _HAS_DDGS

    # ---------- Tavily ----------

    def _call_tavily(
        self,
        query: str,
        top_k: int,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        呼叫 Tavily Search API，回傳標準化結果列表。
        回傳格式：
          [
            {"title": str, "snippet": str, "source": str},
            ...
          ]
        """
        # 沒 key 或被保險關閉，直接略過 Tavily
        if not self._use_tavily:
            return []
        if not _can_use_tavily():
            if debug:
                print("[extkb] Tavily is disabled by usage guard or quota limit.")
            return []

        payload = {
            "query": query,
            "search_depth": TAVILY_SEARCH_DEPTH,
            "max_results": int(top_k),
            "include_answer": False,
            "include_images": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TAVILY_API_KEY}",
        }

        try:
            r = requests.post(
                TAVILY_ENDPOINT, json=payload, headers=headers, timeout=20
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            msg = str(e).lower()
            if debug:
                print(f"[extkb] Tavily request failed: {e}")

            # 如果錯誤訊息看起來是額度/信用/限流相關，就啟動保險，暫停 Tavily
            quota_keywords = ["quota", "credit", "rate limit", "billing", "payment", "insufficient"]
            if any(k in msg for k in quota_keywords):
                if debug:
                    print("[extkb] Detected possible quota/credit issue, disabling Tavily for today.")
                _mark_tavily_disabled()

            return []

        # 走到這裡代表成功呼叫 Tavily 一次
        _record_tavily_success_call()

        results = data.get("results") or []
        out: List[Dict[str, Any]] = []
        for item in results[:top_k]:
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            source = str(item.get("url") or item.get("source") or "").strip()
            if not (title or snippet or source):
                continue
            out.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "source": source,
                }
            )
        return out

    # ---------- ddgs (DuckDuckGo) ----------

    def _call_ddgs(
        self,
        query: str,
        top_k: int,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        使用 ddgs（或舊版 duckduckgo_search）進行文字搜尋。
        """
        if not self._has_ddgs:
            if debug:
                print("[extkb] ddgs not installed; skip DuckDuckGo fallback")
            return []

        results: List[Dict[str, Any]] = []

        try:
            with DDGS() as ddgs:
                # text search，回傳 generator，每個項目通常有 title / href / body
                for r in ddgs.text(
                    query,
                    max_results=top_k,
                ):
                    title = str(r.get("title") or "").strip()
                    snippet = str(r.get("body") or "").strip()
                    source = str(r.get("href") or "").strip()
                    if not (title or snippet or source):
                        continue
                    results.append(
                        {
                            "title": title,
                            "snippet": snippet,
                            "source": source,
                        }
                    )
        except Exception as e:
            if debug:
                print(f"[extkb] ddgs search failed: {e}")
            return []

        return results

    # ---------- Provider 選擇邏輯 ----------

    def _call_provider(
        self,
        query: str,
        top_k: int,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        統一的搜尋入口：
        - 若有 Tavily key 且未被保險關閉 → 優先用 Tavily
        - Tavily 沒結果或出錯 → 若有 ddgs 則 fallback
        - 如果兩個都不行 → 回 []
        """
        if not self.enabled:
            return []

        # 1) Tavily 優先
        if self._use_tavily:
            hits = self._call_tavily(query, top_k=top_k, debug=debug)
            if hits:
                return hits
            if debug:
                print("[extkb] Tavily 回傳空結果或被保險關閉，嘗試 ddgs fallback")

        # 2) ddgs fallback
        if self._has_ddgs:
            hits = self._call_ddgs(query, top_k=top_k, debug=debug)
            if hits:
                return hits

        # 3) 兩個都掛了
        if debug:
            print("[extkb] no external search provider available/working")
        return []

    # ---------- 對 clp.py 暴露的介面 ----------

    def context(
        self,
        user_query: str,
        top_k: int = 3,
        debug: bool = False,
    ) -> str:
        """
        對 clp.py 用的主函式：
        - 給一個 user_query
        - 回一整段文字（多個搜尋結果連在一起），可以直接塞進 system prompt 的【External Search】區塊。
        """
        if not user_query or not self.enabled:
            return ""

        # top_k 統一限制在合理範圍
        k = max(1, min(int(top_k), 8))

        results = self._call_provider(user_query, top_k=k, debug=debug)
        if not results:
            return ""

        blocks: List[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title") or "(無標題)"
            source = r.get("source") or ""
            head = f"[結果 {i}] {title}"
            if source:
                head += f"（來源：{source}）"
            snippet = r.get("snippet") or ""
            blocks.append(head + "\n" + snippet)

        return "\n---\n".join(blocks)
