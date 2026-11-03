from __future__ import annotations

from features.extkb import ExternalKB


class WebToolManager:
    """
    外部搜尋工具層：
    - 不再負責判斷「要不要用 web」
    - 只負責呼叫 ExternalKB
    - 回傳結構化 observation，供 executor / evaluator 使用
    """

    def __init__(self):
        self._extkb = ExternalKB()

    def search_structured(self, user_prompt: str, top_k: int = 3, debug: bool = False) -> dict:
        query = user_prompt or ""

        if not query.strip():
            return {
                "tool": "web",
                "query": query,
                "results": [],
                "num_results": 0,
                "context": "",
                "error": "empty query",
                "content_type_hint": "web_search_results",
            }

        try:
            results = self._extkb._call_provider(query, top_k=top_k, debug=debug)
        except Exception as e:
            return {
                "tool": "web",
                "query": query,
                "results": [],
                "num_results": 0,
                "context": "",
                "error": str(e),
                "content_type_hint": "web_search_results",
            }

        blocks = []
        for i, r in enumerate(results, 1):
            title = r.get("title") or "(無標題)"
            source = r.get("source") or ""
            snippet = r.get("snippet") or ""
            head = f"[結果 {i}] {title}"
            if source:
                head += f"（來源：{source}）"
            blocks.append(head + "\n" + snippet)

        return {
            "tool": "web",
            "query": query,
            "results": results,
            "num_results": len(results),
            "context": "\n---\n".join(blocks),
            "error": None,
            "content_type_hint": "web_search_results",
        }
