from __future__ import annotations
from typing import Dict, Any, Callable, Optional


def execute_action(
    action,
    kb=None,
    webtools=None,
    memory=None,
    debug: bool = False,
    *,
    prompt: Optional[str] = None,
    memory_text: str = "",
    intent: Optional[str] = None,
    project_context_provider: Optional[Callable[..., Dict[str, Any]]] = None,
    direct_observation_builder: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    統一 action -> observation 執行入口。

    Phase 1 收尾版：
    - 不只執行 retrieve_lab_kb / retrieve_web
    - 也接管 project_context / memory_only / direct_plan / direct_explain / ask_followup
    - 讓 CLP.py 只保留 orchestration，不再維護第二套 action 分流
    """
    at = getattr(action, "action_type", None)
    q = getattr(action, "query", None) or prompt or ""

    def _ok(obs: Dict[str, Any] | None) -> Dict[str, Any]:
        return {
            "action": at,
            "ok": True,
            "observation": obs or {},
        }

    def _err(msg: str) -> Dict[str, Any]:
        return {
            "action": at,
            "ok": False,
            "error": msg,
            "observation": None,
        }

    if at == "retrieve_lab_kb":
        if kb is None:
            return _err("kb unavailable")
        try:
            if not hasattr(kb, "retrieve_structured"):
                return _err("kb does not support structured retrieval")
            obs = kb.retrieve_structured(q, top_k=8, debug=debug)
            if isinstance(obs, dict):
                obs.setdefault("tool", "lab_kb")
            return _ok(obs)
        except Exception as e:
            return _err(str(e))

    if at == "retrieve_web":
        if webtools is None:
            return _err("webtools unavailable")
        try:
            if not hasattr(webtools, "search_structured"):
                return _err("webtools does not support structured retrieval")
            obs = webtools.search_structured(q, top_k=3, debug=debug)
            if isinstance(obs, dict):
                obs.setdefault("tool", "web")
            return _ok(obs)
        except Exception as e:
            return _err(str(e))

    if at == "project_context":
        if project_context_provider is None:
            return _err("project_context provider unavailable")
        try:
            obs = project_context_provider(prompt=q, intent=intent, debug=debug)
            if isinstance(obs, dict):
                obs.setdefault("tool", "project_context")
            return _ok(obs)
        except Exception as e:
            return _err(str(e))

    if at in {"memory_only", "direct_plan", "direct_explain", "ask_followup", "direct_answer"}:
        if direct_observation_builder is None:
            return _err("direct observation builder unavailable")
        try:
            obs = direct_observation_builder(at, prompt=q, mem_text=memory_text)
            if isinstance(obs, dict):
                obs.setdefault("tool", at)
            return _ok(obs)
        except Exception as e:
            return _err(str(e))

    if at == "proactive_hint":
        if direct_observation_builder is not None:
            try:
                obs = direct_observation_builder(
                    "direct_plan",
                    prompt=q,
                    mem_text=memory_text,
                    override_context="請優先提供可執行的修改順序、模組邊界與最小可行版本。",
                )
                if isinstance(obs, dict):
                    obs["tool"] = "direct_plan"
                return _ok(obs)
            except Exception as e:
                return _err(str(e))
        return _ok({
            "tool": "direct_plan",
            "query": q,
            "context": "請優先提供可執行的修改順序、模組邊界與最小可行版本。",
        })

    return _ok(None)
