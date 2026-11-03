# features/observation_judge.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import re


@dataclass
class JudgeResult:
    usable: bool
    reason: str
    next_action: Optional[str] = None
    confidence: float = 0.0


def _extract_json_block(text: str) -> dict | None:
    """
    從模型輸出中盡量抓出 JSON。
    """
    if not text:
        return None

    # 先直接整段試
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 再抓第一個 {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


class ObservationJudge:
    """
    用 LLM 來判斷 observation 是否真的足以回答當前問題。
    只建議在規則無法明確判定時使用。
    """

    def __init__(self, backend):
        self.backend = backend

    def judge(
        self,
        user_prompt: str,
        intent: str,
        observation: dict,
        fallback_tool: Optional[str] = None,
        max_new_tokens: int = 220,
    ) -> JudgeResult:
        obs_text = observation.get("context", "") if isinstance(observation, dict) else ""
        tool = observation.get("tool", "") if isinstance(observation, dict) else ""

        system_prompt = (
            "你是一個專門判斷『工具結果是否足以回答使用者問題』的評估器。\n"
            "你的任務不是回答使用者，而是評估 observation 是否真的可用。\n\n"
            "判斷原則：\n"
            "1. 若 observation 和使用者問題主題不符，usable=false。\n"
            "2. 若 observation 是泛論、論文背景或弱相關內容，無法支撐目前任務，usable=false。\n"
            "3. 若 intent 是 system_explanation，必須真的在解釋目前系統/模組/流程，而不是一般概念。\n"
            "4. 若 intent 是 design_planning，必須真的能支撐具體修改方案，而不是一般性願景。\n"
            "5. 若 usable=false，請建議 next_action；通常是 fallback_tool。\n"
            "6. 僅輸出 JSON，不要輸出其他說明。\n\n"
            "輸出格式：\n"
            "{\n"
            '  "usable": true 或 false,\n'
            '  "reason": "一句簡短理由",\n'
            '  "next_action": "fallback tool name 或 null",\n'
            '  "confidence": 0.0 到 1.0\n'
            "}"
        )

        user_prompt_text = (
            f"使用者問題：\n{user_prompt}\n\n"
            f"intent：{intent}\n"
            f"tool：{tool}\n"
            f"fallback_tool：{fallback_tool}\n\n"
            f"observation：\n{obs_text[:5000]}"
        )

        res = self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.0,
            stop=None,
        )
        text = getattr(res, "text", str(res)).strip()
        obj = _extract_json_block(text)

        if not obj:
            return JudgeResult(
                usable=False,
                reason="judge failed to return valid json",
                next_action=fallback_tool,
                confidence=0.0,
            )

        usable = bool(obj.get("usable", False))
        reason = str(obj.get("reason", "")).strip() or "no reason"
        next_action = obj.get("next_action", None)
        if next_action in ("null", "", "None"):
            next_action = None

        try:
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        return JudgeResult(
            usable=usable,
            reason=reason,
            next_action=next_action or (fallback_tool if not usable else None),
            confidence=confidence,
        )