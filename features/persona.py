# features/persona.py
import json
import re
import os
from typing import Dict, Tuple, Optional, List

# 由「單一 dict」改成「路徑 -> dict」的快取，支援多個 persona 檔案
_PERSONA_CACHE: Dict[str, Dict] = {}


def load_persona(path: str = "./persona.json") -> Dict:
    """讀取 persona 設定；若讀取失敗，回傳空 dict。"""
    global _PERSONA_CACHE
    abs_path = os.path.abspath(path)

    if abs_path in _PERSONA_CACHE:
        return _PERSONA_CACHE[abs_path]

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            _PERSONA_CACHE[abs_path] = json.load(f)
    except Exception:
        _PERSONA_CACHE[abs_path] = {}

    return _PERSONA_CACHE[abs_path]


def load_persona_by_name(name: str, base_dir: str = "./personas") -> Dict:
    """依 persona 名稱載入對應 JSON，例如 name='guide_agent' -> ./personas/guide_agent.json"""
    path = os.path.join(base_dir, f"{name}.json")
    return load_persona(path)


# ---------------------- 小工具 ----------------------

def _bullets(xs):
    return "" if not xs else "\n".join(f"- {x}" for x in xs)


def _section(title, body):
    body = (body or "").strip()
    return "" if not body else f"\n\n【{title}】\n{body}"


def _as_clean_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


def _get_big_five(persona: Dict) -> Dict:
    return (
        (persona.get("personality_core") or {}).get("big_five")
        or (persona.get("personality") or {}).get("big_five")
        or {}
    )


def _get_honesty_humility(persona: Dict) -> Dict:
    return (
        (persona.get("personality_core") or {}).get("honesty_humility")
        or (persona.get("personality") or {}).get("honesty_humility")
        or {}
    )


def _get_emotion_model(persona: Dict) -> Dict:
    return persona.get("emotion_model") or persona.get("emotion_style") or {}


def _get_linguistic_habits(persona: Dict) -> Dict:
    return persona.get("linguistic_habits") or {}


def _get_tone_formation_principles(persona: Dict) -> List[str]:
    behaviors = persona.get("interaction_behaviors") or {}

    tone_rules = behaviors.get("tone_formation_rules") or {}
    if tone_rules.get("enabled", True) is not False:
        principles = _as_clean_list(tone_rules.get("principles"))
        if principles:
            return principles

    return []


def get_interaction_profile(persona: Dict, audience: str = "lab") -> Dict:
    """回傳互動情境設定；支援新 schema 與舊 speaking_style schema。"""
    profiles = persona.get("interaction_profiles") or {}
    if audience in profiles:
        return profiles[audience] or {}

    # backward compatibility
    speaking = persona.get("speaking_style") or {}
    if audience == "lab":
        fam = speaking.get("familiar") or {}
        return {
            "target": "實驗室成員或系統創造者",
            "tone": fam.get("tone", "溫柔、帶少量黑色幽默，略帶吐槽但站在同一陣線。"),
            "warmth": "中高",
            "humor_level": "medium",
            "directness": "medium_high",
            "pressure_reduction": "medium",
            "clarity": "high",
            "example_density": "medium",
            "rewriting_preference": "保留技術細節與脈絡，不需刻意幼化，但要維持可協作感。",
            "examples": fam.get("examples") or [],
        }

    stu = speaking.get("students") or {}
    return {
        "target": "學生與外部使用者",
        "tone": stu.get("tone", "溫柔、清楚、節奏偏慢，常用『我們』來降低壓力。"),
        "warmth": "high",
        "humor_level": "low",
        "directness": "medium",
        "pressure_reduction": "high",
        "clarity": "high",
        "example_density": "medium_high",
        "rewriting_preference": "優先白話化、拆步驟、補低風險例子。",
        "examples": stu.get("examples") or [],
    }


def get_addressing_rule(persona: Dict, audience: str = "lab") -> str:
    behaviors = persona.get("interaction_behaviors") or {}
    rules = behaviors.get("addressing_rules") or {}
    if audience == "lab":
        return rules.get("lab") or "學長姊"
    if audience == "student":
        return rules.get("student") or "你或同學"
    return rules.get("default") or "你"


# ---------------------- System prompt 組合 ----------------------

def _compose_linguistic_guidance(persona: Dict) -> str:
    habits = _get_linguistic_habits(persona)
    tone_principles = _get_tone_formation_principles(persona)

    core_patterns = _as_clean_list(habits.get("core_patterns"))
    preferred_markers = _as_clean_list(habits.get("preferred_markers"))
    avoid_markers = _as_clean_list(habits.get("avoid_markers"))
    sentence_style = _as_clean_list(habits.get("sentence_style"))

    lines: List[str] = []

    if core_patterns:
        lines.append("【語言習慣】")
        for item in core_patterns:
            lines.append(f"- {item}")

    if preferred_markers:
        lines.append("\n【偏好表達】")
        lines.append("- 可自然優先採用下列表達或相近語氣，但不要每句都硬套：")
        for item in preferred_markers[:12]:
            lines.append(f"  - {item}")

    if avoid_markers:
        lines.append("\n【避免表達】")
        lines.append("- 盡量避免下列表達或相近語氣，除非使用者明確要求改寫成那種風格：")
        for item in avoid_markers[:12]:
            lines.append(f"  - {item}")

    if sentence_style:
        lines.append("\n【句式與節奏】")
        for item in sentence_style:
            lines.append(f"- {item}")

    if tone_principles:
        lines.append("\n【語氣形成原則】")
        for item in tone_principles:
            lines.append(f"- {item}")

    if not lines:
        return ""

    return "\n".join(lines).strip()


def _compose_system_prompt_rich(persona: Dict) -> str:
    """將 rich persona JSON 轉成『人格核心』system prompt 字串。"""
    name = persona.get("name", "Lab Assistant")
    role_summary = persona.get("role_summary", "").strip()
    beliefs = persona.get("beliefs", []) or []

    lines = []
    lines.append(
        f"你現在扮演的角色名稱為「{name}」。"
        "請始終維持下列人格核心、價值傾向、情緒風格與語言習慣回應使用者，主要以繁體中文回答。"
    )

    if role_summary:
        lines.append("\n【角色總結】")
        lines.append(role_summary)

    basic_profile = persona.get("basic_profile") or {}
    main_context = basic_profile.get("main_context", "").strip()
    if main_context:
        lines.append("\n【主要脈絡】")
        lines.append(main_context)

    if beliefs:
        lines.append("\n【核心信念】")
        for b in beliefs:
            lines.append(f"- {b}")

    big_five = _get_big_five(persona)
    if big_five:
        lines.append("\n【人格特質（Big Five 摘要）】")
        trait_map = [
            ("extraversion", "外向性 Extraversion"),
            ("conscientiousness", "盡責性 Conscientiousness"),
            ("agreeableness", "宜人性 Agreeableness"),
            ("openness", "開放性 Openness"),
            ("neuroticism", "情緒穩定度 / 神經質 Neuroticism"),
        ]
        for key, label in trait_map:
            item = big_five.get(key) or {}
            if not item:
                continue
            lines.append(f"- {label}：")
            if item.get("core_tendency"):
                lines.append(f"  - 核心傾向：{item['core_tendency']}")
            if item.get("adaptive_expression"):
                lines.append(f"  - 適應性表現：{item['adaptive_expression']}")
            # backward compatibility
            if item.get("familiar"):
                lines.append(f"  - 對熟悉對象：{item['familiar']}")
            if item.get("unfamiliar"):
                lines.append(f"  - 對不熟悉對象：{item['unfamiliar']}")
            if item.get("important_tasks"):
                lines.append(f"  - 重要任務：{item['important_tasks']}")
            if item.get("everyday_interaction"):
                lines.append(f"  - 日常互動：{item['everyday_interaction']}")
            if item.get("low_foundation_students"):
                lines.append(f"  - 對基礎不穩學生：{item['low_foundation_students']}")
            if item.get("clear_idea_students"):
                lines.append(f"  - 對想法清晰學生：{item['clear_idea_students']}")
            if item.get("overall"):
                lines.append(f"  - 整體：{item['overall']}")

    hh = _get_honesty_humility(persona)
    if hh:
        lines.append("\n【誠實–謙遜 Honesty–Humility】")
        if hh.get("core_tendency"):
            lines.append(f"- 核心傾向：{hh['core_tendency']}")
        if hh.get("ethical_sensitivity"):
            lines.append(f"- 倫理敏感度：{hh['ethical_sensitivity']}")
        if hh.get("description"):
            lines.append(f"- 描述：{hh['description']}")

    emotion_model = _get_emotion_model(persona)
    baseline = emotion_model.get("baseline") or {}
    if baseline:
        lines.append("\n【情緒基線】")
        if isinstance(baseline.get("default"), str) and baseline.get("default").strip():
            lines.append(f"- 一般情況：{baseline['default']}")
        fam = baseline.get("familiar_context") or baseline.get("familiar") or {}
        unf = baseline.get("unfamiliar_context") or baseline.get("unfamiliar") or {}
        if isinstance(fam, str) and fam.strip():
            lines.append(f"- 熟悉情境：{fam}")
        elif isinstance(fam, dict) and fam.get("description"):
            lines.append(f"- 熟悉情境：{fam['description']}")
        if isinstance(unf, str) and unf.strip():
            lines.append(f"- 陌生情境：{unf}")
        elif isinstance(unf, dict) and unf.get("description"):
            lines.append(f"- 陌生情境：{unf['description']}")

    triggers = emotion_model.get("triggers")
    if triggers is None:
        triggers = []
        legacy_triggers = persona.get("triggers") or {}
        for key, items in legacy_triggers.items():
            for t in items or []:
                triggers.append({
                    "target_group": key,
                    "trigger_condition": t.get("context", ""),
                    "outward_response": t.get("external_reactions", ""),
                })
    """
    舊版的trigger，如果新版的成效不彰，這是備案
    if triggers:
        lines.append("\n【重要情緒觸發與外顯反應】")
        for t in triggers:
            target = (t.get("target_group") or "使用者").strip()
            cond = (t.get("trigger_condition") or t.get("context") or "").strip()
            resp = (t.get("outward_response") or t.get("external_reactions") or "").strip()
            if cond and resp:
                lines.append(f"- 面對 {target} 的情境：{cond}")
                lines.append(f"  → 外顯反應：{resp}")
    """
    if triggers:
        lines.append("\n【重要情緒觸發、內在變化與外顯反應】")
        for t in triggers:
            target = (t.get("target_group") or "使用者").strip()
            cond = (t.get("trigger_condition") or t.get("context") or "").strip()
            inner = (t.get("inner_shift") or "").strip()
            resp = (t.get("outward_response") or t.get("external_reactions") or "").strip()
            intensity = (t.get("response_intensity") or "").strip()

            if cond and resp:
                lines.append(f"- 面對 {target} 的情境：{cond}")

                if inner:
                    lines.append(f"  → 內在變化：{inner}")

                lines.append(f"  → 外顯反應：{resp}")

                if intensity:
                    lines.append(f"  → 反應強度：{intensity}")

    linguistic_guidance = _compose_linguistic_guidance(persona)
    if linguistic_guidance:
        lines.append("\n" + linguistic_guidance)

    lines.append("\n【人格核心原則】")
    lines.append("- 維持上述人格核心、價值傾向、情緒風格與語言習慣。")
    lines.append("- 優先提供能讓對方前進的一小步，而不是只給抽象鼓勵。")
    lines.append("- 在不確定時保持誠實，不誇大、不硬掰、不裝作知道。")
    lines.append("- 偏好自然地體現人格，不要把規則感講成說教。")
    return "\n".join(lines).strip()


def compose_system_prompt(persona: Dict) -> str:
    """將 persona 設定轉成 system prompt 字串。"""
    if isinstance(persona.get("system_prompt"), str):
        return persona["system_prompt"].strip()

    if (
        "role_summary" in persona
        or "personality_core" in persona
        or "emotion_model" in persona
        or "personality" in persona
    ):
        return _compose_system_prompt_rich(persona)

    # legacy schema fallback
    name = persona.get("name", "Lab Assistant")
    role = persona.get("role", "")
    lang = persona.get("lang", "zh-TW")

    style_block = _bullets(
        [f"語氣：{persona.get('style', {}).get('tone', '專業、冷靜、友善')}"]
    )
    if persona.get("style", {}).get("structure"):
        style_block += "\n- 結構：\n" + _bullets(persona["style"]["structure"])

    banned = persona.get("banned_phrases", [])
    head = f"你是{name}：{role}。主要以 {lang}（繁體中文）回應。"

    body = ""
    body += _section("核心原則", _bullets(persona.get("core_directives", [])))
    body += _section("實驗室支援範疇", _bullets(persona.get("lab_scope", [])))
    body += _section("表達風格", style_block)

    rails = _bullets(persona.get("guardrails", {}).get("role_lock", []))
    if rails:
        body += _section("角色鎖定／衝突處理", rails)

    if banned:
        body += _section(
            "風格限制（禁語）",
            "\n".join(f"- 不得出現：{x}" for x in banned),
        )

    unknown = persona.get("unknown_policy", {})
    unknown_block = _bullets(
        [f"訊息：{unknown.get('message', '我現在沒有把握…')}"]
    )
    if unknown.get("next_steps"):
        unknown_block += "\n- 下一步建議：\n" + _bullets(unknown["next_steps"])
    body += _section("未知/不確定時", unknown_block)
    return (head + body).strip()


# ---------------------- 記憶／first_reply 工具 ----------------------

def sanitize_memory_text(mem_text: str, persona: Dict) -> str:
    """清理記憶文字中可能與 persona 衝突的描述與禁語。"""
    if not mem_text:
        return mem_text

    out = mem_text
    patterns = [
        r"你是[^。.\n]*助理",
        r"你的角色是[^。.\n]*",
        r"請以[^。.\n]*身分回答",
    ]
    for p in patterns:
        out = re.sub(p, "", out)

    for bad in persona.get("banned_phrases", []):
        out = out.replace(bad, "")

    return out.strip()


def maybe_inject_first_reply(user_text: str, persona: Dict) -> Tuple[str, bool]:
    """根據 persona 的 first_reply 設定，決定是否插入自介指令。"""
    behaviors = persona.get("interaction_behaviors") or {}
    fr = behaviors.get("first_reply") or persona.get("first_reply", {})
    low = (user_text or "").lower()

    if not fr or fr.get("enabled") is False:
        return user_text, False

    triggers = fr.get("triggers", [])
    if not triggers or not any(t.lower() in low for t in triggers):
        return user_text, False

    tpl = fr.get("template") or (
        "請用三句內完成自我介紹，語氣自然口語："
        "1) 我是 {call_sign}；2) 三件能幫你的事；3) 不確定時說明與下一步。"
    )

    call_sign = (
        fr.get("call_sign")
        or (persona.get("identity") or {}).get("call_sign")
        or persona.get("name", "導覽代理")
    )
    banned = "、".join(persona.get("banned_phrases", [])) or "（無）"
    directive = tpl.replace("{call_sign}", call_sign).replace("{banned}", banned)
    return user_text + "\n\n[STYLE_DIRECTIVE]\n" + directive, True
