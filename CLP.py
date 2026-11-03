# clp.py — Ollama 專用版本：保留 LabKB / 記憶 / 進度列
import argparse, time, warnings, os, re
from features.progress import configure_progress
from features.persona import (
    load_persona,
    load_persona_by_name,  # ★ 依名稱載入 ./personas/{name}.json
    compose_system_prompt,
    sanitize_memory_text,
    maybe_inject_first_reply,
    get_interaction_profile,
    get_addressing_rule,
)
from features.labkb import LabKB, augment_query
from features.memory_wrap import ensure_memory, persist_memory_threads
from features.webtools import WebToolManager  # 外部檢索模組
DEBUG_EXT = True  # 如有需要，可以之後改成讀環境變數
DEBUG_RUNTIME = True  # 控制是否在終端顯示內部 trace / observation / memory debug
from backends.backend_ollama import OllamaBackend
from features.agent_state import build_agent_state, update_task_thread
from features.agent_planner import AgentPlanner
from features.agent_executor import execute_action
from dataclasses import asdict
from features.task_scope_builder import TaskScopeBuilder
from features.observation_evaluator import ObservationEvaluator
from features.agent_trace import AgentTrace
from features.task_continuity_resolver import TaskContinuityResolver

# --- fallback tokenizer：用字元長度粗估 token 數 ---
from types import SimpleNamespace

class _CharTok:
    def __call__(self, text, add_special_tokens=False):
        length = max(1, len(text) // 4 + 1)
        return SimpleNamespace(input_ids=[0] * length)

# --------- 後端建立（Ollama only） ---------
def build_backend(args) -> OllamaBackend:
    """
    目前只支援 Ollama。若未來要恢復 HF，再新增分支。
    """
    be = OllamaBackend(
        model_name=args.ollama_model,
        opts={"num_ctx": args.num_ctx},
    )
    be.warmup()
    return be

# 把 huggingface_hub.file_download 的 FutureWarning 全部關掉
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"huggingface_hub\.file_download",
)


def _resolve_memory_profile(persona: dict) -> str:
    """
    依 persona 決定記憶分流 key。
    目標：不同人格不要共用同一份 memory / task thread。
    """
    raw = (
        persona.get("name")
        or (persona.get("identity") or {}).get("call_sign")
        or persona.get("role_summary")
        or "default"
    )
    raw = str(raw).strip().lower()
    raw = re.sub(r"[^\w\-]+", "_", raw)
    return raw or "default"

# --------- system prompt 組裝 ---------
def build_system(
    user_prompt: str,
    persona: dict,
    mem_ctx: str,
    audience: str = "lab",
    trace: dict | None = None,
) -> str:
    base_system = compose_system_prompt(persona)
    blocks = [base_system]

    profile = get_interaction_profile(persona, audience=audience)
    addressing = get_addressing_rule(persona, audience=audience)
    target = profile.get("target", "使用者")
    tone = profile.get("tone", "穩定、清楚、友善")
    warmth = profile.get("warmth", "medium")
    humor = profile.get("humor_level", "low")
    directness = profile.get("directness", "medium")
    pressure_reduction = profile.get("pressure_reduction", "medium")
    clarity = profile.get("clarity", "high")
    example_density = profile.get("example_density", "medium")
    rewriting_preference = profile.get("rewriting_preference", "依對象調整表達。")
    examples = profile.get("examples") or []

    interaction_lines = [
        "【Interaction Context】",
        f"- 本輪對話的主要對象：{target}",
        f"- 建議稱呼：{addressing}",
        f"- 語氣：{tone}",
        f"- warmth: {warmth}",
        f"- humor_level: {humor}",
        f"- directness: {directness}",
        f"- pressure_reduction: {pressure_reduction}",
        f"- clarity: {clarity}",
        f"- example_density: {example_density}",
        f"- 改寫偏好：{rewriting_preference}",
    ]
    if examples:
        interaction_lines.append("- 語氣參考：")
        for ex in examples[:3]:
            interaction_lines.append(f"  - {ex}")

    if audience == "student":
        interaction_lines.extend([
            "- 可以把相同資訊改寫得更白話、更容易理解，但白話化的目的只是降低理解門檻，不是增加新內容。",
            "- 如果和前一輪問題相近，可以把同一份已知資訊整理成 1 到 3 個較容易理解的重點面向，也可以補上低風險的直白解釋。",
            "- 不得新增 observation 中未出現的研究方向、論文題目、系統名稱、框架名稱、方法名稱、應用場景、貢獻或成果。",
            "- 若來源僅為 mention_only，最多只能保守提及可能相關，不能擴寫成完整研究成果。",
        ])
    else:
        interaction_lines.extend([
            "- 可以保留較完整的技術脈絡與協作語氣。",
            "- 可有少量吐槽或黑色幽默，但必須維持支持、可靠與共同解題的立場。",
        ])

    blocks.append("\n".join(interaction_lines))

    if trace:
        intent = trace.get("intent") or {}
        strategy = trace.get("strategy") or {}
        primary_obs = trace.get("primary_observation") or {}
        evaluation = trace.get("evaluation") or {}
        fallback_obs = trace.get("fallback_observation") or {}
        task_scope = trace.get("task_scope") or {}

        if intent:
            blocks.append("【Intent】\n" + "\n".join([
                f"- intent: {intent.get('intent')}",
                f"- reason: {intent.get('reason')}",
                f"- confidence: {intent.get('confidence')}",
            ]))

        if strategy:
            blocks.append("【Strategy】\n" + "\n".join([
                f"- primary_tool: {strategy.get('primary_tool')}",
                f"- fallback_tool: {strategy.get('fallback_tool')}",
                f"- must_verify: {strategy.get('must_verify')}",
                f"- reason: {strategy.get('reason')}",
            ]))

        continuity = trace.get("continuity") or {}
        if continuity:
            blocks.append("【Task Continuity】\n" + "\n".join([
                f"- is_continuation: {continuity.get('is_continuation')}",
                f"- relation: {continuity.get('relation')}",
                f"- continued_task_type: {continuity.get('continued_task_type')}",
                f"- preferred_tool: {continuity.get('preferred_tool')}",
                f"- reason: {continuity.get('reason')}",
            ]))

        if primary_obs:
            lines = [f"- tool: {primary_obs.get('tool')}"]
            if primary_obs.get("num_unique_sources") is not None:
                lines.append(f"- num_unique_sources: {primary_obs.get('num_unique_sources')}")
            if primary_obs.get("num_confident_source_candidates") is not None:
                lines.append(f"- num_confident_source_candidates: {primary_obs.get('num_confident_source_candidates')}")
            if primary_obs.get("num_tentative_source_candidates") is not None:
                lines.append(f"- num_tentative_source_candidates: {primary_obs.get('num_tentative_source_candidates')}")
            source_candidates = primary_obs.get("source_candidates") or []
            if source_candidates:
                lines.append("- source_candidates:")
                for c in source_candidates:
                    lines.append(
                        "  - "
                        f"{c.get('rel_path','')} | ownership={c.get('ownership_hint')} | "
                        f"confident={c.get('is_confident_candidate')} | "
                        f"summary_score={c.get('best_summary_chunk_score')} | score={c.get('best_score')}"
                    )
            if primary_obs.get("context"):
                lines.append("\n[Primary Observation]\n" + primary_obs["context"])
            if primary_obs.get("question"):
                lines.append(f"- question: {primary_obs.get('question')}")
            blocks.append("【Primary Observation】\n" + "\n".join(lines))

        if evaluation:
            eval_lines = [
                f"- usable: {evaluation.get('usable')}",
                f"- reason: {evaluation.get('reason')}",
                f"- next_action: {evaluation.get('next_action')}",
            ]
            source_assessment = evaluation.get("source_assessment") or []
            if source_assessment:
                eval_lines.append("- source_assessment:")
                for item in source_assessment:
                    eval_lines.append(
                        "  - "
                        f"{item.get('rel_path','')} | level={item.get('evidence_level')} | "
                        f"ownership={item.get('ownership_hint')} | "
                        f"summary_score={item.get('best_summary_chunk_score')} | "
                        f"score={item.get('best_score')} | rationale={item.get('rationale')}"
                    )
            blocks.append("【Observation Evaluation】\n" + "\n".join(eval_lines))

        if fallback_obs:
            lines = [f"- tool: {fallback_obs.get('tool')}"]
            if fallback_obs.get("context"):
                lines.append("\n[Fallback Observation]\n" + fallback_obs["context"])
            if fallback_obs.get("question"):
                lines.append(f"- question: {fallback_obs.get('question')}")
            blocks.append("【Fallback Observation】\n" + "\n".join(lines))

        if task_scope:
            scope_lines = [
                f"- task_type: {task_scope.get('task_type')}",
                f"- target_scope: {task_scope.get('target_scope')}",
                f"- evidence_basis: {task_scope.get('evidence_basis')}",
            ]

            in_scope_files = task_scope.get("in_scope_files") or []
            out_of_scope_domains = task_scope.get("out_of_scope_domains") or []
            allowed_operations = task_scope.get("allowed_operations") or []
            preferred_answer_shape = task_scope.get("preferred_answer_shape") or []
            candidate_files = task_scope.get("candidate_files") or []
            known_symbols = task_scope.get("known_symbols") or {}
            allowed_change_points = task_scope.get("allowed_change_points") or []
            forbidden_inventions = task_scope.get("forbidden_inventions") or []

            if in_scope_files:
                scope_lines.append("- in_scope_files: " + ", ".join(in_scope_files))
            if out_of_scope_domains:
                scope_lines.append("- out_of_scope_domains: " + ", ".join(out_of_scope_domains))
            if allowed_operations:
                scope_lines.append("- allowed_operations: " + ", ".join(allowed_operations))
            if preferred_answer_shape:
                scope_lines.append("- preferred_answer_shape: " + " -> ".join(preferred_answer_shape))

            if candidate_files:
                scope_lines.append("- candidate_files: " + ", ".join(candidate_files))

            if known_symbols:
                scope_lines.append("- known_symbols:")
                for fname, syms in known_symbols.items():
                    if syms:
                        scope_lines.append(f"  - {fname}: {', '.join(syms)}")

            if allowed_change_points:
                scope_lines.append("- allowed_change_points:")
                for item in allowed_change_points:
                    scope_lines.append(f"  - {item}")

            if forbidden_inventions:
                scope_lines.append("- forbidden_inventions:")
                for item in forbidden_inventions:
                    scope_lines.append(f"  - {item}")

            blocks.append("【Task Scope】\n" + "\n".join(scope_lines))

    if mem_ctx:
        blocks.append("【Conversation Memory】\n" + mem_ctx)
    blocks.append(
        "【Policy】\n"
        "1. 請根據 Intent、Strategy、Observation Evaluation、Observation 與 Task Scope 回答。\n"
        "2. 若 Evaluation 判定某個 observation 不可用，不得假裝它足以支撐答案；若 observation 已判定 usable，請優先根據其中已出現的資訊回答。\n"
        "3. 若證據不足，請誠實說明不足；但若已存在可用 observation，且其中至少有一項可支持內容，則不得整題拒答。\n"
        "4. 對列舉型問題，只能整理 observation 可支持的項目，不要憑模型記憶另起一組主要清單。\n"
        "5. 若列舉的是實驗室研究成果／論文／系統成果：只能列 observation 可支持且可合理判定屬於本實驗室的項目；references、bibliography、related work 不算本實驗室成果。若只有單一來源，只能介紹該一項或其細節，不得擴寫成多項不同成果。\n"
        "6. 若 Observation Evaluation 中存在 source_assessment，必須逐一服從其 evidence_level：elaborate 才能完整介紹；mention_only 最多只能保守提及其可能主題或存在性；ignore 不得寫入答案。\n"
        "7. 對 mention_only 來源，不得補出 observation 中未出現的新論文題目、新系統名稱、新框架名稱、新方法名稱、新應用場景、新貢獻或新成效；若要提及，只能明說『目前還看到一個可能相關來源，但資料不足以完整確認』。\n"
        "8. 若 intent 是 design_planning 或問題承接目前專案脈絡，請優先延續目前這個系統，提供模組、檔案、修改順序與最小可行版本；不要泛談一般 AI agent 願景。\n"
        "9. 不得把 observation 中未出現的函式、類別、檔案、模組責任或新元件描述成既有事實；若是建議，必須明確標示為『建議新增』或『可考慮新增』。\n"
        "10. 若存在 Task Scope，請服從其 target_scope、in_scope_files、out_of_scope_domains、candidate_files、known_symbols、allowed_change_points 與 forbidden_inventions。\n"
        "11. 若使用者問題包含『今年』『目前』『現在』等時間詞，請沿用目前對話時間脈絡，不得自行改寫成其他年份。\n"
        "12. 最終回覆一律使用繁體中文；若來源為簡體中文，請先轉寫為繁體中文後再輸出。\n"
    )

    return "\n\n".join(blocks).strip()

def _build_project_context_observation(prompt: str, intent: str | None = None, debug: bool = False) -> dict:
    from features.project_context import ProjectContextProvider
    provider = ProjectContextProvider(project_root=".")
    return provider.retrieve_structured(
        user_query=prompt,
        intent=intent,
        debug=debug,
    )


def _project_context_provider(prompt: str, intent: str | None = None, debug: bool = False) -> dict:
    return _build_project_context_observation(prompt=prompt, intent=intent, debug=debug)


def _build_direct_observation(mode: str, prompt: str, mem_text: str = "", override_context: str | None = None) -> dict:
    if mode == "memory_only":
        return {
            "tool": "memory_only",
            "query": prompt,
            "context": mem_text or "",
        }

    if mode == "direct_plan":
        return {
            "tool": "direct_plan",
            "query": prompt,
            "context": override_context or (
                "請針對目前這個專案的現有架構提出最小修改方案。"
                "優先討論 CLP.py、webtools.py、labkb.py、memory_repository.py 等現有模組，"
                "並以『先改哪個檔案、再改哪個函式、最後怎麼測試』的順序回答。"
                "不要泛談一般 AI agent 願景，也不要把重點放在重新訓練模型或重建整套系統。"
            ),
        }

    if mode == "direct_explain":
        return {
            "tool": "direct_explain",
            "query": prompt,
            "context": (
                "請直接根據目前可見的系統架構脈絡，解釋當前專案中相關模組如何運作。"
                "優先描述目前程式中的資料流、模組職責、記憶與工具如何接入，不要改成一般理論說明。"
            ),
        }

    if mode == "ask_followup":
        # ask_followup 的 prompt 會由 AgentPlanner / ContinuityResolver 放入
        # 本輪真正要問使用者的澄清問題。
        # 不要在這裡覆蓋成通用定型文，否則上游已產生的 clarification_question 會失效。
        question = (prompt or "").strip() or "我需要更多資訊才能繼續。你想先補充哪一個部分？"
        return {
            "tool": "ask_followup",
            "query": question,
            "question": question,
        }

    return {
        "tool": "direct_answer",
        "query": prompt,
        "context": "",
    }



def _plan_primary_action(prompt: str, mem_text: str, agent_state, continuity=None):
    planner = AgentPlanner()
    return planner.decide(
        user_prompt=prompt,
        memory_text=mem_text,
        agent_state=agent_state,
        continuity=continuity,
    )


def _execute_primary_action(action, prompt: str, mem_text: str, intent: str, labkb: LabKB, webtools: WebToolManager, memory, debug: bool = False) -> dict:
    result = execute_action(
        action,
        kb=labkb,
        webtools=webtools,
        memory=memory,
        debug=debug,
        prompt=prompt,
        memory_text=mem_text,
        intent=intent,
        project_context_provider=_project_context_provider,
        direct_observation_builder=_build_direct_observation,
    )
    return (result or {}).get("observation") or {}


def _build_evaluation_strategy(action):
    fallback_tool = (getattr(action, "meta", {}) or {}).get("fallback_tool")
    must_verify = bool((getattr(action, "meta", {}) or {}).get("must_verify", False))
    return SimpleNamespace(
        primary_tool=getattr(action, "action_type", None),
        fallback_tool=fallback_tool,
        must_verify=must_verify,
        reason=getattr(action, "reason", ""),
    )


def _build_fallback_observation(fallback_tool: str | None, prompt: str, mem_text: str, intent: str, labkb: LabKB, webtools: WebToolManager, memory, debug: bool = False) -> dict:
    fallback_action = SimpleNamespace(
        action_type=fallback_tool or "direct_answer",
        query=prompt,
    )
    result = execute_action(
        fallback_action,
        kb=labkb,
        webtools=webtools,
        memory=memory,
        debug=debug,
        prompt=prompt,
        memory_text=mem_text,
        intent=intent,
        project_context_provider=_project_context_provider,
        direct_observation_builder=_build_direct_observation,
    )
    return (result or {}).get("observation") or _build_direct_observation("direct_answer", prompt=prompt, mem_text=mem_text)


# --------- 小工具：判斷回覆看起來有沒有被截斷 ---------
def _looks_incomplete(text: str, max_new_tokens: int) -> bool:
    """
    粗略判斷一段回覆是不是「很可能被長度上限切斷」：
    - 沒有以句號/問號/感嘆號等收尾
    - 本身長度不太短
    - 長度接近 max_new_tokens 的預算（以字數估 token）
    對中文做特別處理：CJK 比例高時，用「字數」本身估 token，而不是 //3。
    """
    if not text:
        return False

    stripped = text.rstrip()

    # 1) 如果已經正常收尾，就不要續寫
    if stripped[-1] in "。!?…．.」』）)】]":
        return False

    # 2) 太短的就當成正常簡短回答，不續寫（稍微放寬門檻）
    if len(stripped) < 100:
        return False

    # 3) 判斷是否以中日韓文字為主
    cjk_count = sum(1 for ch in stripped if "\u4e00" <= ch <= "\u9fff")
    cjk_ratio = cjk_count / max(1, len(stripped))

    if cjk_ratio > 0.3:
        # 中文為主：token 大約 ≈ 字數
        approx_tokens = len(stripped)
    else:
        # 英文或混合：維持原本「字數 // 3」估法
        approx_tokens = max(1, len(stripped) // 3)

    if approx_tokens < 0.8 * max_new_tokens:
        return False

    return True

# --------- 單輪生成（Ollama）---------
def generate_once(
    be: OllamaBackend,
    persona: dict,
    prompt: str,
    audience: str = "lab",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0,
):
    tok_like = getattr(be, "tokenizer_like", None) or _CharTok()

    try:
        model_id = getattr(be, "model_name", None) or "unknown"
        memory_profile = _resolve_memory_profile(persona)
        memory = ensure_memory(
            tokenizer=tok_like,
            summarizer_fn=lambda text: "• " + text[:110],
            backend="ollama",
            model_name=model_id,
            profile=memory_profile,
        )
        raw_mem = memory.build_memory_context(prompt)
        if DEBUG_RUNTIME:
            print("\n[DEBUG memory context]")
            print(raw_mem)

    except Exception:
        memory = None
        raw_mem = ""

    mem_text = sanitize_memory_text(raw_mem, persona)

    # ---- Phase 1 主幹：agent_state -> planner -> executor -> evaluation -> fallback ----
    labkb = LabKB()
    webtools = WebToolManager()

    agent_state = build_agent_state(memory, prompt)
    evaluator = ObservationEvaluator(backend=be)
    trace = AgentTrace()

    continuity_resolver = TaskContinuityResolver()
    continuity = continuity_resolver.resolve(
        user_prompt=prompt,
        memory_text=mem_text,
        agent_state=agent_state,
    )
    trace.continuity = {
        "is_continuation": continuity.is_continuation,
        "relation": continuity.relation,
        "continued_task_type": continuity.continued_task_type,
        "continued_task_subject": continuity.continued_task_subject,
        "preferred_tool": continuity.preferred_tool,
        "preferred_query": continuity.preferred_query,
        "confidence": continuity.confidence,
        "reason": continuity.reason,
    }

    primary_action = _plan_primary_action(
        prompt=prompt,
        mem_text=mem_text,
        agent_state=agent_state,
        continuity=continuity,
    )
    action_meta = (getattr(primary_action, "meta", {}) or {})
    planner_intent = action_meta.get("intent", "general")
    planner_intent_reason = action_meta.get("intent_reason", getattr(primary_action, "reason", ""))
    trace.intent = {
        "intent": planner_intent,
        "reason": planner_intent_reason,
        "confidence": getattr(primary_action, "confidence", 0.0),
        "source": "agent_planner",
    }
    agent_state.last_action = getattr(primary_action, "action_type", None)
    agent_state.confidence = getattr(primary_action, "confidence", 0.0)

    planning_trace = {
        "primary_tool": getattr(primary_action, "action_type", None),
        "fallback_tool": (getattr(primary_action, "meta", {}) or {}).get("fallback_tool"),
        "must_verify": bool((getattr(primary_action, "meta", {}) or {}).get("must_verify", False)),
        "reason": getattr(primary_action, "reason", ""),
        "source": "agent_planner",
    }
    trace.strategy = planning_trace

    primary_obs = _execute_primary_action(
        action=primary_action,
        prompt=prompt,
        mem_text=mem_text,
        intent=planner_intent,
        labkb=labkb,
        webtools=webtools,
        memory=memory,
        debug=DEBUG_EXT,
    )
    trace.primary_observation = primary_obs

    evaluation_strategy = _build_evaluation_strategy(primary_action)
    evaluation_result = evaluator.evaluate(
        user_prompt=prompt,
        intent=planner_intent,
        strategy=evaluation_strategy,
        observation=primary_obs,
        continuity=continuity,
    )
    trace.evaluation = asdict(evaluation_result)

    if not evaluation_result.usable:
        fallback_tool = evaluation_result.next_action or evaluation_strategy.fallback_tool or "direct_answer"
        trace.fallback_observation = _build_fallback_observation(
            fallback_tool=fallback_tool,
            prompt=prompt,
            mem_text=mem_text,
            intent=planner_intent,
            labkb=labkb,
            webtools=webtools,
            memory=memory,
            debug=DEBUG_EXT,
        )

    # ---- build task scope ----
    scope_builder = TaskScopeBuilder()
    scope_source_obs = trace.fallback_observation or trace.primary_observation or {}
    task_scope = scope_builder.build(
        user_prompt=prompt,
        intent=planner_intent,
        observation=scope_source_obs,
    )
    final_trace = trace.to_dict()
    final_trace["task_scope"] = scope_builder.to_dict(task_scope)

    # --- Debug 用 ---
    if DEBUG_RUNTIME:
        print("\n[Intent]")
        print("intent     =", planner_intent)
        print("reason     =", planner_intent_reason)
        print("confidence =", getattr(primary_action, "confidence", 0.0))

        print("\n[Strategy]")
        print("primary    =", planning_trace.get("primary_tool"))
        print("fallback   =", planning_trace.get("fallback_tool"))
        print("must_verify=", planning_trace.get("must_verify"))
        print("reason     =", planning_trace.get("reason"))

        print("\n[Primary Observation]")
        print(trace.primary_observation)

        print("\n[Evaluation]")
        print("usable     =", evaluation_result.usable)
        print("reason     =", evaluation_result.reason)
        print("next_action=", evaluation_result.next_action)

        if trace.fallback_observation:
            print("\n[Fallback Observation]")
            print(trace.fallback_observation)
    # ----------------

    # 自介觸發
    orig_max_new_tokens = max_new_tokens
    prompt_for_model, is_intro = maybe_inject_first_reply(prompt, persona)
    if is_intro:
        temperature = max(temperature, 0.85)
        top_p = max(top_p, 0.9)

    # ask_followup 直接回
    final_obs = trace.fallback_observation or trace.primary_observation or {}
    agent_state = update_task_thread(
        agent_state=agent_state,
        user_prompt=prompt,
        continuity=continuity,
        planner_action=primary_action,
        final_observation=final_obs,
        evaluation=evaluation_result,
    )
    if memory is not None:
        try:
            memory.current_task_thread = agent_state.current_thread
            memory.task_thread_history = agent_state.thread_history
            persist_memory_threads(memory)
        except Exception:
            pass
    if isinstance(final_obs, dict) and final_obs.get("tool") == "ask_followup":
        reply = final_obs.get("question", "我需要更多資訊才能繼續。")
        if memory is not None:
            memory.add_user(prompt)
            memory.add_assistant(reply)
        return reply

    sys_prompt = build_system(
        user_prompt=prompt,
        persona=persona,
        mem_ctx=mem_text,
        audience=audience,
        trace=final_trace,
    )
    """
    print("\n[DEBUG sys_prompt tail]")
    print(sys_prompt[-1000:])
    print("\n[DEBUG user_prompt]")
    print(repr(prompt_for_model))
    """
    res = be.generate(
        system_prompt=sys_prompt,
        user_prompt=prompt_for_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop=None,
    )
    """
    print("\n[DEBUG generate raw res]")
    print(type(res))
    print(repr(res))
    if hasattr(res, "text"):
        print("[DEBUG res.text repr]")
        print(repr(res.text))
    """
    reply = getattr(res, "text", str(res)).strip()

    for lead in ("助手：", "Assistant:", "assistant:", "模型>", "Assistant"):
        if reply.startswith(lead):
            reply = reply[len(lead):].lstrip()
            break

    # 記錄一般對話
    if memory is not None:
        memory.add_user(prompt)
        memory.add_assistant(reply)

        # 第一版：如果是規劃/設計題，把它登記成 task/open_loop
        try:
            if planner_intent in ("design_planning", "system_explanation"):
                if hasattr(memory, "add_task"):
                    memory.add_task(
                        title=prompt[:80],
                        detail=f"intent={planner_intent}, primary_tool={planning_trace.get('primary_tool')}",
                        status="active",
                    )
        except Exception:
            pass

    # 原本續寫機制先保留
    """
    if _looks_incomplete(reply, orig_max_new_tokens):
        try:
            raw_mem2 = memory.build_memory_context(prompt) if memory is not None else ""
            mem_text2 = sanitize_memory_text(raw_mem2, persona)

            sys_prompt2 = build_system(
                user_prompt=prompt,
                persona=persona,
                mem_ctx=mem_text2,
                audience=audience,
                trace=final_trace,
            )

            cont_user = (
                "請延續你上一段對使用者問題的回答，繼續完成說明與收尾，"
                "不要重頭再講，也不要修改前面已經說過的內容。"
            )

            res2 = be.generate(
                system_prompt=sys_prompt2,
                user_prompt=cont_user,
                max_new_tokens=min(256, orig_max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=None,
            )
            cont = getattr(res2, "text", str(res2)).strip()

            for lead in ("助手：", "Assistant:", "assistant:", "模型>", "Assistant"):
                if cont.startswith(lead):
                    cont = cont[len(lead):].lstrip()
                    break

            if cont:
                reply = reply.rstrip() + "\n" + cont
        except Exception:
            pass
"""
    return reply

# --------- 互動模式 ---------
def run_cli(args, persona: dict):
    print("[CLI] LLM 互動模式。輸入 /quit 離開。")
    print(f"[CLI] 目前預設受眾模式：{args.audience}（lab 或 student）")
    be = build_backend(args)
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"/q", "/quit", "exit"}:
            break

        # ★ 每輪可用 @lab / @student prefix 覆寫受眾
        audience = args.audience
        if user.startswith("@lab"):
            audience = "lab"
            user = user[len("@lab"):].lstrip()
        elif user.startswith("@student"):
            audience = "student"
            user = user[len("@student"):].lstrip()

        # ★ 根據受眾，在 user_prompt 前面加一句明確說明
        if audience == "lab":
            user_for_model = (
                "（本輪對話的對象是實驗室成員或系統創造者，你可以用對熟人的語氣、"
                "適度吐槽與黑色幽默，但仍要保持支持與協助。）\n" + user
            )
        else:  # student
            user_for_model = (
                "（本輪對話的對象是學生或外部使用者，請用溫柔、稍微可愛、降低壓力的口吻，"
                "避免過度吐槽與太重的黑色幽默。"
                "如果這個問題跟你在前一輪對實驗室成員說明的內容類似，"
                "請主動改寫成大學生也容易理解的版本："
                "可以保留相同的研究方向名稱，但用更白話、舉例、比喻的方式說明，"
                "不要逐字重複上一輪的句子。）\n"
                + user
            )

        t0 = time.time()
        reply = generate_once(
            be,
            persona=persona,
            prompt=user,
            audience=audience,  # ★ 傳入本輪受眾
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repeat_penalty,
        )
        print(f"模型> {reply}\n[耗時] {(time.time() - t0) * 1000:.0f} ms")


def run_gradio(args, persona: dict):
    import gradio as gr
    be = build_backend(args)

    def _respond(message, history):
        # Gradio 版本：先用預設 audience；之後你也可以改 UI 加按鈕切換
        return generate_once(
            be,
            persona=persona,
            prompt=message,
            audience=args.audience,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repeat_penalty,
        )

    with gr.Blocks(title="LLM (Ollama)") as demo:
        gr.Markdown("# LLM Chat (Ollama)")
        gr.Markdown(f"**目前 persona：** {persona.get('name', 'unknown')}　｜　**受眾模式：** {args.audience}")
        gr.ChatInterface(fn=_respond, chatbot=gr.Chatbot(height=480))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
    )


# --------- 進入點 ---------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Persona-Based Tour Guiding LLM Agent Runner (Ollama only)")
    m = p.add_mutually_exclusive_group()
    m.add_argument("--cli", action="store_true")
    m.add_argument("--gradio", action="store_true")

    # Ollama 參數
    p.add_argument("--ollama_model", default="taiwan-llama3-70b")
    p.add_argument("--num_ctx", type=int, default=8192)

    # 生成參數
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repeat_penalty", type=float, default=1.0)

    # 顯示細節
    p.add_argument("--progress", dest="progress", action="store_true")
    p.add_argument("--no-progress", dest="progress", action="store_false")
    p.set_defaults(progress=False)

    # persona 選擇
    p.add_argument(
        "--persona",
        type=str,
        default=None,
        help="persona name (from ./personas/{name}.json). leave empty to use ./persona.json",
    )

    # ★ 新增：受眾模式
    p.add_argument(
        "--audience",
        type=str,
        default="lab",
        choices=["lab", "student"],
        help="default audience type: lab (實驗室成員) or student (學生/外部使用者)",
    )

    args = p.parse_args()
    configure_progress(args.progress)

    # 根據參數載入 persona
    if args.persona:
        # 例如 --persona guide_agent -> ./personas/guide_agent.json
        persona = load_persona_by_name(args.persona, base_dir="./personas")
    else:
        # 維持原本行為：讀 ./persona.json
        persona = load_persona("./persona.json")

    # 預設 CLI
    if args.gradio:
        run_gradio(args, persona)
    else:
        run_cli(args, persona)
