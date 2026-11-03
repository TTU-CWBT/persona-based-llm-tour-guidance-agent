from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


ActionType = Literal[
    "direct_answer",
    "retrieve_lab_kb",
    "retrieve_web",
    "ask_followup",
    "proactive_hint",
    "project_context",
    "memory_only",
    "direct_plan",
    "direct_explain",
]

IntentType = Literal[
    "design_planning",
    "system_explanation",
    "knowledge_lookup",
    "external_fresh_info",
    "followup",
    "general",
]


@dataclass
class AgentAction:
    action_type: ActionType
    reason: str
    query: Optional[str] = None
    confidence: float = 0.0
    meta: Optional[Dict[str, Any]] = None


class AgentPlanner:
    def decide(
        self,
        user_prompt: str,
        memory_text: str = "",
        agent_state=None,
        intent: Optional[str] = None,
        continuity=None,
    ) -> AgentAction:
        q = (user_prompt or "").strip()
        if not q:
            return AgentAction(
                "direct_answer",
                "empty input",
                confidence=0.1,
                meta={
                    "intent": "general",
                    "intent_reason": "empty input",
                    "fallback_tool": "direct_answer",
                    "must_verify": False,
                },
            )

        if continuity and getattr(continuity, "is_continuation", False):
            action = self._decide_from_continuity(user_prompt, continuity, agent_state)
            if action is not None:
                return action

        inferred_intent, intent_reason, intent_conf = self._infer_intent(
            user_prompt=user_prompt,
            memory_text=memory_text,
            agent_state=agent_state,
        )
        final_intent = intent or inferred_intent

        if final_intent == "external_fresh_info":
            return AgentAction(
                "retrieve_web",
                "needs fresh or external information",
                query=user_prompt,
                confidence=max(intent_conf, 0.9),
                meta={
                    "intent": final_intent,
                    "intent_reason": intent_reason,
                    "fallback_tool": "direct_answer",
                    "must_verify": True,
                    "task_type": "external_info",
                },
            )

        if final_intent == "followup":
            return AgentAction(
                "memory_only",
                "followup should rely on current conversation context first",
                query=user_prompt,
                confidence=max(intent_conf, 0.8),
                meta={
                    "intent": final_intent,
                    "intent_reason": intent_reason,
                    "fallback_tool": "ask_followup",
                    "must_verify": True,
                },
            )

        if final_intent == "design_planning":
            return AgentAction(
                "project_context",
                "design/planning request should inspect current project implementation first",
                query=user_prompt,
                confidence=max(intent_conf, 0.85),
                meta={
                    "intent": final_intent,
                    "intent_reason": intent_reason,
                    "fallback_tool": "direct_plan",
                    "must_verify": True,
                    "task_type": "design_planning",
                },
            )

        if final_intent == "system_explanation":
            return AgentAction(
                "project_context",
                "current system mechanism explanation should inspect current project implementation first",
                query=user_prompt,
                confidence=max(intent_conf, 0.82),
                meta={
                    "intent": final_intent,
                    "intent_reason": intent_reason,
                    "fallback_tool": "direct_explain",
                    "must_verify": True,
                    "task_type": "system_explanation",
                },
            )

        if final_intent == "knowledge_lookup":
            return AgentAction(
                "retrieve_lab_kb",
                "lab knowledge or guided-system profile lookup should consult internal KB first",
                query=user_prompt,
                confidence=max(intent_conf, 0.78),
                meta={
                    "intent": final_intent,
                    "intent_reason": intent_reason,
                    "fallback_tool": "direct_answer",
                    "must_verify": True,
                    "task_type": "lab_research_lookup",
                },
            )

        short_q = q.lower()
        planning_nudge_keywords = [
            "怎麼做", "如何做", "要改哪些", "怎麼改", "實作", "架構", "設計",
            "規劃", "步驟", "落地", "修改哪些部分", "how to", "design",
            "implement", "architecture",
        ]
        if any(k in short_q for k in planning_nudge_keywords):
            return AgentAction(
                "proactive_hint",
                "planning-like request benefits from proactive next-step guidance",
                query=user_prompt,
                confidence=0.7,
                meta={
                    "intent": final_intent,
                    "intent_reason": intent_reason,
                    "fallback_tool": "direct_answer",
                    "must_verify": False,
                },
            )

        if len(short_q) <= 3:
            return AgentAction(
                "ask_followup",
                "input too short or ambiguous",
                confidence=0.5,
                meta={
                    "intent": final_intent,
                    "intent_reason": intent_reason,
                    "fallback_tool": "direct_answer",
                    "must_verify": False,
                    "question": "你希望我先從架構、程式、還是記憶系統開始？",
                },
            )

        return AgentAction(
            "direct_answer",
            "general explanatory request",
            confidence=max(intent_conf, 0.6),
            meta={
                "intent": final_intent,
                "intent_reason": intent_reason,
                "fallback_tool": "direct_answer",
                "must_verify": False,
                "task_type": "general",
            },
        )

    def _decide_from_continuity(self, user_prompt: str, continuity, agent_state=None) -> Optional[AgentAction]:
        relation = getattr(continuity, "relation", "clarify")
        task_type = getattr(continuity, "continued_task_type", None)
        preferred_tool = getattr(continuity, "preferred_tool", None)
        preferred_query = getattr(continuity, "preferred_query", None) or user_prompt
        conf = max(0.8, float(getattr(continuity, "confidence", 0.0) or 0.0))
        reason = getattr(continuity, "reason", "task continuity")

        if getattr(continuity, "needs_clarification", False) or preferred_tool == "ask_followup":
            clarification_question = (
                getattr(continuity, "clarification_question", None)
                or preferred_query
                or "這裡需要再確認一下，你希望我比較或說明的是哪一個對象？"
            )
            return AgentAction(
                "ask_followup",
                reason or "continuation needs clarification",
                query=clarification_question,
                confidence=conf,
                meta={
                    "intent": "followup",
                    "intent_reason": reason or "continuation needs clarification",
                    "continuation_relation": relation,
                    "continued_task_type": task_type,
                    "fallback_tool": "direct_answer",
                    "must_verify": False,
                    "task_type": task_type or "clarification",
                    "question": clarification_question,
                },
            )

        if relation in {"expand", "compare"} and task_type == "lab_research_lookup":
            return AgentAction(
                "retrieve_lab_kb",
                "continuation should extend current lab research/system-profile lookup thread",
                query=preferred_query,
                confidence=conf,
                meta={
                    "intent": "knowledge_lookup",
                    "intent_reason": reason,
                    "continuation_relation": relation,
                    "continued_task_type": task_type,
                    "fallback_tool": "direct_answer",
                    "must_verify": True,
                    "task_type": "lab_research_lookup",
                },
            )

        if relation in {"clarify", "shift"} and task_type == "lab_research_lookup":
            return AgentAction(
                "retrieve_lab_kb",
                "continuation should stay on current lab KB source of truth",
                query=preferred_query,
                confidence=conf,
                meta={
                    "intent": "knowledge_lookup",
                    "intent_reason": reason,
                    "continuation_relation": relation,
                    "continued_task_type": task_type,
                    "fallback_tool": "direct_answer",
                    "must_verify": True,
                    "task_type": "lab_research_lookup",
                },
            )

        if relation in {"clarify", "shift", "compare", "expand"} and task_type in {
            "system_explanation",
            "design_planning",
            "system_or_project_context",
        }:
            return AgentAction(
                "project_context",
                "continuation should stay on current project/system context thread",
                query=preferred_query,
                confidence=conf,
                meta={
                    "intent": "system_explanation" if task_type != "design_planning" else "design_planning",
                    "intent_reason": reason,
                    "continuation_relation": relation,
                    "continued_task_type": task_type,
                    "fallback_tool": "direct_explain" if task_type != "design_planning" else "direct_plan",
                    "must_verify": True,
                    "task_type": task_type,
                },
            )

        if preferred_tool == "retrieve_web" or task_type == "external_info":
            return AgentAction(
                "retrieve_web",
                "continuation should stay on current external information thread",
                query=preferred_query,
                confidence=conf,
                meta={
                    "intent": "external_fresh_info",
                    "intent_reason": reason,
                    "continuation_relation": relation,
                    "continued_task_type": task_type,
                    "fallback_tool": "direct_answer",
                    "must_verify": True,
                    "task_type": "external_info",
                },
            )

        return None

    def _infer_intent(self, user_prompt: str, memory_text: str = "", agent_state=None) -> tuple[IntentType, str, float]:
        q = (user_prompt or "").strip().lower()
        mem = (memory_text or "").lower()
        task = ((getattr(agent_state, "active_task", None) or "") if agent_state is not None else "").lower()

        if not q:
            return "general", "empty input", 0.1

        if self._is_persona_performance_prompt(q):
            return "general", "persona / role performance prompt", 0.78

        # 先處理明確的系統介紹 / 系統特色題
        # 這類題現在應優先進 lab_kb，因為已經有獨立整理過的介紹文件
        if self._is_guided_system_profile_lookup(q):
            return "knowledge_lookup", "asking for guided-system profile / core features from KB", 0.88

        followup_cues = [
            "那個", "這部分", "上述", "接下來", "那要怎麼",
            "然後呢", "這樣改", "這樣修", "剛剛", "上一段", "上面", "下面",
        ]
        explicit_followup_patterns = [
            "你剛剛", "剛剛提到", "上一段", "上面那段", "前面那個",
            "這個再講", "這部分再講", "這個是什麼意思",
        ]
        explicit_noun_after_this = [
            "這個系統", "這套系統",
            "這個代理", "這套代理",
            "這個導覽代理系統", "這套導覽代理系統",
            "這個研究", "這個模組", "這個功能", "這個架構",
        ]
        is_followup_like = (
            any(cue in q for cue in followup_cues)
            or any(p in q for p in explicit_followup_patterns)
        )
        this_with_explicit_noun = any(p in q for p in explicit_noun_after_this)

        if is_followup_like and not this_with_explicit_noun:
            if mem.strip() or task.strip():
                return "followup", "context-dependent followup", 0.8

        planning_keywords = [
            "怎麼做", "如何做", "要改哪些", "怎麼改", "實作", "架構", "設計",
            "規劃", "步驟", "落地", "修改方案", "最小修改方案", "最小可行",
            "修改哪些部分", "重構", "收斂", "整併", "how to", "design",
            "implement", "architecture", "refactor",
        ]
        if any(k in q for k in planning_keywords):
            return "design_planning", "planning / modification request", 0.85

        external_strong_keywords = [
            "最新", "最近", "今年", "價格", "售價", "上市", "發售", "新聞",
            "github", "paper", "論文", "conference", "版本", "官方網站", "下載",
            "安裝", "latest", "price", "release", "news", "version", "download", "install",
        ]
        external_time_cues = ["現在", "目前", "當前", "現況"]
        external_object_keywords = [
            "天氣", "股價", "匯率", "法規", "規定", "公告", "消息", "產品", "套件",
            "框架", "模型版本", "比賽", "比分", "總統", "ceo", "weather", "stock",
            "exchange rate", "law", "regulation", "announcement", "product", "package",
            "framework", "score",
        ]
        if any(k in q for k in external_strong_keywords):
            return "external_fresh_info", "time-sensitive or external info request", 0.9
        if any(k in q for k in external_time_cues) and any(k in q for k in external_object_keywords):
            return "external_fresh_info", "current-status request about an external object", 0.85

        if self._is_current_system_explanation_or_presentation(q):
            return "system_explanation", "asking to explain current project mechanism/implementation", 0.84

        if self._is_lab_research_lookup(q):
            return "knowledge_lookup", "asking about lab research / publications / result items", 0.78

        system_keywords = [
            "記憶系統", "memory", "planner", "executor", "webtools",
            "lab kb", "labkb", "clp", "persona", "系統", "專案", "模組", "agent", "架構",
        ]
        explanation_keywords = [
            "怎麼運作", "如何運作", "原理", "流程", "架構是什麼", "負責什麼",
            "做什麼", "作用", "用途", "怎麼取用", "怎麼接", "長什麼樣子",
            "長怎樣", "目前長怎樣", "現在長怎樣", "整體架構", "整體流程",
            "怎麼串起來", "如何串起來",
        ]
        project_state_cues = ["現在", "目前", "當前", "現況"]
        if any(k in q for k in system_keywords):
            if any(k in q for k in explanation_keywords):
                return "system_explanation", "asking how current system works", 0.85
            if any(k in q for k in project_state_cues):
                return "system_explanation", "asking about the current project/system state", 0.8

        return "general", "fallback general intent", 0.5

    def _is_persona_performance_prompt(self, q: str) -> bool:
        role_style_cues = [
            "角色口吻", "角色語氣", "你的風格", "你的角色方式", "自我介紹",
            "開場白", "你會怎麼回應", "你會怎麼回答", "不像真的導覽員",
            "不像真的實驗室導覽員",
        ]
        system_target_cues = [
            "本研究", "本系統",
            "應用於導覽任務之基於人格設定大型語言模型代理人架構設計",
            "design of a persona-based llm agent architecture for tour guiding tasks",
            "本研究提出之導覽代理人架構", "本研究提出的導覽代理人架構",
            "人格設定大型語言模型代理人架構", "基於人格設定大型語言模型代理人架構",
            "persona-based llm agent architecture", "導覽代理人架構",
            "導覽代理系統", "代理系統", "目前這套系統",
            "這套代理", "這個代理", "這套系統", "這個系統",
        ]
        return any(cue in q for cue in role_style_cues) and not any(cue in q for cue in system_target_cues)

    def _is_guided_system_profile_lookup(self, q: str) -> bool:
        system_targets = [
            "應用於導覽任務之基於人格設定大型語言模型代理人架構設計",
            "design of a persona-based llm agent architecture for tour guiding tasks",
            "本研究提出之導覽代理人架構",
            "本研究提出的導覽代理人架構",
            "本研究的導覽代理人架構",
            "人格設定大型語言模型代理人架構",
            "基於人格設定大型語言模型代理人架構",
            "persona-based llm agent architecture",
            "tour guiding tasks",
            "導覽代理人架構",
            "本研究提出的導覽代理系統",
            "本研究的導覽代理系統",
            "導覽代理系統",
            "本系統",
            "這套系統",
            "這個系統",
            "這套導覽代理系統",
            "這套代理",
            "這個導覽代理系統",
            "這個代理",
            "你這個導覽代理系統",
            "你這個系統",
        ]
        profile_cues = [
            "核心特色", "特色", "定位", "角色", "扮演什麼角色", "是誰",
            "差別", "有什麼差別", "和一般聊天機器人", "和一般 chatbot",
            "記憶", "記憶機制", "工具", "為什麼需要工具",
            "評估方式", "如何判斷代理表現", "為什麼重要",
            "主要目標", "核心設計", "想解決什麼問題", "最重要的兩個功能",
            "簡單介紹", "介紹", "說明", "口頭導覽", "簡單導覽",
        ]
        mechanism_cues = [
            "怎麼運作", "如何運作", "流程", "原理", "怎麼接", "怎麼串起來",
            "整體架構", "負責什麼", "模組", "實作", "修改哪些部分",
        ]
        return (
            any(t in q for t in system_targets)
            and any(c in q for c in profile_cues)
            and not any(c in q for c in mechanism_cues)
        )

    def _is_current_system_explanation_or_presentation(self, q: str) -> bool:
        current_system_targets = [
            "記憶系統", "memory", "planner", "executor", "webtools", "lab kb",
            "labkb", "clp", "persona", "模組", "架構", "專案", "agent",
            "應用於導覽任務之基於人格設定大型語言模型代理人架構設計",
            "design of a persona-based llm agent architecture for tour guiding tasks",
            "本研究提出之導覽代理人架構", "本研究提出的導覽代理人架構",
            "人格設定大型語言模型代理人架構", "基於人格設定大型語言模型代理人架構",
            "persona-based llm agent architecture", "導覽代理人架構",
            "目前這套系統", "本系統", "這套系統", "這個系統",
        ]
        explanation_cues = [
            "怎麼運作", "如何運作", "原理", "流程", "架構是什麼", "負責什麼",
            "做什麼", "作用", "用途", "怎麼取用", "怎麼接", "長什麼樣子",
            "長怎樣", "目前長怎樣", "現在長怎樣", "整體架構", "整體流程",
            "怎麼串起來", "如何串起來",
        ]
        return any(t in q for t in current_system_targets) and any(c in q for c in explanation_cues)

    def _is_lab_research_lookup(self, q: str) -> bool:
        lab_targets = [
            "實驗室",
            "這間實驗室",
            "這個實驗室",
            "你們實驗室",
            "研究室",
            "ai lab",
            "lab",
            "我們實驗室",
            "我們 ai lab",
            "本實驗室",
            "622",
        ]

        lab_knowledge_cues = [
            "研究成果",
            "研究方向",
            "研究內容",
            "目前做過的研究",
            "主要在做什麼",
            "目前主要在做什麼",
            "目前在做什麼",
            "做什麼",
            "介紹",
            "說明",
            "有哪些",
            "列出",
            "整理",
            "比較",
            "論文",
            "paper",
            "publication",
            "成果",

            # 參觀學生導覽情境
            "帶我們認識",
            "帶我認識",
            "認識這間實驗室",
            "認識實驗室",
            "介紹這間實驗室",
            "介紹一下這間實驗室",
            "帶我們參觀",
            "帶我參觀",
            "導覽",
        ]

        current_system_targets = [
            "應用於導覽任務之基於人格設定大型語言模型代理人架構設計",
            "design of a persona-based llm agent architecture for tour guiding tasks",
            "本研究提出之導覽代理人架構",
            "本研究提出的導覽代理人架構",
            "本研究的導覽代理人架構",
            "人格設定大型語言模型代理人架構",
            "基於人格設定大型語言模型代理人架構",
            "persona-based llm agent architecture",
            "本研究提出的導覽代理系統",
            "本研究的導覽代理系統",
            "本系統",
            "導覽代理系統",
            "代理系統",
            "這套代理",
            "這個代理",
            "這套系統",
            "這個系統",
        ]

        return (
            any(t in q for t in lab_targets)
            and any(c in q for c in lab_knowledge_cues)
            and not any(s in q for s in current_system_targets)
        )