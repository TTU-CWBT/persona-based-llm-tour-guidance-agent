# features/task_scope_builder.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class TaskScope:
    task_type: str
    target_scope: str
    in_scope_files: List[str]
    out_of_scope_domains: List[str]
    allowed_operations: List[str]
    preferred_answer_shape: List[str]
    evidence_basis: str

    # 原本 plan_skeleton_builder 的內容，直接併進來
    candidate_files: List[str]
    known_symbols: Dict[str, List[str]]
    allowed_change_points: List[str]
    forbidden_inventions: List[str]


class TaskScopeBuilder:
    """
    合併版：
    - 建立任務範圍（scope）
    - 建立可安全規劃的骨架（skeleton）
    """

    def build(self, user_prompt: str, intent: str, observation: dict | None) -> TaskScope:
        obs = observation or {}
        tool = obs.get("tool", "")
        files_used = obs.get("files_used", []) or []
        ctx = obs.get("context", "") or ""

        # 先抽目前 observation 中已知的符號
        known_symbols: Dict[str, List[str]] = {}
        for f in files_used:
            if f == "CLP.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    [
                        "generate_once",
                        "_plan_primary_action",
                        "_execute_primary_action",
                        "_build_fallback_observation",
                        "build_system",
                        "run_cli",
                        "_looks_incomplete",
                    ],
                )
            elif f == "memory_repository.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    [
                        "ConversationMemory",
                        "Turn",
                        "MemoryItem",
                        "add_task",
                        "add_open_loop",
                        "get_open_loops",
                        "get_tasks",
                        "build_memory_context",
                    ],
                )
            elif f == "agent_state.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    ["AgentState", "build_agent_state", "active_task", "open_loops", "confidence"],
                )
            elif f == "agent_planner.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    ["AgentPlanner", "AgentAction", "decide", "_infer_intent"],
                )
            elif f == "agent_executor.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    ["execute_action", "retrieve_lab_kb", "retrieve_web", "ask_followup", "proactive_hint"],
                )
            elif f == "observation_evaluator.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    [
                        "ObservationEvaluator",
                        "EvaluationResult",
                        "_judge_if_available",
                        "evaluate",
                    ],
                )
            elif f == "webtools.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    ["WebToolManager", "search_structured"],
                )
            elif f == "labkb.py":
                known_symbols[f] = self._collect_symbols(
                    ctx,
                    ["LabKB", "retrieve_structured", "context", "augment_query"],
                )
            else:
                known_symbols[f] = []

        forbidden_inventions = [
            "把 observation 中未出現的新檔案當成既有實作",
            "把 observation 中未出現的新函式或新類別當成既有實作",
            "把 observation 中未出現的新工具名稱當成既有實作",
            "把 observation 中未出現的新欄位或新評分機制當成既有實作",
        ]

        # ---- design_planning ----
        if intent == "design_planning":
            return TaskScope(
                task_type="system_architecture_modification",
                target_scope="current Python agent system",
                in_scope_files=files_used,
                out_of_scope_domains=[
                    "website frontend",
                    "UI layout",
                    "visual design",
                    "homepage / navigation bar / FAQ page",
                    "marketing copy",
                ],
                allowed_operations=[
                    "modify existing modules",
                    "adjust routing logic",
                    "refine evaluation logic",
                    "extend memory structure",
                    "improve project context extraction",
                    "change tool orchestration flow",
                ],
                preferred_answer_shape=[
                    "先改哪個檔案",
                    "再改哪些函式或類別",
                    "修改目的",
                    "驗證方式",
                ],
                evidence_basis=f"derived from {tool} observation and currently retrieved project files",

                candidate_files=files_used,
                known_symbols=known_symbols,
                allowed_change_points=self._build_allowed_change_points(
                    intent=intent,
                    files_used=files_used,
                    known_symbols=known_symbols,
                ),
                forbidden_inventions=forbidden_inventions,
            )

        # ---- system_explanation ----
        if intent == "system_explanation":
            return TaskScope(
                task_type="current_system_mechanism_explanation",
                target_scope="current Python agent system implementation",
                in_scope_files=files_used,
                out_of_scope_domains=[
                    "general cognitive theory",
                    "generic AI memory theory",
                    "unverified external architecture",
                ],
                allowed_operations=[
                    "describe current modules",
                    "describe current data flow",
                    "describe current memory / retrieval / routing behavior",
                ],
                preferred_answer_shape=[
                    "目前由哪些模組負責",
                    "資料如何流動",
                    "目前怎麼運作",
                ],
                evidence_basis=f"derived from {tool} observation and currently retrieved project files",

                candidate_files=files_used,
                known_symbols=known_symbols,
                allowed_change_points=[],
                forbidden_inventions=forbidden_inventions,
            )

        # ---- external_fresh_info ----
        if intent == "external_fresh_info":
            return TaskScope(
                task_type="external_information_summary",
                target_scope="current external information only",
                in_scope_files=[],
                out_of_scope_domains=[
                    "internal project architecture",
                    "unsupported speculation",
                ],
                allowed_operations=[
                    "summarize retrieved external items",
                    "compare retrieved items",
                ],
                preferred_answer_shape=[
                    "先列 observation 中出現的項目",
                    "再補充簡短說明",
                ],
                evidence_basis=f"derived from {tool} observation",

                candidate_files=[],
                known_symbols={},
                allowed_change_points=[],
                forbidden_inventions=forbidden_inventions,
            )

        # ---- fallback ----
        return TaskScope(
            task_type="general_answering",
            target_scope="current conversation request",
            in_scope_files=files_used,
            out_of_scope_domains=[],
            allowed_operations=["answer conservatively based on evidence"],
            preferred_answer_shape=["先回答核心問題", "必要時補充說明"],
            evidence_basis=f"derived from {tool} observation",

            candidate_files=files_used,
            known_symbols=known_symbols,
            allowed_change_points=[],
            forbidden_inventions=forbidden_inventions,
        )

    def to_dict(self, scope: TaskScope) -> dict:
        return asdict(scope)

    def _collect_symbols(self, context: str, candidates: List[str]) -> List[str]:
        text = context or ""
        found = []
        for c in candidates:
            if c in text:
                found.append(c)
        return found

    def _build_allowed_change_points(
        self,
        intent: str,
        files_used: List[str],
        known_symbols: Dict[str, List[str]],
    ) -> List[str]:
        points: List[str] = []

        if intent != "design_planning":
            return points

        if "CLP.py" in files_used:
            syms = known_symbols.get("CLP.py", [])
            if "generate_once" in syms:
                points.append("調整 CLP.py 的 generate_once 主流程接線")
            if "_plan_primary_action" in syms:
                points.append("調整 CLP.py 的 _plan_primary_action 決策接線")
            if "_execute_primary_action" in syms:
                points.append("調整 CLP.py 的 _execute_primary_action 工具執行接線")
            if "_build_fallback_observation" in syms:
                points.append("調整 CLP.py 的 _build_fallback_observation fallback 流程")
            if "build_system" in syms:
                points.append("調整 CLP.py 的 build_system prompt 組裝")

        if "agent_planner.py" in files_used:
            syms = known_symbols.get("agent_planner.py", [])
            if "decide" in syms:
                points.append("調整 agent_planner.py 的 decide action 決策")
            if "_infer_intent" in syms:
                points.append("收斂 agent_planner.py 的 intent 推論與 action 映射")

        if "agent_executor.py" in files_used:
            syms = known_symbols.get("agent_executor.py", [])
            if "execute_action" in syms:
                points.append("調整 agent_executor.py 的 execute_action 執行邊界")

        if "observation_evaluator.py" in files_used:
            syms = known_symbols.get("observation_evaluator.py", [])
            if "evaluate" in syms:
                points.append("調整 observation_evaluator.py 的 evaluate 可用性判斷")

        if "memory_repository.py" in files_used:
            syms = known_symbols.get("memory_repository.py", [])
            if "ConversationMemory" in syms:
                points.append("擴充 memory_repository.py 的 ConversationMemory 記憶結構")
            if "add_task" in syms or "add_open_loop" in syms:
                points.append("補強 memory_repository.py 的 task/open_loop 使用方式")

        if "webtools.py" in files_used:
            syms = known_symbols.get("webtools.py", [])
            if "search_structured" in syms:
                points.append("調整 webtools.py 的 search_structured observation 輸出")

        if "labkb.py" in files_used:
            syms = known_symbols.get("labkb.py", [])
            if "retrieve_structured" in syms:
                points.append("調整 labkb.py 的 retrieve_structured 結構化輸出")

        return points