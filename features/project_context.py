from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class ProjectFileSpec:
    name: str
    candidates: List[str]
    purpose: str


class ProjectContextProvider:
    """
    Phase 1 收斂版：
    - 依 intent 選檔案
    - 優先對齊目前 agent 主幹
    - 支援 features/ 與專案根目錄兩種路徑結構
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()

        self.file_specs: List[ProjectFileSpec] = [
            ProjectFileSpec("CLP.py", ["CLP.py"], "主流程與 orchestration"),
            ProjectFileSpec("agent_state.py", ["features/agent_state.py", "agent_state.py"], "agent 狀態"),
            ProjectFileSpec("agent_planner.py", ["features/agent_planner.py", "agent_planner.py"], "action 決策"),
            ProjectFileSpec("agent_executor.py", ["features/agent_executor.py", "agent_executor.py"], "action 執行"),
            ProjectFileSpec("observation_evaluator.py", ["features/observation_evaluator.py", "observation_evaluator.py"], "observation 評估"),
            ProjectFileSpec("task_scope_builder.py", ["features/task_scope_builder.py", "task_scope_builder.py"], "回答邊界與骨架"),
            ProjectFileSpec("agent_trace.py", ["features/agent_trace.py", "agent_trace.py"], "trace 紀錄"),
            ProjectFileSpec("memory_repository.py", ["memory_repository.py", "features/memory_repository.py"], "記憶系統"),
            ProjectFileSpec("webtools.py", ["features/webtools.py", "webtools.py"], "外部搜尋工具"),
            ProjectFileSpec("labkb.py", ["features/labkb.py", "labkb.py"], "lab KB 封裝"),
        ]

    def _resolve_existing_path(self, spec: ProjectFileSpec) -> Optional[Path]:
        for rel in spec.candidates:
            abs_path = self.project_root / rel
            if abs_path.exists():
                return abs_path
        return None

    def summarize_known_files(self, debug: bool = False) -> dict:
        rows = []
        for spec in self.file_specs:
            abs_path = self._resolve_existing_path(spec)
            rows.append({
                "name": spec.name,
                "path": str(abs_path) if abs_path else None,
                "exists": bool(abs_path),
                "purpose": spec.purpose,
            })
        return {"tool": "project_context", "files": rows, "num_files": len(rows)}

    def retrieve_structured(self, user_query: str, intent: Optional[str] = None, debug: bool = False) -> dict:
        chosen = self._select_files(user_query=user_query, intent=intent)
        blocks = []
        files_used = []

        for spec in chosen:
            abs_path = self._resolve_existing_path(spec)
            if abs_path is None:
                continue
            try:
                text = abs_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            summary = self._summarize_file(spec, text, user_query=user_query, intent=intent)
            if summary.strip():
                files_used.append(spec.name)
                blocks.append(summary)

        context_parts = []
        if intent == "design_planning":
            overview = self._build_design_overview(chosen)
            if overview.strip():
                context_parts.append(overview)
        context_parts.extend(blocks)

        return {
            "tool": "project_context",
            "query": user_query,
            "intent": intent,
            "files_used": files_used,
            "num_files": len(files_used),
            "context": "\n\n".join(context_parts),
            "usable_hint": len(files_used) > 0,
        }

    def _build_design_overview(self, chosen: List[ProjectFileSpec]) -> str:
        focus_map = {
            "CLP.py": "單輪主流程、generate_once、build_system",
            "agent_state.py": "active_task、open_loops、confidence 等狀態欄位",
            "agent_planner.py": "intent-aware action 決策中心",
            "agent_executor.py": "retrieve_lab_kb / retrieve_web / ask_followup 等執行入口",
            "observation_evaluator.py": "observation 可用性判斷與 fallback",
            "task_scope_builder.py": "scope / allowed_change_points / forbidden_inventions",
            "memory_repository.py": "記憶結構、長短期記憶與檢索",
            "webtools.py": "外部搜尋 structured observation 封裝",
            "labkb.py": "lab KB structured retrieval / context",
        }
        lines = ["[可優先修改的檔案]"]
        for spec in chosen:
            desc = focus_map.get(spec.name)
            if desc:
                lines.append(f"- {spec.name}：{desc}")
        return "\n".join(lines)

    def _select_files(self, user_query: str, intent: Optional[str]) -> List[ProjectFileSpec]:
        q = (user_query or "").lower()
        intent = intent or ""

        if intent == "system_explanation":
            if "記憶" in q or "memory" in q:
                names = ["memory_repository.py", "CLP.py", "agent_state.py"]
            elif "web" in q or "搜尋" in q:
                names = ["webtools.py", "agent_executor.py", "CLP.py"]
            elif "lab kb" in q or "labkb" in q:
                names = ["labkb.py", "agent_executor.py", "CLP.py"]
            elif "planner" in q or "agent" in q:
                names = ["agent_planner.py", "agent_executor.py", "agent_state.py", "CLP.py"]
            else:
                names = [
                    "CLP.py", "agent_state.py", "agent_planner.py", "agent_executor.py",
                    "observation_evaluator.py", "task_scope_builder.py",
                    "memory_repository.py", "webtools.py", "labkb.py",
                ]
            return [s for s in self.file_specs if s.name in names]

        if intent == "design_planning":
            names = [
                "CLP.py", "agent_state.py", "agent_planner.py", "agent_executor.py",
                "observation_evaluator.py", "task_scope_builder.py",
                "memory_repository.py", "webtools.py", "labkb.py",
            ]
            return [s for s in self.file_specs if s.name in names]

        names = ["CLP.py", "agent_planner.py", "agent_executor.py", "memory_repository.py", "webtools.py"]
        return [s for s in self.file_specs if s.name in names]

    def _summarize_file(self, spec: ProjectFileSpec, text: str, user_query: str, intent: Optional[str]) -> str:
        defs = self._extract_defs(text, max_defs=14)
        snippets = self._extract_design_snippets(spec.name, text, user_query=user_query, max_snippets=4) if intent == "design_planning" else self._extract_explanation_snippets(spec.name, text, user_query=user_query, max_snippets=3)

        parts = [f"[檔案] {spec.name}", f"- 用途：{spec.purpose}"]
        if defs:
            parts.append("- 主要結構：")
            for d in defs[:8]:
                parts.append(f"  - {d}")
        if snippets:
            parts.append("- 可修改相關片段：" if intent == "design_planning" else "- 相關片段：")
            parts.extend(snippets)
        return "\n".join(parts)

    def _extract_defs(self, text: str, max_defs: int = 12) -> List[str]:
        defs = []
        for ln in text.splitlines():
            stripped = ln.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                defs.append(stripped)
            if len(defs) >= max_defs:
                break
        return defs

    def _extract_explanation_snippets(self, file_name: str, text: str, user_query: str, max_snippets: int = 3) -> List[str]:
        q = (user_query or "").lower()
        file_keywords = {
            "memory_repository.py": ["conversationmemory", "build_memory_context", "add_user", "add_assistant", "summary", "turn", "task", "open_loop", "memory"],
            "CLP.py": ["generate_once", "build_system", "_plan_primary_action", "_execute_primary_action", "intent", "evaluation", "task_scope"],
            "agent_state.py": ["AgentState", "build_agent_state", "active_task", "open_loops", "confidence"],
            "agent_planner.py": ["AgentPlanner", "AgentAction", "decide", "_infer_intent"],
            "agent_executor.py": ["execute_action", "retrieve_lab_kb", "retrieve_web", "ask_followup", "proactive_hint"],
            "webtools.py": ["search_structured", "WebToolManager"],
            "labkb.py": ["context", "LabKB", "augment_query"],
            "task_scope_builder.py": ["TaskScopeBuilder", "build", "allowed_change_points"],
        }
        keywords = file_keywords.get(file_name, [])
        if "記憶" in q or "memory" in q:
            keywords += ["memory", "summary", "turn", "build_memory_context"]
        return self._grab_snippets_by_keywords(text, keywords, max_snippets=max_snippets)

    def _extract_design_snippets(self, file_name: str, text: str, user_query: str, max_snippets: int = 4) -> List[str]:
        design_targets = {
            "CLP.py": ["generate_once", "_plan_primary_action", "_execute_primary_action", "_build_fallback_observation", "build_system"],
            "agent_state.py": ["AgentState", "build_agent_state"],
            "agent_planner.py": ["class AgentPlanner", "def decide", "def _infer_intent", "class AgentAction"],
            "agent_executor.py": ["execute_action", "retrieve_lab_kb", "retrieve_web", "ask_followup", "proactive_hint"],
            "observation_evaluator.py": ["class ObservationEvaluator", "def evaluate", "ObservationJudge"],
            "task_scope_builder.py": ["class TaskScopeBuilder", "def build", "allowed_change_points"],
            "webtools.py": ["search_structured", "WebToolManager"],
            "labkb.py": ["context", "LabKB", "augment_query"],
            "memory_repository.py": ["ConversationMemory", "build_memory_context", "add_task", "add_open_loop", "get_open_loops", "get_tasks"],
        }
        return self._grab_snippets_by_keywords(text, design_targets.get(file_name, []), max_snippets=max_snippets)

    def _grab_snippets_by_keywords(self, text: str, keywords: List[str], max_snippets: int = 3) -> List[str]:
        if not keywords:
            return []
        lines = text.splitlines()
        hits = []
        seen = set()
        for i, ln in enumerate(lines):
            low = ln.lower()
            for k in keywords:
                if k.lower() in low:
                    start = max(0, i - 2)
                    end = min(len(lines), i + 5)
                    snippet = "\n".join(lines[start:end]).strip()
                    key = snippet[:200]
                    if snippet and key not in seen:
                        seen.add(key)
                        hits.append(snippet)
                    break
            if len(hits) >= max_snippets:
                break
        return [f"```python\n{sn}\n```" for sn in hits[:max_snippets]]
