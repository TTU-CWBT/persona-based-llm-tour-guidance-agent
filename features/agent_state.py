from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import uuid


@dataclass
class TaskThread:
    task_id: str
    task_type: str
    task_subject: str
    source_of_truth: str
    status: str = "active"          # active / done / blocked / background
    turn_mode: str = "initial"      # initial / clarify / expand / compare / shift / close
    last_query: str = ""
    last_successful_tool: str = ""
    last_successful_observation_summary: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    active_task: Optional[str] = None
    user_goal: Optional[str] = None
    task_status: str = "idle"          # idle / active / blocked / done
    last_action: Optional[str] = None
    confidence: float = 0.0
    open_loops: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)
    current_thread: Optional[TaskThread] = None
    thread_history: List[TaskThread] = field(default_factory=list)


def build_agent_state(memory, user_prompt: str) -> AgentState:
    """
    Phase 1 continuity 版：
    - 從 memory 撈 open loops
    - 若 memory 暫存了 current_task_thread，就接回 AgentState
    - active_task 仍保留本輪問題，避免舊邏輯失效
    """
    open_loops = []
    state = AgentState(
        active_task=user_prompt.strip() if user_prompt else None,
        user_goal=user_prompt.strip() if user_prompt else None,
        task_status="active" if user_prompt else "idle",
        last_action=None,
        confidence=0.0,
        open_loops=[],
        notes={},
    )

    if memory is not None and hasattr(memory, "get_open_loops"):
        try:
            open_loops = memory.get_open_loops()
        except Exception:
            open_loops = []
    state.open_loops = open_loops or []

    if memory is not None:
        try:
            thread = getattr(memory, "current_task_thread", None)
            if isinstance(thread, TaskThread):
                state.current_thread = thread
            elif isinstance(thread, dict) and thread.get("task_type"):
                state.current_thread = TaskThread(**thread)
        except Exception:
            pass
        try:
            hist = getattr(memory, "task_thread_history", None) or []
            converted = []
            for item in hist:
                if isinstance(item, TaskThread):
                    converted.append(item)
                elif isinstance(item, dict) and item.get("task_type"):
                    converted.append(TaskThread(**item))
            state.thread_history = converted
        except Exception:
            pass

    return state


def summarize_observation_for_thread(observation: dict | None) -> str:
    if not isinstance(observation, dict):
        return ""
    parts = []
    tool = observation.get("tool")
    if tool:
        parts.append(f"tool={tool}")
    for key in ("query", "path", "title"):
        val = observation.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}={val.strip()[:120]}")
            break
    ctx = observation.get("context", "") or ""
    if ctx.strip():
        parts.append(ctx.strip().replace("\n", " ")[:220])
    return " | ".join(parts)[:400]


def _normalize_task_type(task_type: str | None, planner_action=None) -> str:
    t = (task_type or "").strip().lower()
    if t:
        return t
    action_type = getattr(planner_action, "action_type", "") or ""
    if action_type == "retrieve_lab_kb":
        return "lab_research_lookup"
    if action_type == "project_context":
        return "system_or_project_context"
    if action_type == "retrieve_web":
        return "external_info"
    return "general"


def update_task_thread(
    agent_state: AgentState,
    user_prompt: str,
    continuity=None,
    planner_action=None,
    final_observation: dict | None = None,
    evaluation=None,
) -> AgentState:
    if agent_state is None:
        return agent_state

    usable = bool(getattr(evaluation, "usable", False)) if evaluation is not None else True
    if not usable:
        return agent_state

    obs = final_observation or {}
    obs_tool = (obs.get("tool") or getattr(planner_action, "action_type", "direct_answer") or "direct_answer")
    obs_summary = summarize_observation_for_thread(obs)

    relation = getattr(continuity, "relation", "new_task") if continuity is not None else "new_task"
    is_cont = bool(getattr(continuity, "is_continuation", False)) if continuity is not None else False
    continued_task_type = getattr(continuity, "continued_task_type", None) if continuity is not None else None
    continued_subject = getattr(continuity, "continued_task_subject", None) if continuity is not None else None
    conf = float(getattr(continuity, "confidence", 0.0)) if continuity is not None else float(getattr(planner_action, "confidence", 0.0) or 0.0)

    if is_cont and agent_state.current_thread is not None:
        th = agent_state.current_thread
        th.turn_mode = relation or th.turn_mode
        th.last_query = (obs.get("query") or getattr(planner_action, "query", None) or user_prompt or "").strip()
        th.last_successful_tool = obs_tool
        th.last_successful_observation_summary = obs_summary or th.last_successful_observation_summary
        th.source_of_truth = obs_tool or th.source_of_truth
        th.confidence = max(th.confidence, conf)
        if continued_task_type:
            th.task_type = continued_task_type
        if continued_subject:
            th.task_subject = continued_subject
        return agent_state

    if agent_state.current_thread is not None:
        old = agent_state.current_thread
        old.status = "background"
        agent_state.thread_history.append(old)

    action_meta = (getattr(planner_action, "meta", {}) or {})
    task_type = _normalize_task_type(action_meta.get("task_type") or action_meta.get("intent"), planner_action)
    task_subject = (
        getattr(continuity, "continued_task_subject", None)
        or (obs.get("query") if isinstance(obs, dict) else None)
        or getattr(planner_action, "query", None)
        or user_prompt
    )
    agent_state.current_thread = TaskThread(
        task_id=str(uuid.uuid4())[:8],
        task_type=task_type,
        task_subject=(task_subject or user_prompt or "").strip(),
        source_of_truth=obs_tool,
        status="active",
        turn_mode=(relation if relation != "new_task" else "initial"),
        last_query=(obs.get("query") or getattr(planner_action, "query", None) or user_prompt or "").strip(),
        last_successful_tool=obs_tool,
        last_successful_observation_summary=obs_summary,
        confidence=max(conf, float(getattr(planner_action, "confidence", 0.0) or 0.0)),
        metadata={},
    )
    return agent_state
