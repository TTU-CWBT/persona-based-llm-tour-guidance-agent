from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from features.observation_judge import ObservationJudge


@dataclass
class EvaluationResult:
    usable: bool
    reason: str
    next_action: Optional[str] = None
    source_assessment: list[dict] = field(default_factory=list)


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def _count_matches(text: str, keywords: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in keywords if k.lower() in t)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_strategy_task_type(strategy) -> Optional[str]:
    task_type = getattr(strategy, "task_type", None)
    if task_type:
        return task_type

    meta = getattr(strategy, "meta", None)
    if isinstance(meta, dict):
        return meta.get("task_type") or meta.get("intent")

    return None


def _lab_project_signal_score(text: str) -> int:
    project_terms = [
        "clp", "memory_repository", "conversationmemory", "webtools",
        "labkb", "lab kb", "planner", "executor", "intent", "strategy",
        "build_memory_context", "add_user", "add_assistant", "agentaction",
        "turn", "summary", "task", "open_loop", "模組", "系統", "流程", "架構",
        "observation", "evaluation", "project_context", "retrieve_structured",
    ]
    return _count_matches(text, project_terms)


def _lab_research_signal_score(text: str) -> int:
    research_terms = [
        "研究", "成果", "論文", "實驗", "方法", "模型", "系統", "資料集",
        "paper", "abstract", "introduction", "method", "methods", "result", "results",
        "experiment", "experiments", "dataset", "evaluation", "conference", "journal",
        "doi", "accuracy", "f1", "precision", "recall",
    ]
    return _count_matches(text, research_terms)




def _source_text_signal(text: str) -> int:
    t = (text or "").lower()
    strong = [
        "abstract", "this paper", "we propose", "we present", "we introduce",
        "system framework", "workflow", "prototype", "proof-of-concept", "poc",
        "concluding remarks", "conclusion", "introduction",
    ]
    weak = [
        "method", "experiment", "evaluation", "dataset", "application", "design learning",
    ]
    score = sum(2 for term in strong if term in t)
    score += sum(1 for term in weak if term in t)
    return score


def _classify_source_candidate(candidate: dict) -> dict:
    ownership_hint = candidate.get("ownership_hint", "unclear")
    best_score = float(candidate.get("best_score", 0.0) or 0.0)
    summary_score = int(candidate.get("best_summary_chunk_score", 0) or 0)
    ownership_signal = int(candidate.get("ownership_signal_score", 0) or 0)
    text = candidate.get("representative_text", "") or ""
    text_signal = _source_text_signal(text)

    if (
        ownership_hint in {"likely_lab_authored", "possibly_lab_authored"}
        and best_score >= 0.43
        and (summary_score >= 3 or text_signal >= 4 or ownership_signal >= 5)
    ):
        level = "elaborate"
        rationale = "source has enough ownership and content signals to support a fuller research introduction"
    elif best_score >= 0.42 and (summary_score >= 0 or text_signal >= 1 or ownership_signal >= 1):
        level = "mention_only"
        rationale = "source appears relevant but evidence is insufficient for a full research writeup"
    else:
        level = "ignore"
        rationale = "source evidence is too weak or too fragmentary to use safely"

    return {
        "rel_path": candidate.get("rel_path", ""),
        "ownership_hint": ownership_hint,
        "best_score": best_score,
        "best_summary_chunk_score": summary_score,
        "ownership_signal_score": ownership_signal,
        "evidence_level": level,
        "rationale": rationale,
    }


def _assess_source_candidates(observation: dict) -> list[dict]:
    return [_classify_source_candidate(c) for c in (observation.get("source_candidates") or [])]

def _is_research_expansion_turn(intent: str, strategy, continuity) -> bool:
    relation = getattr(continuity, "relation", "") if continuity else ""
    task_type = getattr(strategy, "task_type", None)
    return bool(
        continuity
        and getattr(continuity, "is_continuation", False)
        and relation in {"expand", "compare"}
        and (intent == "knowledge_lookup" or task_type == "lab_research_lookup")
    )


def _lab_kb_expansion_evaluation(observation: dict) -> EvaluationResult:
    num_unique_sources = int(observation.get("num_unique_sources", 0) or 0)
    source_assessment = _assess_source_candidates(observation)
    elaborate_sources = [s for s in source_assessment if s.get("evidence_level") == "elaborate"]
    mention_only_sources = [s for s in source_assessment if s.get("evidence_level") == "mention_only"]

    if num_unique_sources >= 2 and elaborate_sources:
        if mention_only_sources:
            return EvaluationResult(
                True,
                "lab_kb expansion has one elaboratable source and additional mention-only candidates; only elaborate sources may be expanded",
                None,
                source_assessment,
            )
        return EvaluationResult(
            True,
            "lab_kb expansion has multiple sources with at least one credible elaboratable lab candidate",
            None,
            source_assessment,
        )

    if num_unique_sources >= 2 and mention_only_sources:
        return EvaluationResult(
            True,
            "lab_kb expansion has only mention-only additional sources; answer must stay conservative and must not invent titles or details",
            None,
            source_assessment,
        )

    if num_unique_sources == 1 and elaborate_sources:
        return EvaluationResult(
            False,
            "lab_kb expansion only repeated one elaboratable source; not enough evidence for 'other research' expansion",
            None,
            source_assessment,
        )

    return EvaluationResult(
        False,
        "lab_kb expansion lacks source-level evidence for additional research items",
        None,
        source_assessment,
    )


class ObservationEvaluator:
    """
    長期版：
    - 先用規則粗篩
    - 灰區時交給 LLM judge
    """

    def __init__(self, backend=None):
        self.backend = backend
        self.judge = ObservationJudge(backend) if backend is not None else None

    def _judge_if_available(self, user_prompt: str, intent: str, strategy, observation: dict) -> EvaluationResult:
        if self.judge is None:
            return EvaluationResult(False, "judge unavailable", strategy.fallback_tool)

        jr = self.judge.judge(
            user_prompt=user_prompt,
            intent=intent,
            observation=observation,
            fallback_tool=strategy.fallback_tool,
        )
        return EvaluationResult(
            usable=jr.usable,
            reason=f"[judge] {jr.reason}",
            next_action=jr.next_action,
        )

    def evaluate(self, user_prompt: str, intent: str, strategy, observation: dict | None, continuity=None) -> EvaluationResult:
        if continuity and getattr(continuity, "is_continuation", False):
            relation = getattr(continuity, "relation", "")
            tool0 = (observation or {}).get("tool") if isinstance(observation, dict) else None
            if relation in {"expand", "compare"} and tool0 == "direct_answer":
                return EvaluationResult(False, "continuation expand/compare turn should not fall back to direct_answer without evidence", strategy.fallback_tool)

        if not strategy.must_verify:
            return EvaluationResult(True, "verification not required", None)

        if not observation:
            return EvaluationResult(False, "empty observation", strategy.fallback_tool)

        tool = observation.get("tool")

        # -----------------------
        # lab_kb 檢查
        # -----------------------
        if tool == "lab_kb":
            num_hits = _safe_int(observation.get("num_hits", 0), 0)
            top_score = _safe_float(observation.get("top_score", 0.0), 0.0)
            best_score = _safe_float(observation.get("best_score", top_score), top_score)
            first_score = _safe_float(observation.get("first_score", top_score), top_score)
            avg_score = _safe_float(observation.get("avg_score", 0.0), 0.0)
            ctx = observation.get("context", "") or ""
            err = observation.get("error")
            task_type = _get_strategy_task_type(strategy)

            if err:
                return EvaluationResult(False, f"lab_kb error: {err}", strategy.fallback_tool)

            if num_hits <= 0:
                return EvaluationResult(False, "lab_kb returned no hits", strategy.fallback_tool)

            if not ctx.strip():
                return EvaluationResult(False, "lab_kb context empty", strategy.fallback_tool)

            project_score = _lab_project_signal_score(ctx)
            research_score = _lab_research_signal_score(ctx)
            source_assessment = _assess_source_candidates(observation)
            usable_sources = [
                s for s in source_assessment
                if s.get("evidence_level") in {"elaborate", "mention_only"}
            ]

            # 不再只用單一 top_score 做硬性擋門檻。
            # labkb.py 的 rerank 會為了來源多樣化調整順序，因此第一筆 hit
            # 不必然代表整個 observation 的最佳證據。
            # 只有在「分數、命中數、內容訊號、來源證據」都偏弱時，才視為不可用。
            if (
                best_score < 0.45
                and num_hits < 2
                and research_score < 2
                and project_score < 2
                and not usable_sources
            ):
                return EvaluationResult(
                    False,
                    (
                        "lab_kb evidence too weak "
                        f"(best_score={best_score:.3f}, first_score={first_score:.3f}, "
                        f"avg_score={avg_score:.3f}, num_hits={num_hits}, "
                        f"research_score={research_score}, project_score={project_score})"
                    ),
                    strategy.fallback_tool,
                    source_assessment,
                )

            if intent == "knowledge_lookup" or task_type == "lab_research_lookup":
                if _is_research_expansion_turn(intent, strategy, continuity):
                    expansion_result = _lab_kb_expansion_evaluation(observation)
                    if expansion_result.usable:
                        return expansion_result
                    return EvaluationResult(False, expansion_result.reason, strategy.fallback_tool, expansion_result.source_assessment)

                if research_score >= 2 or num_hits >= 2 or best_score >= 0.60 or usable_sources:
                    return EvaluationResult(
                        True,
                        (
                            "lab_kb looks usable for research / knowledge lookup "
                            f"(best_score={best_score:.3f}, num_hits={num_hits}, "
                            f"research_score={research_score})"
                        ),
                        None,
                        source_assessment,
                    )
                return self._judge_if_available(user_prompt, intent, strategy, observation)

            if intent == "system_explanation":
                if project_score >= 6:
                    return EvaluationResult(True, "lab_kb strongly matches current system explanation", None, source_assessment)
                if research_score >= project_score + 2 and project_score < 3:
                    return EvaluationResult(False, "lab_kb looks like research content, not current system explanation", strategy.fallback_tool, source_assessment)
                return self._judge_if_available(user_prompt, intent, strategy, observation)

            if intent == "design_planning":
                planning_terms = [
                    "修改", "實作", "步驟", "檔案", "架構", "模組",
                    "implement", "architecture", "module", "function"
                ]
                planning_score = _count_matches(ctx, planning_terms)
                if project_score >= 6 and planning_score >= 3:
                    return EvaluationResult(True, "lab_kb strongly matches implementation planning", None, source_assessment)
                if research_score >= project_score + 2 and project_score < 3:
                    return EvaluationResult(False, "lab_kb looks like research content, not implementation planning evidence", strategy.fallback_tool, source_assessment)
                return self._judge_if_available(user_prompt, intent, strategy, observation)

            return self._judge_if_available(user_prompt, intent, strategy, observation)

        # -----------------------
        # project_context 檢查
        # -----------------------
        if tool == "project_context":
            num_files = observation.get("num_files", 0)
            ctx = observation.get("context", "") or ""

            if num_files <= 0:
                return EvaluationResult(False, "project_context returned no files", strategy.fallback_tool)

            if not ctx.strip():
                return EvaluationResult(False, "project_context context empty", strategy.fallback_tool)

            return EvaluationResult(True, "project_context looks usable", None)

        # -----------------------
        # web 檢查
        # -----------------------
        if tool == "web":
            num_results = observation.get("num_results", 0)
            ctx = observation.get("context", "") or ""
            err = observation.get("error")
            query = observation.get("query", "") or user_prompt

            if err:
                return EvaluationResult(False, f"web error: {err}", strategy.fallback_tool)

            if num_results <= 0:
                return EvaluationResult(False, "web returned no results", strategy.fallback_tool)

            if not ctx.strip():
                return EvaluationResult(False, "web context empty", strategy.fallback_tool)

            system_terms = [
                "clp", "planner", "executor", "webtools", "labkb", "memory",
                "agent", "專案", "系統", "模組", "架構", "流程",
            ]
            external_terms = [
                "最新", "最近", "今年", "價格", "售價", "上市", "發售", "新聞",
                "github", "paper", "論文", "conference", "版本", "官方網站", "下載", "安裝",
                "latest", "price", "release", "news", "version", "download", "install",
            ]
            low_signal_terms = [
                "什麼是系統架構圖", "架構圖", "medium", "processon", "instagram",
                "drawing tutorial", "tutorial", "範例", "教學",
            ]

            asks_about_current_system = _contains_any(query, system_terms) and _contains_any(query, ["現在", "目前", "當前", "現況", "長什麼樣子", "長怎樣", "整體架構", "怎麼運作", "流程"])
            web_has_project_signal = _count_matches(ctx, system_terms) >= 4
            web_looks_generic = _count_matches(ctx, low_signal_terms) >= 2
            clearly_external = _contains_any(query, external_terms)

            if asks_about_current_system and not clearly_external:
                if not web_has_project_signal:
                    return EvaluationResult(False, "web looks unrelated to the current project/system question", strategy.fallback_tool)
                if web_looks_generic:
                    return EvaluationResult(False, "web result looks like generic architecture/tutorial content", strategy.fallback_tool)

            return EvaluationResult(True, "web observation looks usable", None)

        # -----------------------
        # memory_only
        # -----------------------
        if tool == "memory_only":
            ctx = observation.get("context", "") or ""
            if not ctx.strip():
                return EvaluationResult(False, "memory context empty for followup", strategy.fallback_tool)
            ql = (user_prompt or "").lower()
            if any(k in ql for k in ["還有", "其他", "更多", "補充", "再列"]) and any(k in ql for k in ["研究", "成果", "論文", "實驗室"]):
                signal_count = _count_matches(ctx, ["abstract", "doi", "conference", "journal", "paper", "研究", "成果", "論文", "實驗", "方法", "系統"])
                if signal_count < 4:
                    return EvaluationResult(False, "memory_only lacks enough evidence for knowledge expansion followup", strategy.fallback_tool)
            return EvaluationResult(True, "memory context usable", None)

        # -----------------------
        # direct_* 類
        # -----------------------
        if tool in {"direct_plan", "direct_explain", "direct_answer", "ask_followup"}:
            return EvaluationResult(True, "direct mode usable", None)

        return EvaluationResult(False, f"unknown observation tool: {tool}", strategy.fallback_tool)
