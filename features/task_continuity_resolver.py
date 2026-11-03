from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class ContinuityDecision:
    is_continuation: bool
    relation: str                     # new_task / clarify / expand / compare / shift / close
    continued_task_type: Optional[str] = None
    continued_task_subject: Optional[str] = None
    preferred_tool: Optional[str] = None
    preferred_query: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

    # 若延續題本身缺少必要條件，先追問，不要急著進檢索工具。
    # 典型情境：前一輪列出三個以上項目，使用者只問「這兩個差在哪」。
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


class TaskContinuityResolver:
    def resolve(self, user_prompt: str, memory_text: str = "", agent_state=None) -> ContinuityDecision:
        q = (user_prompt or "").strip()
        ql = q.lower()
        thread = getattr(agent_state, 'current_thread', None) if agent_state is not None else None
        if thread is None or getattr(thread, 'status', None) != 'active':
            return ContinuityDecision(False, 'new_task', confidence=0.9, reason='no active task thread')

        overlap = self._score_overlap(ql, thread)
        relation = self._infer_relation(ql)
        short_like = len(q) <= 40
        thread_subject = (getattr(thread, 'task_subject', '') or '').lower()
        source_bonus = 0.1 if getattr(thread, 'source_of_truth', '') in {'lab_kb', 'project_context', 'web'} else 0.0
        score = overlap + (0.15 if short_like else 0.0) + source_bonus + (0.2 if relation != 'new_task' else 0.0)

        if score < 0.35 and relation == 'new_task':
            return ContinuityDecision(False, 'new_task', confidence=0.55, reason='low continuation score')

        task_type = getattr(thread, 'task_type', None)
        preferred_tool = self._infer_preferred_tool(task_type, getattr(thread, 'source_of_truth', None))

        clarification = self._build_underspecified_comparison_clarification(
            user_prompt=q,
            memory_text=memory_text,
            thread=thread,
        )
        if clarification:
            return ContinuityDecision(
                True,
                'clarify',
                continued_task_type=task_type,
                continued_task_subject=getattr(thread, 'task_subject', None),
                preferred_tool='ask_followup',
                preferred_query=q,
                confidence=min(0.95, max(0.75, score)),
                reason='comparison target is underspecified',
                needs_clarification=True,
                clarification_question=clarification,
            )

        preferred_query = self._rewrite_query_for_relation(q, relation, thread_subject, task_type)
        return ContinuityDecision(
            True,
            relation if relation != 'new_task' else 'clarify',
            continued_task_type=task_type,
            continued_task_subject=getattr(thread, 'task_subject', None),
            preferred_tool=preferred_tool,
            preferred_query=preferred_query,
            confidence=min(0.95, max(0.55, score)),
            reason=f'active thread continuation inferred (score={score:.2f}, relation={relation})',
        )

    def _score_overlap(self, ql: str, thread) -> float:
        texts = [
            (getattr(thread, 'task_subject', '') or '').lower(),
            (getattr(thread, 'last_query', '') or '').lower(),
            (getattr(thread, 'last_successful_observation_summary', '') or '').lower(),
        ]
        keywords = set()
        for text in texts:
            for tok in re.findall(r'[a-zA-Z_]{3,}|[一-鿿]{2,}', text):
                keywords.add(tok)
        if not keywords:
            return 0.0
        matched = sum(1 for kw in keywords if kw in ql)
        if matched >= 4:
            return 0.5
        if matched >= 2:
            return 0.35
        if matched >= 1:
            return 0.2
        return 0.0

    def _infer_relation(self, ql: str) -> str:
        if any(k in ql for k in ['比較', '差別', '差異', '差在哪', '差在哪裡', '不同', 'compare', 'versus', 'vs']):
            return 'compare'
        if any(k in ql for k in ['還有', '其他', '更多', '另外', '補充', '再列', '再介紹', '除此之外']):
            return 'expand'
        if any(k in ql for k in ['是什麼意思', '再講', '講清楚', '細講', '怎麼做', '怎麼運作', '為什麼']):
            return 'clarify'
        if any(k in ql for k in ['那', '其中', '那個', '這部分', '那部分']):
            return 'shift'
        return 'new_task'

    def _infer_preferred_tool(self, task_type: str | None, source_of_truth: str | None) -> Optional[str]:
        if source_of_truth == 'lab_kb' or task_type == 'lab_research_lookup':
            return 'retrieve_lab_kb'
        if source_of_truth == 'project_context' or task_type in {'system_explanation', 'design_planning', 'system_or_project_context'}:
            return 'project_context'
        if source_of_truth == 'web' or task_type == 'external_info':
            return 'retrieve_web'
        return None

    def _build_underspecified_comparison_clarification(
        self,
        user_prompt: str,
        memory_text: str,
        thread,
    ) -> Optional[str]:
        """
        偵測「比較任務」是否缺少比較對象。

        這個檢查放在工具檢索前：
        - 若前文列出三個以上候選項目
        - 使用者本輪只說「這兩個／兩者／差在哪」等不完整指涉
        - 且本輪沒有明確點名兩個候選項目
        就先請使用者指定要比較的兩個對象。
        """
        q = (user_prompt or "").strip()
        ql = q.lower()
        if not self._is_comparison_request(ql):
            return None

        candidates = self._extract_recent_candidates(memory_text=memory_text, thread=thread)
        if len(candidates) < 3:
            return None

        explicit_count = self._count_explicit_candidate_mentions(ql, candidates)
        has_ambiguous_pair_ref = self._has_ambiguous_pair_reference(ql)

        # 「差在哪」未必一定會寫出「這兩個」，但當前文候選 >= 3 且本輪未點名兩項時，
        # 直接比較就會要求系統自行猜測，因此也應追問。
        if explicit_count >= 2:
            return None
        if not has_ambiguous_pair_ref and explicit_count > 0:
            return None

        return self._format_candidate_clarification_question(candidates)

    def _is_comparison_request(self, ql: str) -> bool:
        comparison_cues = [
            '比較', '差別', '差異', '差在哪', '差在哪裡', '不同',
            'compare', 'versus', ' vs ', 'vs.',
        ]
        return any(cue in ql for cue in comparison_cues)

    def _has_ambiguous_pair_reference(self, ql: str) -> bool:
        pair_refs = [
            '這兩個', '那兩個', '兩個', '兩者', '這兩者', '那兩者',
            '這兩項', '那兩項', '這兩種', '那兩種', '它們', '他們',
        ]
        return any(ref in ql for ref in pair_refs)

    def _extract_recent_candidates(self, memory_text: str, thread) -> list[str]:
        """
        從最近對話與 task thread 摘要中抽出前一輪列舉過的候選項目。
        這不是用來回答知識內容，只是用來判斷「前文是否有多個可比較對象」。
        """
        texts = [
            memory_text or "",
            getattr(thread, 'last_successful_observation_summary', '') or '',
            getattr(thread, 'task_subject', '') or '',
            getattr(thread, 'last_query', '') or '',
        ]
        combined = "\n".join(t for t in texts if t)

        candidates: list[str] = []
        seen: set[str] = set()

        # 主要抓「1. xxx」「2. xxx」「- xxx」這類前一輪回答中的列舉項。
        enum_pattern = re.compile(
            r'^\s*(?:\d+\s*[\.\、\)]|[•\-])\s*([^\n：:]{2,80}(?:[：:][^\n]{0,80})?)',
            re.MULTILINE,
        )
        for m in enum_pattern.finditer(combined):
            item = self._clean_candidate_label(m.group(1))
            key = self._normalize_candidate_key(item)
            if not item or len(item) < 2 or key in seen:
                continue
            if self._looks_like_non_candidate(item):
                continue
            seen.add(key)
            candidates.append(item)
            if len(candidates) >= 8:
                break

        # 若最近回答不是條列式，補抓常見以頓號/逗號串列的研究項目。
        if len(candidates) < 3:
            known_labels = [
                'Steerable Generative AI-Based Virtual Tutor for Design Learning',
                '生成式 AI 虛擬導師',
                '虛擬導師',
                'Edge AI 與手勢控制冷氣',
                '手勢控制冷氣',
                '應用於導覽任務之基於人格設定大型語言模型代理人架構設計',
                'Design of a Persona-Based LLM Agent Architecture for Tour Guiding Tasks',
                '基於人格設定大型語言模型代理人架構',
                '導覽代理人架構',
                '導覽代理系統',
            ]
            for label in known_labels:
                if label.lower() in combined.lower():
                    key = self._normalize_candidate_key(label)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(label)

        return candidates

    def _clean_candidate_label(self, item: str) -> str:
        s = (item or "").strip()
        s = re.sub(r'\s+', ' ', s)
        # 移除記憶區塊或 markdown 分隔線常見前綴，避免把「--- 最近對話」當成候選項。
        s = re.sub(r'^[-—–]+\s*', '', s).strip()
        # 保留標題，不把後面的長解釋一起當成候選名稱。
        s = re.split(r'[：:。；;，,]\s*', s, maxsplit=1)[0].strip()
        # 移除常見前綴。
        s = re.sub(r'^(第一個|第二個|第三個|第一項|第二項|第三項|首先|再來|最後)\s*', '', s).strip()
        return s[:80].strip()

    def _looks_like_non_candidate(self, item: str) -> bool:
        low = item.lower().strip()
        if not low:
            return True
        # 純分隔線、記憶標題或 debug 標題不是可比較的候選項。
        if re.fullmatch(r'[-—–\s]+', low):
            return True
        bad_fragments = [
            '根據提供', '以下是', '這些研究', '您好', 'user', 'assistant',
            'debug', 'intent', 'strategy', 'primary observation', 'evaluation',
            '最近對話', '短期記憶', '長期記憶', '系統整理的記憶內容',
            'memory context', 'conversation memory', 'primary observation',
            'fallback observation', 'observation evaluation',
        ]
        return any(b in low for b in bad_fragments)

    def _normalize_candidate_key(self, item: str) -> str:
        return re.sub(r'\s+', '', (item or '').lower())

    def _candidate_aliases(self, candidate: str) -> list[str]:
        c = (candidate or "").lower()
        aliases = {candidate.strip(), c}

        if 'virtual tutor' in c or '虛擬導師' in candidate or '生成式' in candidate:
            aliases.update([
                '虛擬導師', '生成式 ai 虛擬導師', '生成式ai虛擬導師',
                'virtual tutor', 'steerable generative',
            ])
        if 'edge' in c or '手勢' in candidate or '冷氣' in candidate or '空調' in candidate:
            aliases.update([
                'edge ai', '手勢控制冷氣', '手勢辨識', '冷氣', '空調',
            ])
        if (
            'persona-based llm agent architecture' in c
            or 'tour guiding tasks' in c
            or '導覽型' in candidate
            or '導覽代理' in candidate
            or '人格設定大型語言模型代理人架構' in candidate
        ):
            aliases.update([
                '應用於導覽任務之基於人格設定大型語言模型代理人架構設計',
                'design of a persona-based llm agent architecture for tour guiding tasks',
                'persona-based llm agent architecture',
                '基於人格設定大型語言模型代理人架構',
                '人格設定大型語言模型代理人架構',
                '導覽代理人架構', '導覽代理系統', '導覽型代理',
            ])

        # 對中文長標題保留一小段可辨識片段，避免完全依賴固定別名。
        compact = self._normalize_candidate_key(candidate)
        if len(compact) >= 4:
            aliases.add(compact[:12])

        return [a for a in aliases if a]

    def _count_explicit_candidate_mentions(self, ql: str, candidates: list[str]) -> int:
        matched = 0
        for candidate in candidates:
            aliases = self._candidate_aliases(candidate)
            if any(self._normalize_candidate_key(alias) in self._normalize_candidate_key(ql) for alias in aliases):
                matched += 1
        return matched

    def _format_candidate_clarification_question(self, candidates: list[str]) -> str:
        shown = candidates[:5]
        lines = ["你想比較的是哪兩個項目？剛剛提到的候選項目包含："]
        for i, item in enumerate(shown, 1):
            lines.append(f"{i}. {item}")
        lines.append("請先告訴我你要比較哪兩個，我再針對那兩項說明差異。")
        return "\n".join(lines)

    def _rewrite_query_for_relation(self, q: str, relation: str, task_subject: str, task_type: str | None) -> str:
        if relation == 'expand' and task_type == 'lab_research_lookup':
            base = task_subject or '目前實驗室研究成果'
            # 關鍵修正：保留使用者這一輪明確補充的新線索，不要在 continuity 重寫時把它洗掉。
            # 例如「手勢辨識」這種新主題詞，若被改寫成通用的「還有其他成果嗎」，
            # 檢索就只會繼續貼著上一篇論文跑，永遠撈不到另一篇候選成果。
            return (
                f'使用者本輪補充線索：{q}。'
                f'請以這個線索為優先，並針對{base}，'
                '找出目前資料庫中其他不同來源的研究成果、論文或系統；'
                '優先補充前面未提到、且來自不同論文或不同檔案的項目。'
            )
        if relation == 'compare' and task_subject:
            return f'針對{task_subject}，並保留使用者這一輪提到的線索（{q}），請比較相關項目。'
        if relation in {'clarify', 'shift'} and task_subject:
            return f'針對{task_subject}，請回答：{q}'
        return q
