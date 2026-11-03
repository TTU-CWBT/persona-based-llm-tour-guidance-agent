# memory_repository.py
from __future__ import annotations
import os, json, time, re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Callable

# --------- 小工具 ---------
def _now_ts() -> float:
    return time.time()

def _simple_tok_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)

def _normalize(text: str) -> str:
    # 粗略正規化用於關鍵詞比對
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)  # 英數 & 中文保留
    return text

def _keyword_set(text: str) -> set:
    toks = _normalize(text).split()
    # 去除太短的 token
    return set([t for t in toks if len(t) >= 2])

# --------- 資料結構 ---------
@dataclass
class Turn:
    role: str            # "user" 或 "assistant"
    text: str
    ts: float

@dataclass
class MemoryItem:
    kind: str            # "turn" | "summary" | "task" | "open_loop"
    text: str
    ts: float
    round_id: int        # 第幾輪（user->assistant 完成算一輪）
    score_meta: Optional[dict] = None

class ConversationMemory:
    """
    輕量記憶系統：
    - 短期記憶：保存最近 N 輪完整對話（user+assistant）
    - 長期記憶：每 M 輪把這 M 輪摘要為一條 summary
    - 檢索：關鍵詞重疊 + 近期性衰減
    - Token 預算：記憶文字總長不超過 max_memory_tokens
    """
    def __init__(
        self,
        tokenizer,
        store_path: str = "./memory_store.jsonl",
        max_short_rounds: int = 6,        # 短期：保留最近 6 輪
        summary_every_m_rounds: int = 8,  # 長期：每 8 輪生成摘要
        max_memory_tokens: int = 512,     # 插入到提示的記憶 token 上限
        recency_half_life_rounds: int = 12,  # 近期性衰減
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        self.tokenizer = tokenizer
        self.store_path = store_path
        base, ext = os.path.splitext(store_path)
        self.task_store_path = base + "__task" + (ext or ".jsonl")
        self.max_short_rounds = max_short_rounds
        self.summary_every_m_rounds = summary_every_m_rounds
        self.max_memory_tokens = max_memory_tokens
        self.recency_half_life_rounds = recency_half_life_rounds
        self.summarizer = summarizer  # 可用 LLM 做摘要的 callback

        self._turn_buffer: List[Turn] = []     # 暫存當前輪（先 user，再 assistant）
        self._round_count: int = 0             # 完成的輪數
        self._mem: List[MemoryItem] = []       # 全部記憶項（turn or summary）

        self._load()

    # ---------- 永久化 ----------
    def _load(self):
        # 主記憶檔：turn / summary
        if os.path.exists(self.store_path):
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    kind = obj.get("kind")
                    if kind in ("turn", "summary"):
                        self._mem.append(MemoryItem(**obj))

        # 任務檔：task / open_loop
        if os.path.exists(self.task_store_path):
            with open(self.task_store_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    kind = obj.get("kind")
                    if kind in ("task", "open_loop"):
                        self._mem.append(MemoryItem(**obj))

        turn_like_round_ids = [
            mi.round_id
            for mi in self._mem
            if mi.kind in ("turn", "summary")
        ]

        if turn_like_round_ids:
            self._round_count = max(turn_like_round_ids)
        else:
            self._round_count = 0

    def _append_and_flush(self, item: MemoryItem):
        self._mem.append(item)

        if item.kind in ("task", "open_loop"):
            target_path = self.task_store_path
        else:
            target_path = self.store_path

        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    # ---------- 對外 API ----------
    def add_user(self, text: str):
        self._turn_buffer.append(Turn(role="user", text=text, ts=_now_ts()))

    def add_assistant(self, text: str):
        self._turn_buffer.append(Turn(role="assistant", text=text, ts=_now_ts()))
        # 一輪完成（user + assistant），寫入短期記憶
        self._round_count += 1
        rid = self._round_count
        # 收攏該輪兩個 turn（若 assistant 先來/異常，做降級處理）
        user_text = ""
        assistant_text = ""
        for t in self._turn_buffer[-2:]:
            if t.role == "user": user_text = t.text
            elif t.role == "assistant": assistant_text = t.text
        packed = f"[User]\n{user_text}\n\n[Assistant]\n{assistant_text}"
        self._append_and_flush(MemoryItem(
            kind="turn", text=packed, ts=_now_ts(), round_id=rid
        ))
        # 總結/修剪
        self._maybe_summarize()
        self._trim_short_term()

    def add_task(self, title: str, detail: str = "", status: str = "active"):
        """
        記錄一個任務狀態，例如：
        - 正在進行的系統改造
        - 某個尚未完成的規劃
        """
        rid = max(1, self._round_count)
        text = f"[Task][{status}] {title}"
        if detail:
            text += f"\n{detail}"

        self._append_and_flush(MemoryItem(
            kind="task",
            text=text,
            ts=_now_ts(),
            round_id=rid,
            score_meta={"status": status},
        ))
    
    def add_open_loop(self, text: str):
        """
        記錄尚未完成、未來可能要延續的事項。
        例如：
        - 之後要補第 2 階段 planner
        - 還要改 memory task 記憶
        """
        rid = max(1, self._round_count)
        self._append_and_flush(MemoryItem(
            kind="open_loop",
            text=text,
            ts=_now_ts(),
            round_id=rid,
            score_meta=None,
        ))

    def get_open_loops(self) -> List[str]:
        """
        取最近的未完成事項，讓 agent state / planner 可以讀。
        """
        items = [mi for mi in self._mem if mi.kind == "open_loop"]
        items.sort(key=lambda x: x.ts, reverse=True)
        return [mi.text for mi in items[:5]]
    
    def get_tasks(self, limit: int = 5) -> List[str]:
        """
        取最近的任務記錄。
        """
        items = [mi for mi in self._mem if mi.kind == "task"]
        items.sort(key=lambda x: x.ts, reverse=True)
        return [mi.text for mi in items[:limit]]

    # ---------- 產生可注入的記憶上下文 ----------
    def build_memory_context(self, query: str) -> str:
        """
        回傳一段可放進 system 的記憶文字，長度受 max_memory_tokens 限制
        內容包含：
        - 最近 K 輪原文（短期）
        - 依關聯度與近期性挑出的長期摘要/舊回合
        """
        # 1) 抽最近 K 輪（turns）
        recent_turns = self._get_recent_turns(self.max_short_rounds)

        # 2) 基於 query 做關聯檢索（long-term summaries + 更舊的 turns）
        candidates = [mi for mi in self._mem if mi.kind in ("summary", "turn")]
        ranked = self._rank_by_keyword_and_recency(query, candidates)

        
        # 3) 併合去重，按 token 預算打包
        parts: List[str] = []
        used_ids = set(id(mi) for mi in recent_turns)
        def try_add(label: str, text: str) -> bool:
            next_piece = f"\n\n{label}\n{text}"
            if _simple_tok_count(self.tokenizer, "".join(parts) + next_piece) <= self.max_memory_tokens:
                parts.append(next_piece)
                return True
            return False
        #先放任務 Markdown
        task_md = self._build_task_markdown()
        if task_md:
            try_add("", task_md)

        # 再放最近 K 輪
        if recent_turns:
            chunk = "\n\n--- 最近對話（短期記憶） ---\n" + "\n\n--- ---\n".join(mi.text for mi in recent_turns)
            try_add("", chunk)

        # 再放關聯條目（跳過已放的）
        for mi in ranked:
            if id(mi) in used_ids:
                continue

            if mi.kind == "summary":
                label = "【長期摘要】"
            elif mi.kind == "turn":
                label = "【過往對話】"
            else:
                continue

            ok = try_add(label, mi.text)
            if not ok:
                break

        if not parts:
            return ""
        header = (
            "以下是系統整理的記憶內容。"
            "請在回覆時善用其中資訊，但若與使用者新要求衝突，請以新要求為準。"
        )
        return header + "".join(parts)

    # ---------- 內部：近期/檢索/修剪/摘要 ----------
    def _get_recent_turns(self, k_rounds: int) -> List[MemoryItem]:
        turns = [mi for mi in self._mem if mi.kind == "turn"]
        turns.sort(key=lambda x: x.round_id, reverse=True)
        return list(reversed(turns[:k_rounds]))  # 由舊到新

    def _rank_by_keyword_and_recency(self, query: str, items: List[MemoryItem]) -> List[MemoryItem]:
        qset = _keyword_set(query)
        if not qset:
            # 純近期性排序
            return sorted(items, key=lambda mi: self._recency_weight(mi.round_id), reverse=True)

        scored: List[Tuple[float, MemoryItem]] = []
        for mi in items:
            tset = _keyword_set(mi.text)
            overlap = len(qset & tset)
            rec = self._recency_weight(mi.round_id)
            score = overlap * 1.0 + rec * 0.7
            scored.append((score, mi))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mi for _, mi in scored]

    def _recency_weight(self, round_id: int) -> float:
        if self._round_count <= 0:
            return 0.0
        dist = max(0, self._round_count - round_id)
        # 半衰期：每半衰期分數減半
        return 0.5 ** (dist / max(1, self.recency_half_life_rounds))

    def _trim_short_term(self):
        # 將短期以外的 turn 保留，但短期集合由 build_memory_context 決定
        # 這裡僅做「檔案層級」不修剪，以免丟資料；真正短期只是一種選取方式
        pass

    def _maybe_summarize(self):
        if self._round_count % self.summary_every_m_rounds != 0:
            return
        # 抽出最近 m 輪，做摘要寫入 summary
        m = self.summary_every_m_rounds
        start = max(1, self._round_count - m + 1)
        chunk_turns = [mi for mi in self._mem if (mi.kind == "turn" and start <= mi.round_id <= self._round_count)]
        if not chunk_turns:
            return
        whole = "\n\n".join(mi.text for mi in chunk_turns)
        summary = self._summarize_text(whole)
        self._append_and_flush(MemoryItem(
            kind="summary", text=summary, ts=_now_ts(), round_id=self._round_count
        ))

    def _summarize_text(self, text: str) -> str:
        # 若提供 LLM 摘要器就用；否則啟發式
        if self.summarizer is not None:
            try:
                s = self.summarizer(text)
                if s and s.strip():
                    return s.strip()
            except Exception:
                pass
        # 簡單啟發式：抓關鍵句（含「喜歡/想要/決定/下次/提醒/重點/結論」等詞）+ 前幾句
        lines = [l.strip() for l in re.split(r"[。\.\n]+", text) if l.strip()]
        keys = [l for l in lines if re.search(r"(喜歡|想要|決定|下次|提醒|重點|結論|規劃|偏好|需求)", l)]
        head = lines[:3]
        picks = (keys + head)[:10]
        if not picks:
            picks = lines[:5]
        return "長期摘要：\n- " + "\n- ".join(picks)
    
    def _build_task_markdown(self) -> str:
        tasks = self.get_tasks(limit=5)
        open_loops = self.get_open_loops()

        sections = []

        if tasks:
            lines = ["## 當前任務"]
            for t in tasks:
                lines.append(f"- {t}")
            sections.append("\n".join(lines))

        if open_loops:
            lines = ["## 尚未實作任務"]
            for item in open_loops:
                lines.append(f"- {item}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections).strip()