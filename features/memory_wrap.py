# features/memory_wrap.py
import os
import re
import json
from dataclasses import asdict, is_dataclass
from typing import Optional

from memory_repository import ConversationMemory

# 統一放記憶檔的資料夾
MEMORY_DIR = "./memory"
DEFAULT_STORE_PATH = "./memory_store.jsonl"

# 這幾個 hyperparameters 之後如果要微調，可以在這裡集中改
MAX_SHORT_ROUNDS = 6
SUMMARY_EVERY_M_ROUNDS = 8
MAX_MEMORY_TOKENS = 512
RECENCY_HALF_LIFE_ROUNDS = 12


def _ensure_memory_dir() -> None:
    """確保記憶檔資料夾存在。"""
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _safe_name(name: str) -> str:
    """
    把模型名稱、backend 名稱裡不適合當檔名的字元換成底線。

    例如：
      "llama-3-taiwan-70b-instruct" -> "llama-3-taiwan-70b-instruct"
      "yentinglin/Llama3"          -> "yentinglin_Llama3"
    """
    return re.sub(r"[^\w\-]+", "_", name)


def _resolve_store_path(
    backend: Optional[str],
    model_name: Optional[str],
    profile: str = "default",
    fallback: str = DEFAULT_STORE_PATH,
) -> str:
    """
    依 backend + model_name + profile 決定記憶檔路徑。

    - 有 backend & model_name：
        ./memory/{backend}__{model}__{profile}.jsonl
    - 否則：
        退回既有的 fallback（共用一個檔）
    """
    if not backend or not model_name:
        return fallback

    _ensure_memory_dir()
    b = _safe_name(backend)
    m = _safe_name(model_name)
    p = _safe_name(profile or "default")
    return os.path.join(MEMORY_DIR, f"{b}__{m}__{p}.jsonl")


def ensure_memory(
    tokenizer,
    summarizer_fn,
    store_path: Optional[str] = None,
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    profile: str = "default",
) -> ConversationMemory:
    """
    建立一個 ConversationMemory 實例。

    使用邏輯：
    - 呼叫端有給 store_path：直接使用該路徑。
    - 沒有給 store_path，但有 backend / model_name：
        依 backend + model_name + profile 決定檔名。
    - 什麼都沒給：
        退回原本的 "./memory_store.jsonl"（所有情境共用一個檔）。
    """
    if store_path is None:
        store_path = _resolve_store_path(
            backend=backend,
            model_name=model_name,
            profile=profile,
            fallback=DEFAULT_STORE_PATH,
        )

    memory = ConversationMemory(
        tokenizer=tokenizer,
        store_path=store_path,
        max_short_rounds=MAX_SHORT_ROUNDS,
        summary_every_m_rounds=SUMMARY_EVERY_M_ROUNDS,
        max_memory_tokens=MAX_MEMORY_TOKENS,
        recency_half_life_rounds=RECENCY_HALF_LIFE_ROUNDS,
        summarizer=summarizer_fn,
    )
    return hydrate_memory_threads(memory)



def _thread_state_path(store_path: str) -> str:
    return f"{store_path}.threads.json"


def _serialize_thread(thread):
    if thread is None:
        return None
    if is_dataclass(thread):
        return asdict(thread)
    if isinstance(thread, dict):
        return thread
    data = {}
    for key in (
        "task_id", "task_type", "task_subject", "source_of_truth",
        "status", "turn_mode", "last_query", "last_successful_tool",
        "last_successful_observation_summary", "confidence", "metadata",
    ):
        if hasattr(thread, key):
            data[key] = getattr(thread, key)
    return data or None


def hydrate_memory_threads(memory: ConversationMemory) -> ConversationMemory:
    store_path = getattr(memory, "store_path", None)
    if not store_path:
        return memory

    path = _thread_state_path(store_path)
    if not os.path.exists(path):
        return memory

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        setattr(memory, "current_task_thread", payload.get("current_task_thread"))
        setattr(memory, "task_thread_history", payload.get("task_thread_history") or [])
    except Exception:
        setattr(memory, "current_task_thread", None)
        setattr(memory, "task_thread_history", [])
    return memory


def persist_memory_threads(memory: ConversationMemory) -> None:
    store_path = getattr(memory, "store_path", None)
    if not store_path:
        return

    path = _thread_state_path(store_path)
    current_thread = _serialize_thread(getattr(memory, "current_task_thread", None))
    history = getattr(memory, "task_thread_history", None) or []
    serialized_history = []
    for item in history:
        obj = _serialize_thread(item)
        if obj:
            serialized_history.append(obj)

    payload = {
        "current_task_thread": current_thread,
        "task_thread_history": serialized_history,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
