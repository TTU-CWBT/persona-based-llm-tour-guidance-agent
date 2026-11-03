import os
from typing import Optional, List, Dict, Any
from types import SimpleNamespace
import requests
from .backend_base import LLMBackend, GenResult

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# 可用環境變數調整 timeout，預設連線 10 秒、讀取 600 秒
CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "600"))

class DummyTokenizer:
    """給記憶系統估算 token 用的簡易 tokenizer."""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, return_tensors=None, add_special_tokens: bool = False):
        # 保守以「字元/4」近似，至少長度為 1
        length = max(1, len(text) // 4 + 1)
        # 回傳一個有 .input_ids 的物件，符合 _simple_tok_count 的假設
        return SimpleNamespace(input_ids=[0] * length)

    def decode(self, *args, **kwargs) -> str:
        return ""

class OllamaBackend(LLMBackend):
    """透過 Ollama /api/chat 的後端實作。"""

    def __init__(self, model_name: str, opts: Optional[Dict[str, Any]] = None):
        # 這裡沿用原本的命名 self.model（外部若有使用不會壞）
        self.model = model_name
        self.opts = opts or {}
        self._tok = DummyTokenizer()

    @property
    def model_name(self) -> str:
        """提供一個統一的 model 名稱屬性給外部使用（例如記憶 key）。"""
        return self.model

    def warmup(self) -> None:
        """
        確認：
        1) Ollama daemon 有在跑
        2) 指定的 model 存在（名稱打錯會直接報錯）
        """
        # 1) daemon 存活？
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=(3, 5))
        r.raise_for_status()

        # 2) 檢查 model 是否在 tags 裡
        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Ollama /api/tags 回傳不是合法 JSON：{e}")

        models = data.get("models") or []
        available = [m.get("name") for m in models if isinstance(m, dict)]

        if self.model not in available:
            hint = ", ".join(available) if available else "(目前沒有任何模型)"
            raise RuntimeError(
                f"Ollama 模型 '{self.model}' 不存在。\n"
                f"請確認名稱是否正確（可用 `ollama list` 查），目前可用模型：{hint}"
            )

    @property
    def tokenizer_like(self):
        return self._tok

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> GenResult:
        """
        呼叫 Ollama /api/chat，使用 chat 格式的 system + user 訊息。
        回傳 GenResult(text=...)。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        options: Dict[str, Any] = {
            "num_ctx": self.opts.get("num_ctx", 8192),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "num_predict": max_new_tokens,
        }
        if stop:
            options["stop"] = stop

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        r.raise_for_status()

        data = r.json()
        message = data.get("message")
        if not isinstance(message, dict):
            # 不符合預期格式時，直接丟錯，不給空字串假裝成功
            raise RuntimeError(f"Ollama 回傳內容格式不符：{data}")

        txt = str(message.get("content", "")).strip()
        return GenResult(txt)
