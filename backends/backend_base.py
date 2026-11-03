from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GenResult:
    """LLM 單輪生成結果的最小封裝。"""
    text: str


class LLMBackend(ABC):
    """所有 LLM 後端的共同介面。"""

    @abstractmethod
    def warmup(self) -> None:
        """初始化／檢查連線與模型可用性。"""
        ...

    @abstractmethod
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
        """執行一次生成，回傳 GenResult。"""
        ...

    @property
    def tokenizer_like(self):
        """可選：給記憶模組一個簡單 token 計數器用。"""
        return None
