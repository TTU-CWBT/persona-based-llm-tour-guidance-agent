from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class AgentTrace:
    intent: Optional[Dict[str, Any]] = None
    strategy: Optional[Dict[str, Any]] = None
    continuity: Optional[Dict[str, Any]] = None
    primary_observation: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    fallback_observation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)
