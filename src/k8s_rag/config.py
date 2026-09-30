"""Config loading. Every run is defined by a single YAML file so an eval
result can always be traced back to the exact settings that produced it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunConfig:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RunConfig":
        import os
        path = path or os.environ.get("K8S_RAG_CONFIG") or "config/default.yaml"
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        with open(p) as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw, path=p)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def load_prompt(self) -> dict[str, str]:
        version = self.get("generation", "prompt_version", default="v1")
        prompt_path = REPO_ROOT / "config" / "prompts" / f"{version}.yaml"
        with open(prompt_path) as f:
            return yaml.safe_load(f)
