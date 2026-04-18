import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None


class VisionSettings(BaseModel):
    blur_threshold: float = 95.0
    min_width: int = 320
    min_height: int = 320
    non_palm_threshold: float = 0.30
    blurry_threshold: float = 0.70


class LLMSettings(BaseModel):
    base_url: str = ""
    api_key: str = ""
    text_model: str = ""
    vision_model: str = ""
    summary_model: str = ""
    rewrite_model: str = ""
    reflect_model: str = ""
    temperature: float = 0.3


class RAGSettings(BaseModel):
    hybrid_top_k: int = 20
    rerank_top_k: int = 5
    chunk_size: int = 350
    chunk_overlap: int = 50
    dense_weight: float = 0.45
    bm25_weight: float = 0.35
    graph_weight: float = 0.20


class SessionSettings(BaseModel):
    storage_file: str = "storage/sessions.json"


class Settings(BaseModel):
    app_name: str = "手相双Agent系统"
    app_version: str = "0.1.0"
    vision: VisionSettings = Field(default_factory=VisionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"
ENV_PATH = ROOT_DIR / ".env"


if load_dotenv is not None and ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)


def _load_yaml() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    content = CONFIG_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(content) or {}


def _apply_env_overrides(raw: dict) -> dict:
    llm = raw.setdefault("llm", {})
    llm["api_key"] = os.getenv("LLM_API_KEY", llm.get("api_key", ""))
    llm["base_url"] = os.getenv("LLM_BASE_URL", llm.get("base_url", ""))
    llm["text_model"] = os.getenv("LLM_TEXT_MODEL", llm.get("text_model", ""))
    llm["vision_model"] = os.getenv("LLM_VISION_MODEL", llm.get("vision_model", ""))
    llm["summary_model"] = os.getenv("LLM_SUMMARY_MODEL", llm.get("summary_model", ""))
    llm["rewrite_model"] = os.getenv("LLM_REWRITE_MODEL", llm.get("rewrite_model", ""))
    llm["reflect_model"] = os.getenv("LLM_REFLECT_MODEL", llm.get("reflect_model", ""))
    return raw


def _load_settings() -> Settings:
    raw = _load_yaml()
    raw = _apply_env_overrides(raw)
    return Settings(**raw)


settings = _load_settings()
