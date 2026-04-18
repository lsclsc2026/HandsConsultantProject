from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GateCategory(str, Enum):
    NON_PALM = "non_palm"
    BLURRY = "blurry"
    PALM = "palm"


class GateResult(BaseModel):
    category: GateCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class PalmFeatureProfile(BaseModel):
    finger_gap: str = "未知"
    fingerprint_pattern: str = "未知"
    life_line: str = "未知"
    head_line: str = "未知"
    heart_line: str = "未知"
    career_line: str = "未知"
    sun_line: str = "未知"
    marriage_line: str = "未知"
    notes: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    session_id: str
    gate: GateResult
    profile: PalmFeatureProfile | None = None
    base_info: str | None = None
    report: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    query: str


class TracePayload(BaseModel):
    rewritten_query: str
    hyde_query: str
    retrieved_contexts: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: TracePayload | None = None


class SessionRecord(BaseModel):
    session_id: str
    profile: PalmFeatureProfile | None = None
    base_info: str | None = None
    report: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    message_count: int = 0
    preview: str = "新会话"


class DeleteSessionResponse(BaseModel):
    deleted: bool
    session_id: str


class RebuildKnowledgeResponse(BaseModel):
    ok: bool
    rule_docs: int
    qa_docs: int
    graph_nodes: int
    graph_edges: int
