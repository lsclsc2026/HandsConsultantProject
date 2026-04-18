from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.schemas import (
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    DeleteSessionResponse,
    RebuildKnowledgeResponse,
    SessionRecord,
    SessionSummary,
)
from app.services.palm_service import PalmService

router = APIRouter()
service = PalmService()


@router.post("/palm/analyze", response_model=AnalyzeResponse)
async def analyze_palm(
    image: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    initial_query: str | None = Form(default=None),
) -> AnalyzeResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传图片为空")
    return service.analyze(
        image_bytes=image_bytes,
        session_id=session_id,
        initial_query=initial_query,
    )


@router.post("/palm/chat", response_model=ChatResponse)
def chat_with_master(payload: ChatRequest) -> ChatResponse:
    try:
        return service.chat(session_id=payload.session_id, query=payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return service.list_sessions()


@router.post("/sessions", response_model=SessionSummary)
def create_session() -> SessionSummary:
    return service.create_session()


@router.get("/sessions/{session_id}", response_model=SessionRecord)
def get_session(session_id: str) -> SessionRecord:
    try:
        return service.get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str) -> DeleteSessionResponse:
    deleted = service.delete_session(session_id)
    return DeleteSessionResponse(deleted=deleted, session_id=session_id)


@router.post("/knowledge/rebuild", response_model=RebuildKnowledgeResponse)
def rebuild_knowledge() -> RebuildKnowledgeResponse:
    return service.rebuild_knowledge()
