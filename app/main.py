from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logger import setup_logging

setup_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="手相双 Agent + 高级 RAG 示例工程",
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def index() -> dict:
    return {
        "message": "手相双Agent服务运行中",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }
