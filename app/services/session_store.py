import json
import uuid
from pathlib import Path

from app.core.config import ROOT_DIR, settings
from app.core.schemas import SessionRecord


class SessionStore:
    def __init__(self) -> None:
        self.path = ROOT_DIR / settings.session.storage_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict[str, dict]:
        content = self.path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        return json.loads(content)

    def _save(self, payload: dict[str, dict]) -> None:
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def get(self, session_id: str) -> SessionRecord | None:
        payload = self._load()
        record = payload.get(session_id)
        if not record:
            return None
        return SessionRecord(**record)

    def save(self, record: SessionRecord) -> None:
        payload = self._load()
        payload[record.session_id] = record.model_dump()
        self._save(payload)

    def list_records(self) -> list[SessionRecord]:
        payload = self._load()
        records: list[SessionRecord] = []
        for session_id, raw in payload.items():
            try:
                records.append(SessionRecord(**raw))
            except Exception:
                records.append(SessionRecord(session_id=session_id))
        records.sort(key=lambda x: len(x.history), reverse=True)
        return records

    def delete(self, session_id: str) -> bool:
        payload = self._load()
        if session_id not in payload:
            return False
        payload.pop(session_id, None)
        self._save(payload)
        return True

    def get_or_create(self, session_id: str | None = None) -> SessionRecord:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing
        new_id = session_id or self.new_session_id()
        record = SessionRecord(session_id=new_id)
        self.save(record)
        return record
