from app.agents.interpreter_agent import InterpreterAgent
from app.agents.vision_agent import VisionAgent
from app.core.schemas import (
    AnalyzeResponse,
    ChatResponse,
    GateCategory,
    GateResult,
    PalmFeatureProfile,
    RebuildKnowledgeResponse,
    SessionRecord,
    SessionSummary,
)
from app.services.llm_client import LLMClient
from app.services.prompt_service import PromptService
from app.services.session_store import SessionStore


class PalmService:
    def __init__(self) -> None:
        self.prompts = PromptService()
        self.llm = LLMClient()
        self.store = SessionStore()
        self.vision_agent = VisionAgent(self.llm, self.prompts)
        self.interpreter = InterpreterAgent(self.llm, self.prompts)

    def analyze(
        self,
        image_bytes: bytes,
        session_id: str | None = None,
        initial_query: str | None = None,
    ) -> AnalyzeResponse:
        record = self.store.get_or_create(session_id)

        initial_user_message = None
        if initial_query and initial_query.strip():
            initial_user_message = f"[首轮问题] {initial_query.strip()}\n[已上传手相图片]"

        # 如果当前会话已经完成过相同首轮分析，则直接返回既有结果，避免重复写入历史。
        if (
            record.profile is not None
            and record.base_info
            and record.report
            and initial_user_message
            and len(record.history) >= 2
            and record.history[-2].get("role") == "user"
            and record.history[-2].get("content") == initial_user_message
            and record.history[-1].get("role") == "assistant"
            and record.history[-1].get("content") == record.report
        ):
            return AnalyzeResponse(
                session_id=record.session_id,
                gate=GateResult(
                    category=GateCategory.PALM,
                    confidence=1.0,
                    reason="已复用当前会话的首轮分析结果。",
                ),
                profile=record.profile,
                base_info=record.base_info,
                report=record.report,
            )

        gate, profile = self.vision_agent.analyze(image_bytes)

        if gate.category != GateCategory.PALM or profile is None:
            record.history.append({
                "role": "system",
                "content": f"图片预检未通过: {gate.reason}",
            })
            self.store.save(record)
            return AnalyzeResponse(session_id=record.session_id, gate=gate)

        base_info, report = self.interpreter.generate_initial_report(profile, initial_query=initial_query)
        record.profile = PalmFeatureProfile(**profile.model_dump())
        record.base_info = base_info
        record.report = report

        if initial_user_message:
            record.history.append(
                {
                    "role": "user",
                    "content": initial_user_message,
                }
            )
        record.history.append(
            {
                "role": "assistant",
                "content": f"【手相的基础信息】\n{base_info}\n\n【综合信息】\n{report}",
            }
        )
        self.store.save(record)

        return AnalyzeResponse(
            session_id=record.session_id,
            gate=gate,
            profile=profile,
            base_info=base_info,
            report=report,
        )

    def chat(self, session_id: str, query: str) -> ChatResponse:
        record = self.store.get(session_id)
        if not record:
            raise ValueError("会话不存在，请先上传手相图。")
        if not record.profile:
            raise ValueError("当前会话没有手相特征，请先执行 analyze 接口。")

        answer, trace = self.interpreter.answer_followup(
            query=query,
            profile=record.profile,
            base_info=record.base_info or "",
            report=record.report or "",
            history=record.history,
        )

        record.history.append({"role": "user", "content": query})
        record.history.append({"role": "assistant", "content": answer})
        self.store.save(SessionRecord(**record.model_dump()))

        return ChatResponse(session_id=session_id, answer=answer, trace=trace)

    def create_session(self) -> SessionSummary:
        record = self.store.get_or_create(None)
        return SessionSummary(
            session_id=record.session_id,
            message_count=0,
            preview="新会话",
        )

    def list_sessions(self) -> list[SessionSummary]:
        records = self.store.list_records()
        items: list[SessionSummary] = []
        for record in records:
            preview = "新会话"
            if record.history:
                preview = str(record.history[-1].get("content", "新会话")).replace("\n", " ")[:48]
            items.append(
                SessionSummary(
                    session_id=record.session_id,
                    message_count=len(record.history),
                    preview=preview,
                )
            )
        return items

    def get_session(self, session_id: str) -> SessionRecord:
        record = self.store.get(session_id)
        if not record:
            raise ValueError("会话不存在")
        return record

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete(session_id)

    def rebuild_knowledge(self) -> RebuildKnowledgeResponse:
        self.interpreter = InterpreterAgent(self.llm, self.prompts)
        retriever = self.interpreter.retriever
        graph = retriever.graph.graph
        rule_docs = sum(1 for s, _ in retriever.docs if s == "rule")
        qa_docs = sum(1 for s, _ in retriever.docs if s == "qa")
        return RebuildKnowledgeResponse(
            ok=True,
            rule_docs=rule_docs,
            qa_docs=qa_docs,
            graph_nodes=graph.number_of_nodes(),
            graph_edges=graph.number_of_edges(),
        )
