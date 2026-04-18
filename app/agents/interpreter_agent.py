import re

from app.core.schemas import PalmFeatureProfile, TracePayload
from app.rag.hyde import HydeGenerator
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import Reranker
from app.services.llm_client import LLMClient
from app.services.prompt_service import PromptService


class InterpreterAgent:
    def __init__(self, llm: LLMClient, prompts: PromptService) -> None:
        self.llm = llm
        self.prompts = prompts
        self.rewriter = QueryRewriter(llm, prompts)
        self.hyde = HydeGenerator(llm, prompts)
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

    @staticmethod
    def _profile_text(profile: PalmFeatureProfile) -> str:
        return (
            f"指缝：{profile.finger_gap}；"
            f"指纹：{profile.fingerprint_pattern}；"
            f"生命线：{profile.life_line}；"
            f"智慧线：{profile.head_line}；"
            f"感情线：{profile.heart_line}；"
            f"事业线：{profile.career_line}；"
            f"太阳线：{profile.sun_line}；"
            f"婚姻线：{profile.marriage_line}。"
            f"补充：{'；'.join(profile.notes)}"
        )

    @staticmethod
    def _base_info_text(profile: PalmFeatureProfile) -> str:
        items = [
            f"指缝特征：{profile.finger_gap}",
            f"指纹情况：{profile.fingerprint_pattern}",
            f"生命线：{profile.life_line}",
            f"智慧线：{profile.head_line}",
            f"感情线：{profile.heart_line}",
            f"事业线：{profile.career_line}",
        ]
        if profile.sun_line != "未知":
            items.append(f"太阳线：{profile.sun_line}")
        if profile.marriage_line != "未知":
            items.append(f"婚姻线：{profile.marriage_line}")
        if profile.notes:
            items.append(f"补充观察：{'；'.join(profile.notes[:3])}")
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _history_text(history: list[dict], max_turns: int = 6) -> str:
        if not history:
            return "无"
        lines: list[str] = []
        for item in history[-max_turns:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _build_context(chunks: list) -> str:
        contexts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            contexts.append(f"[{idx}|{chunk.source}|{chunk.score:.3f}] {chunk.text}")
        return "\n\n".join(contexts)

    @staticmethod
    def _sanitize_output(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(修订后的最终答案|正式回答|最终答案)\s*", "", cleaned)
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    @staticmethod
    def _extract_answer_block(text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        reasoning_match = re.search(r"\[REASONING\]([\s\S]*?)(\[ANSWER\]|$)", text)
        answer_match = re.search(r"\[ANSWER\]([\s\S]*)$", text)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else text.strip()
        return reasoning, answer

    @staticmethod
    def _fallback_answer(query: str, profile_text: str, context: str) -> str:
        head = "以下为基于规则库的初步解读，仅供娱乐与参考，不构成现实决策依据。"
        return (
            f"{head}\n\n"
            f"你的问题：{query}\n"
            f"手相特征摘要：{profile_text}\n"
            f"关联知识：\n{context[:900]}"
        )

    def generate_initial_report(self, profile: PalmFeatureProfile, initial_query: str | None = None) -> tuple[str, str]:
        query = (
            initial_query.strip()
            if initial_query and initial_query.strip()
            else "请生成首轮解盘，覆盖性格、财运、事业、感情、健康五个维度。"
        )
        profile_text = self._profile_text(profile)
        base_info = self._base_info_text(profile)
        chunks = self.retriever.retrieve(query, profile_text)
        reranked = self.reranker.rerank(query, chunks)
        context = self._build_context(reranked)

        system_prompt = self.prompts.load("agent2_initial_prompt.txt")
        user_prompt = self.prompts.render(
            "用户手相特征画像：\n{{profile}}\n\n检索上下文：\n{{context}}\n\n用户问题：{{query}}",
            {
                "profile": profile_text,
                "context": context,
                "query": query,
            },
        )
        draft = self.llm.chat(task="summary", system_prompt=system_prompt, user_prompt=user_prompt)
        if draft:
            return base_info, self._sanitize_output(draft)
        return base_info, self._fallback_answer(query, profile_text, context)

    def answer_followup(
        self,
        query: str,
        profile: PalmFeatureProfile,
        base_info: str,
        report: str,
        history: list[dict],
    ) -> tuple[str, TracePayload]:
        profile_text = self._profile_text(profile)
        history_text = self._history_text(history)

        rewritten = self.rewriter.rewrite(query, profile_text, history_text)
        hyde_query = self.hyde.generate(rewritten, profile_text)

        chunks = self.retriever.retrieve(
            query=f"{rewritten}\n{hyde_query}",
            profile_text=profile_text,
        )
        reranked = self.reranker.rerank(rewritten, chunks)
        context = self._build_context(reranked)

        system_prompt = self.prompts.load("followup_reasoning_prompt.txt")
        user_prompt = self.prompts.render(
            "手相基础信息：\n{{base_info}}\n\n综合信息：\n{{report}}\n\n用户手相特征画像：\n{{profile}}\n\n检索上下文：\n{{context}}\n\n对话历史：\n{{history}}\n\n用户问题：{{query}}",
            {
                "base_info": base_info,
                "report": report,
                "profile": profile_text,
                "context": context,
                "history": history_text,
                "query": query,
            },
        )
        reasoning_draft = self.llm.chat(task="text", system_prompt=system_prompt, user_prompt=user_prompt)
        reasoning_text, answer_draft = self._extract_answer_block(reasoning_draft)
        if not answer_draft:
            answer_draft = self._fallback_answer(query, profile_text, context)

        reflect_system = self.prompts.load("logic_validation_prompt.txt")
        reflect_user = self.prompts.render(
            "用户问题：{{query}}\n\n手相基础信息：{{base_info}}\n\n综合信息：{{report}}\n\n手相特征：{{profile}}\n\n检索上下文：{{context}}\n\n推理链：{{reasoning}}\n\n草稿答案：{{draft}}",
            {
                "query": query,
                "base_info": base_info,
                "report": report,
                "profile": profile_text,
                "context": context,
                "reasoning": reasoning_text,
                "draft": answer_draft,
            },
        )
        revised = self.llm.chat(
            task="reflect",
            system_prompt=reflect_system,
            user_prompt=reflect_user,
            temperature=0.1,
        )

        final_answer = self._sanitize_output(revised if revised else answer_draft)
        trace = TracePayload(
            rewritten_query=rewritten,
            hyde_query=hyde_query,
            retrieved_contexts=[chunk.text for chunk in reranked] + ([f"REASONING: {reasoning_text}"] if reasoning_text else []),
        )
        return final_answer, trace
