from app.services.llm_client import LLMClient
from app.services.prompt_service import PromptService


class QueryRewriter:
    def __init__(self, llm: LLMClient, prompts: PromptService) -> None:
        self.llm = llm
        self.prompts = prompts

    def rewrite(self, query: str, profile_text: str, history_text: str) -> str:
        system_prompt = self.prompts.load("query_rewrite_prompt.txt")
        user_prompt = self.prompts.render(
            "原始问题：{{query}}\n\n手相特征：{{profile}}\n\n对话历史：{{history}}",
            {
                "query": query,
                "profile": profile_text,
                "history": history_text,
            },
        )
        rewritten = self.llm.chat(
            task="rewrite",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        if rewritten:
            return rewritten
        return f"结合手相特征 {profile_text} ，回答问题：{query}"
