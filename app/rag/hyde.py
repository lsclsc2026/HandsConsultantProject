from app.services.llm_client import LLMClient
from app.services.prompt_service import PromptService


class HydeGenerator:
    def __init__(self, llm: LLMClient, prompts: PromptService) -> None:
        self.llm = llm
        self.prompts = prompts

    def generate(self, rewritten_query: str, profile_text: str) -> str:
        system_prompt = self.prompts.load("hyde_prompt.txt")
        user_prompt = self.prompts.render(
            "改写后的检索问题：{{query}}\n\n手相特征：{{profile}}",
            {
                "query": rewritten_query,
                "profile": profile_text,
            },
        )
        pseudo = self.llm.chat(
            task="rewrite",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )
        if pseudo:
            return pseudo
        return rewritten_query
