import logging

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        llm_cfg = settings.llm
        self.default_temperature = llm_cfg.temperature
        self.models = {
            "text": llm_cfg.text_model,
            "vision": llm_cfg.vision_model,
            "summary": llm_cfg.summary_model or llm_cfg.text_model,
            "rewrite": llm_cfg.rewrite_model or llm_cfg.text_model,
            "reflect": llm_cfg.reflect_model or llm_cfg.text_model,
        }
        self.enabled = bool(llm_cfg.api_key and llm_cfg.text_model)
        self.client: OpenAI | None = None

        if self.enabled:
            kwargs = {"api_key": llm_cfg.api_key}
            if llm_cfg.base_url:
                kwargs["base_url"] = llm_cfg.base_url
            self.client = OpenAI(**kwargs)

    @staticmethod
    def _normalize(content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content).strip()

    def chat(
        self,
        *,
        task: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        model = self.models.get(task) or self.models["text"]
        if not self.client or not model:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=model,
                temperature=temperature if temperature is not None else self.default_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return self._normalize(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM chat 调用失败: %s", exc)
            return ""

    def vision(
        self,
        *,
        prompt: str,
        image_data_url: str,
    ) -> str:
        model = self.models.get("vision")
        if not self.client or not model:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    }
                ],
            )
            return self._normalize(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision 调用失败: %s", exc)
            return ""
