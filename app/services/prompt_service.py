from pathlib import Path

from app.core.config import ROOT_DIR


class PromptService:
    def __init__(self) -> None:
        self.base_dir = ROOT_DIR / "prompts"
        self._cache: dict[str, str] = {}

    def load(self, filename: str) -> str:
        if filename in self._cache:
            return self._cache[filename]
        prompt_path = self.base_dir / filename
        content = prompt_path.read_text(encoding="utf-8")
        self._cache[filename] = content
        return content

    @staticmethod
    def render(template: str, values: dict[str, str]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", value)
        return rendered
