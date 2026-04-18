import json
import re

from app.core.schemas import GateCategory, GateResult, PalmFeatureProfile
from app.services.cv_enhance import ImageEnhancer, LocalPalmGate
from app.services.llm_client import LLMClient
from app.services.prompt_service import PromptService


class VisionAgent:
    def __init__(self, llm: LLMClient, prompts: PromptService) -> None:
        self.llm = llm
        self.prompts = prompts
        self.gate = LocalPalmGate()
        self.enhancer = ImageEnhancer()

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        if not raw:
            return None
        raw = raw.strip()
        raw = raw.replace("```json", "").replace("```", "")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _fallback_profile(blur_score: float) -> PalmFeatureProfile:
        notes = ["未接入视觉大模型，以下为降级提取结果。"]
        if blur_score < 110:
            notes.append("图像清晰度一般，细线判断可信度较低。")
        else:
            notes.append("图像清晰度尚可，可进行初步分析。")

        return PalmFeatureProfile(
            finger_gap="中等",
            fingerprint_pattern="纹理可见但细节不足",
            life_line="可见，深浅待二次确认",
            head_line="可见，走向较平直",
            heart_line="可见，末端细节不明确",
            career_line="隐约可见",
            sun_line="不明显",
            marriage_line="需近距离图像确认",
            notes=notes,
        )

    def analyze(self, image_bytes: bytes) -> tuple[GateResult, PalmFeatureProfile | None]:
        gate = self.gate.classify(image_bytes)
        if gate.category != GateCategory.PALM:
            return gate, None

        enhanced = self.enhancer.build(image_bytes)
        vision_prompt = self.prompts.load("vision_extract_prompt.txt")
        prompt = self.prompts.render(
            vision_prompt,
            {
                "blur_score": f"{enhanced.blur_score:.2f}",
                "image_shape": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
            },
        )

        raw = self.llm.vision(prompt=prompt, image_data_url=enhanced.edge_data_url)
        parsed = self._extract_json(raw)
        if not parsed:
            return gate, self._fallback_profile(enhanced.blur_score)

        parsed.setdefault("notes", [])
        parsed["notes"].append("提取基于图像与边缘增强视图。")
        try:
            profile = PalmFeatureProfile(**parsed)
        except Exception:  # noqa: BLE001
            profile = self._fallback_profile(enhanced.blur_score)
        return gate, profile
