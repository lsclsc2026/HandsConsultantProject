import base64
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings
from app.core.schemas import GateCategory, GateResult


try:
    import mediapipe as mp  # type: ignore
except Exception:  # noqa: BLE001
    mp = None


@dataclass
class EnhancedImage:
    original_data_url: str
    edge_data_url: str
    blur_score: float
    shape: tuple[int, int]


class LocalPalmGate:
    def __init__(self) -> None:
        self.cfg = settings.vision
        self.hands = None
        if mp is not None:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
            )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray | None:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _mediapipe_score(self, image_bgr: np.ndarray) -> float:
        if not self.hands:
            return 0.0
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        if result.multi_hand_landmarks:
            return 0.88
        return 0.0

    @staticmethod
    def _skin_ratio_score(image_bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 25, 60], dtype=np.uint8)
        upper1 = np.array([20, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1)
        ratio = float(mask.mean() / 255.0)
        return min(1.0, ratio * 2.2)

    def classify(self, image_bytes: bytes) -> GateResult:
        image = self._decode_image(image_bytes)
        if image is None:
            return GateResult(
                category=GateCategory.NON_PALM,
                confidence=0.0,
                reason="图片解码失败，请重新上传。",
            )

        h, w = image.shape[:2]
        if h < self.cfg.min_height or w < self.cfg.min_width:
            return GateResult(
                category=GateCategory.BLURRY,
                confidence=0.45,
                reason="图片尺寸过小，请上传更清晰的手掌照片。",
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = self._laplacian_variance(gray)
        mp_score = self._mediapipe_score(image)
        skin_score = self._skin_ratio_score(image)

        hand_confidence = max(mp_score, 0.55 * mp_score + 0.45 * skin_score)
        blur_ratio = min(1.0, blur_score / max(self.cfg.blur_threshold, 1.0))

        if hand_confidence < self.cfg.non_palm_threshold and skin_score < 0.18:
            return GateResult(
                category=GateCategory.NON_PALM,
                confidence=round(hand_confidence, 3),
                reason="检测结果更像非手掌图片。",
            )

        # 自适应阈值：画面清晰时适度放宽手掌置信度门槛，减少误判为 blurry。
        pass_threshold = self.cfg.blurry_threshold
        if blur_score >= self.cfg.blur_threshold:
            pass_threshold = max(self.cfg.non_palm_threshold + 0.08, self.cfg.blurry_threshold - 0.14)

        # 极低清晰度仍需拦截，避免把不可识别图片送入后续链路。
        if blur_ratio < 0.45 and hand_confidence < 0.65:
            confidence = min(0.69, max(hand_confidence, 0.35))
            return GateResult(
                category=GateCategory.BLURRY,
                confidence=round(confidence, 3),
                reason="手掌图像偏模糊/反光，建议在明亮环境重拍。",
            )

        if hand_confidence < pass_threshold:
            confidence = min(0.69, max(hand_confidence, 0.35))
            return GateResult(
                category=GateCategory.BLURRY,
                confidence=round(confidence, 3),
                reason="手掌图像质量一般，但可尝试调整角度与光线后重拍。",
            )

        confidence = min(0.98, hand_confidence + 0.06)
        return GateResult(
            category=GateCategory.PALM,
            confidence=round(confidence, 3),
            reason="检测到有效手掌图像。",
        )


class ImageEnhancer:
    @staticmethod
    def _to_data_url(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return ""
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("图片无法解析")
        return image

    def build(self, image_bytes: bytes) -> EnhancedImage:
        image = self._decode_image(image_bytes)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced_gray, 7, 55, 55)
        edges = cv2.Canny(denoised, threshold1=40, threshold2=120)

        edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        merged = cv2.addWeighted(image, 0.72, edge_rgb, 0.28, 0)

        h, w = image.shape[:2]
        return EnhancedImage(
            original_data_url=self._to_data_url(image),
            edge_data_url=self._to_data_url(merged),
            blur_score=blur_score,
            shape=(h, w),
        )
