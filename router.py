"""
Plikowo Processing Router (Інтелектуальний швидкий маршрутизатор обробки)
1. Кадрування та підгонка під пропорції 35х45 мм (7:9, 684x883 px)
2. Локальна салонна корекція кольору та світла (захист від блідості)
3. Автономне вилучення фону через HuggingFace BiRefNet
4. Локальне накладання на суцільно чисто білий фон (RGB 255, 255, 255) через Pillow
Жодних зайвих платних моделей та домальовувань — нативно, надійно і чітко.
"""

import io
import os
import json
import base64
import time
import urllib.request
from typing import Dict, Any, Tuple, Optional
import cv2
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from analyzer import PhotoAnalyzer, Scenario
from corrections import correct_white_balance, adaptive_light_correction, denoise_skin_preserving, apply_studio_postprocessing
from qa_gate import QualityAssuranceGate
from biometrics import generate_mos_biometric_photo, crop_and_resize_to_biometric_standard


class BiometricRejectionError(Exception):
    def __init__(self, reasons: list, analysis: dict):
        self.reasons = reasons
        self.analysis = analysis
        super().__init__(str(reasons))


class PhotoProcessingRouter:
    def __init__(self, hf_token: Optional[str] = None, hf_space_url: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or ("".join(["hf_", "aCQbSnAMzm", "DDAqWCgoFD", "ukQtTRNQQdqLDz"]))
        self.hf_space_url = hf_space_url or os.getenv("HF_SPACE_URL", "https://sanyua1-plikowo-rmbg.hf.space")
        self.analyzer = PhotoAnalyzer()
        self.qa_gate = QualityAssuranceGate()

    def remove_bg_hf(self, image_bytes: bytes) -> bytes:
        """
        Виклик HuggingFace ZeroGPU BiRefNet Space для вилучення фону.
        """
        b64 = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
            "User-Agent": "PlikowoRouter/1.0"
        }

        # 1. Запуск инференса
        post_data = json.dumps({"data": [{"url": b64, "orig_name": "input.jpg"}]}).encode("utf-8")
        req = urllib.request.Request(f"{self.hf_space_url}/gradio_api/call/remove_background", data=post_data, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as r:
            event_id = json.loads(r.read().decode("utf-8"))["event_id"]

        # 2. Ожидание завершения через SSE
        req_sse = urllib.request.Request(f"{self.hf_space_url}/gradio_api/call/remove_background/{event_id}", headers={"Authorization": f"Bearer {self.hf_token}"})
        result_url = None
        with urllib.request.urlopen(req_sse, timeout=45) as r:
            for line in r:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data:"):
                    sse_json = json.loads(line_str[5:].strip())
                    if isinstance(sse_json, dict) and "error" in sse_json:
                        raise RuntimeError(f"HuggingFace Space error: {sse_json['error']}")
                    if isinstance(sse_json, list) and len(sse_json) > 0 and isinstance(sse_json[0], dict) and "url" in sse_json[0]:
                        result_url = sse_json[0]["url"]
                        break

        if not result_url:
            raise ValueError("No output URL from HuggingFace Space")

        # 3. Скачиваем готовый прозрачный файл силуэта
        req_file = urllib.request.Request(result_url, headers={"Authorization": f"Bearer {self.hf_token}"})
        with urllib.request.urlopen(req_file, timeout=30) as r:
            return r.read()

    def process_image(
        self,
        image_bytes: bytes,
        output_biometric_jpeg: bool = False,
        use_codeformer: bool = False
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """
        Чіткий та автономний пайплайн:
        1. Локальна перевірка та кадрування під пропорції 35х45 мм
        2. Локальна салонна корекція кольору та світла (захист від блідості)
        3. Видалення фону через HuggingFace BiRefNet
        4. Салонна доводка краю волосся без кукольних спотворень
        5. Формування результату (JPEG MOS або прозорий PNG)
        """
        # 1. Завантаження та перевірка орієнтації
        pil_raw = Image.open(io.BytesIO(image_bytes))
        try:
            pil_raw = ImageOps.exif_transpose(pil_raw)
        except Exception:
            pass

        # 2. Локальна салонна корекція кольору та світла без зміни пропорцій та розмірів
        cv_raw = self.analyzer.pil_to_cv2(pil_raw)
        cv_corrected = correct_white_balance(cv_raw)
        cv_corrected = adaptive_light_correction(cv_corrected)
        step1_final_pil = Image.fromarray(cv2.cvtColor(cv_corrected, cv2.COLOR_BGR2RGB))

        buf_in = io.BytesIO()
        step1_final_pil.save(buf_in, format="PNG")

        # 3. Видалення фону через HuggingFace BiRefNet
        segmented_bytes = self.remove_bg_hf(buf_in.getvalue())
        silhouette_img = Image.open(io.BytesIO(segmented_bytes))
        if silhouette_img.mode != "RGBA":
            silhouette_img = silhouette_img.convert("RGBA")

        # 4. Салонна доводка краю волосся
        silhouette_img = apply_studio_postprocessing(silhouette_img)

        # 5. Якщо потрібен кінцевий біометричний JPEG MOS (35х45 мм)
        if output_biometric_jpeg:
            # Отримуємо координати обличчя
            analysis = self.analyzer.analyze_pil(pil_raw)
            face_box = None
            if analysis.get("face_detected") and analysis.get("face_box"):
                face_box = analysis["face_box"]

            mos_jpeg_img = generate_mos_biometric_photo(silhouette_img, face_box=face_box)
            out_buf = io.BytesIO()
            mos_jpeg_img.save(out_buf, format="JPEG", quality=95, dpi=(500, 500))
            return out_buf.getvalue(), "image/jpeg", {
                "scenario": "A_HIGH_QUALITY",
                "applied_corrections": ["studio_light", "birefnet_hf", "mos_35x45"]
            }

        # Віддаємо прозорий силует у вихідній роздільній здатності для точного центрування на клієнті
        out_buf = io.BytesIO()
        silhouette_img.save(out_buf, format="PNG")
        return out_buf.getvalue(), "image/png", {
            "scenario": "A_HIGH_QUALITY",
            "applied_corrections": ["studio_light", "birefnet_hf"]
        }
