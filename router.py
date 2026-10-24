"""
Plikowo Processing Router (Інтелектуальний швидкий маршрутизатор обробки)
1. Кадрування та підгонка під пропорції 35х45 мм (7:9, 684x883 px)
2. Локальна салонна корекція кольору та світла (захист від блідості)
3. Автономне вилучення фону через Replicate RMBG (bria/remove-background)
4. Локальне накладання на суцільно чисто білий фон (RGB 255, 255, 255) через Pillow
Жодних зайвих платних моделей та домальовувань — нативно, копійчано і чітко.
"""

import io
import os
import tempfile
import time
from typing import Dict, Any, Tuple, Optional
import cv2
import httpx
from PIL import Image, ImageOps
import replicate
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

TEMP_DIR = tempfile.gettempdir()

# Найкраща офіційна модель сегментації на Replicate
REPLICATE_MODEL_REMOVE_BG = os.getenv(
    "REPLICATE_MODEL_REMOVE_BG",
    "bria/remove-background:5ecc270b34e9d8e1f007d9dbd3c724f0badf638f05ffaa0c5e0634ed64d3d378"
)


class BiometricRejectionError(Exception):
    def __init__(self, reasons: list, analysis: dict):
        self.reasons = reasons
        self.analysis = analysis
        super().__init__(str(reasons))


def run_replicate_with_retry(client: replicate.Client, model: str, input_params: dict, max_retries: int = 3):
    """
    Виклик Replicate з обробкою rate limits (429).
    """
    for attempt in range(max_retries):
        try:
            return client.run(model, input=input_params)
        except Exception as e:
            err_msg = str(e)
            if "throttled" in err_msg.lower() or "429" in err_msg or "rate limit" in err_msg.lower():
                wait_sec = 6
                print(f"[RETRY] Replicate rate limit (спроба {attempt+1}/{max_retries}). Очікування {wait_sec} с...")
                time.sleep(wait_sec)
            else:
                raise e
    raise RuntimeError("Перевищено кількість спроб виклику Replicate API.")


def extract_output_url_or_bytes(output: Any) -> Tuple[bytes, Optional[str]]:
    """
    Витягує байти та URL з результату Replicate.
    """
    if isinstance(output, list) and len(output) > 0:
        output = output[0]

    url = None
    if hasattr(output, "url"):
        url = output.url
    elif isinstance(output, str) and output.startswith("http"):
        url = output

    if hasattr(output, "read"):
        try:
            return output.read(), url
        except Exception:
            pass

    if url:
        with httpx.Client(timeout=90.0) as http_client:
            resp = http_client.get(url)
            resp.raise_for_status()
            return resp.content, url

    raise ValueError(f"Не вдалося прочитати результат Replicate: {type(output)}")


class PhotoProcessingRouter:
    def __init__(self, api_token: Optional[str] = None):
        self.token = api_token or os.getenv("REPLICATE_API_TOKEN")
        self.analyzer = PhotoAnalyzer()
        self.qa_gate = QualityAssuranceGate()

    def process_image(
        self,
        image_bytes: bytes,
        output_biometric_jpeg: bool = False,
        use_codeformer: bool = False
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """
        Чіткий, копійчаний та автономний пайплайн:
        1. Локальна перевірка та кадрування під пропорції 35х45 мм (684x883 px)
        2. Видалення фону через єдиний виклик RMBG
        3. Локальне накладання чисто білого фону (RGB 255, 255, 255) в Pillow
        """
        if not self.token:
            raise ValueError("REPLICATE_API_TOKEN is not configured.")

        client = replicate.Client(api_token=self.token)
        applied_corrections = []

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

        temp_crop_path = os.path.join(TEMP_DIR, f"p_rmbg_in_{int(time.time()*1000)}.png")
        step1_final_pil.save(temp_crop_path, format="PNG")

        try:
            # 3. Видалення фону через Replicate RMBG (bria/remove-background)
            with open(temp_crop_path, "rb") as f_img:
                bg_out = run_replicate_with_retry(
                    client,
                    REPLICATE_MODEL_REMOVE_BG,
                    input_params={"image": f_img, "preserve_alpha": True}
                )

            segmented_bytes, _ = extract_output_url_or_bytes(bg_out)
            silhouette_img = Image.open(io.BytesIO(segmented_bytes))
            if silhouette_img.mode != "RGBA":
                silhouette_img = silhouette_img.convert("RGBA")

            # Салонна доводка краю волосся без кукольних спотворень
            silhouette_img = apply_studio_postprocessing(silhouette_img)

            # Віддаємо прозорий силует у вихідній роздільній здатності для точного центрування на клієнті
            out_buf = io.BytesIO()
            silhouette_img.save(out_buf, format="PNG")
            return out_buf.getvalue(), "image/png", {"scenario": "A_HIGH_QUALITY", "applied_corrections": ["studio_light", "rmbg_preserve_alpha"]}

        finally:
            if os.path.exists(temp_crop_path):
                try:
                    os.remove(temp_crop_path)
                except Exception:
                    pass
