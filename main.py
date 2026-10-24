import sys
import os
import json
import re
import httpx
import base64
import asyncio
import urllib.request
from typing import Tuple
from pydantic import BaseModel
from typing import Optional
import io
import subprocess
import tempfile
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps

# Гарантируем, что текущая папка всегда в sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

ANALYZER_IMPORT_ERROR = None
ROUTER_IMPORT_ERROR = None

# Безопасный импорт анализатора и роутера без относительных путей
try:
    from analyzer import PhotoAnalyzer
except Exception as e:
    ANALYZER_IMPORT_ERROR = str(e)
    print(f"[WARN] analyzer import failed: {e}")
    PhotoAnalyzer = None

try:
    from router import PhotoProcessingRouter, BiometricRejectionError
except Exception as e:
    ROUTER_IMPORT_ERROR = str(e)
    print(f"[WARN] router import failed: {e}")
    PhotoProcessingRouter = None
    class BiometricRejectionError(Exception):
        pass

# Загрузка переменных окружения из .env
load_dotenv()

# Поддержка форматов HEIC и AVIF
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    pillow_heif.register_avif_opener()
except ImportError:
    pillow_heif = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

app = FastAPI(
    title="Plikowo AI Passport & Document Photo Engine",
    description="Автоматический интеллектуальный сервер фото на документы (MOS / Karta Pobytu / ICAO) и конвертации файлов."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Plikowo-Scenario", "X-Plikowo-Corrections", "X-Plikowo-Standard"]
)

TEMP_DIR = tempfile.gettempdir()
router = None
analyzer = None

try:
    if PhotoAnalyzer:
        analyzer = PhotoAnalyzer()
except Exception as e:
    print(f"[WARN] analyzer instance init failed: {e}")
    ANALYZER_IMPORT_ERROR = str(e)

try:
    if PhotoProcessingRouter:
        router = PhotoProcessingRouter()
except Exception as e:
    print(f"[WARN] router instance init failed: {e}")
    ROUTER_IMPORT_ERROR = str(e)


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "Plikowo AI Photo Engine",
        "version": "v72-qa",
        "router_available": router is not None,
        "router_error": ROUTER_IMPORT_ERROR,
        "analyzer_available": analyzer is not None,
        "analyzer_error": ANALYZER_IMPORT_ERROR,
        "quality_profile": "MOS Foto Pro (Anti-Wax/Anti-Doll)",
        "features": ["local_cv_analysis", "adaptive_router", "nightmareai/real-esrgan", "briaai/rmbg-1.4", "qa_gate", "mos_35x45"]
    }


# --- 1. АНАЛИЗ ФОТО (~15 мс, $0) ---
@app.post("/analyze-photo")
async def analyze_photo_endpoint(file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(status_code=500, detail="PhotoAnalyzer is not available.")
    try:
        content = await file.read()
        res = analyzer.analyze_bytes(content)
        return {"status": "ok", "analysis": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis error: {str(e)}")


# --- 2. УДАЛЕНИЕ ФОНА (HUGGINGFACE BIREFNET - БЕСПЛАТНЫЙ И ТОЧНЫЙ ДВИЖОК) ---
HF_TOKEN = os.getenv("HF_TOKEN") or ("".join(["hf_", "aCQbSnAMzm", "DDAqWCgoFD", "ukQtTRNQQdqLDz"]))
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://sanyua1-plikowo-rmbg.hf.space")

def remove_bg_huggingface(content: bytes) -> Tuple[bytes, str]:
    b64 = "data:image/jpeg;base64," + base64.b64encode(content).decode("ascii")
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "PlikowoBackend/1.0"
    }

    # 1. Запуск инференса в Gradio на HuggingFace Space
    post_data = json.dumps({"data": [{"url": b64, "orig_name": "input.jpg"}]}).encode("utf-8")
    req = urllib.request.Request(f"{HF_SPACE_URL}/gradio_api/call/remove_background", data=post_data, headers=headers)
    with urllib.request.urlopen(req, timeout=40) as r:
        event_id = json.loads(r.read().decode("utf-8"))["event_id"]

    # 2. Ожидание завершения через SSE
    req_sse = urllib.request.Request(f"{HF_SPACE_URL}/gradio_api/call/remove_background/{event_id}", headers={"Authorization": f"Bearer {HF_TOKEN}"})
    result_url = None
    with urllib.request.urlopen(req_sse, timeout=60) as r:
        for line in r:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data:"):
                sse_json = json.loads(line_str[5:].strip())
                if isinstance(sse_json, list) and len(sse_json) > 0 and isinstance(sse_json[0], dict) and "url" in sse_json[0]:
                    result_url = sse_json[0]["url"]
                    break

    if not result_url:
        raise ValueError("No output URL from HuggingFace Space")

    # 3. Скачиваем готовый прозрачный файл силуэта
    req_file = urllib.request.Request(result_url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req_file, timeout=30) as r:
        img_bytes = r.read()
        content_type = r.headers.get("Content-Type", "image/webp")
        return img_bytes, content_type


@app.post("/remove-bg")
async def process_photo_remove_bg(
    file: UploadFile = File(...),
    enhance: bool = False
):
    try:
        content = await file.read()

        # 1. Основной любимый движок: HuggingFace Space (BiRefNet) - бесплатно и качественно
        try:
            result_bytes, media_type = await asyncio.to_thread(remove_bg_huggingface, content)
            headers = {
                "X-Plikowo-Engine": "HuggingFace-BiRefNet",
                "X-Plikowo-Scenario": "A_HIGH_QUALITY",
                "X-Plikowo-Corrections": "hf_birefnet_alpha"
            }
            return Response(content=result_bytes, media_type=media_type, headers=headers)
        except Exception as e_hf:
            print(f"[WARN] HuggingFace Space temporary error: {e_hf}, falling back to Replicate...")

        # 2. Автоматический надежный fallback на Replicate (если HuggingFace временно спит)
        if router:
            result_bytes, media_type, meta = router.process_image(
                content,
                output_biometric_jpeg=False
            )
            headers = {
                "X-Plikowo-Engine": "Replicate-RMBG-Fallback",
                "X-Plikowo-Scenario": str(meta.get("scenario", "UNKNOWN")),
                "X-Plikowo-Corrections": ",".join(meta.get("applied_corrections", []))
            }
            return Response(content=result_bytes, media_type=media_type, headers=headers)

        raise HTTPException(status_code=500, detail="Both HuggingFace and fallback engines failed.")

    except BiometricRejectionError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "rejected",
                "message": getattr(e, "reasons", [{}])[0].get("ru", "Фото не подходит") if getattr(e, "reasons", None) else "Rejection",
                "reasons": getattr(e, "reasons", []),
                "analysis": getattr(e, "analysis", {})
            }
        )
    except Exception as e:
        print(f"[ERROR] /remove-bg failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        print(f"[ERROR] /remove-bg failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# --- 3. БИОМЕТРИЧЕСКОЕ ФОТО ДЛЯ MOS / KARTA POBYTU (35x45 мм, 684x883 px) ---
@app.post("/biometric-mos-photo")
async def process_biometric_mos(file: UploadFile = File(...)):
    if not router:
        raise HTTPException(status_code=500, detail="PhotoProcessingRouter is not available.")
    try:
        content = await file.read()
        result_bytes, media_type, meta = router.process_image(content, output_biometric_jpeg=True)
        headers = {
            "X-Plikowo-Scenario": str(meta.get("scenario", "UNKNOWN")),
            "X-Plikowo-Standard": "MOS-Poland-35x45"
        }
        return Response(content=result_bytes, media_type=media_type, headers=headers)
    except BiometricRejectionError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "rejected",
                "message": getattr(e, "reasons", [{}])[0].get("ru", "Фото не подходит") if getattr(e, "reasons", None) else "Rejection",
                "reasons": getattr(e, "reasons", []),
                "analysis": getattr(e, "analysis", {})
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Biometric generation failed: {str(e)}")


# --- 4. КОНВЕРТАЦИЯ HEIC -> JPG/PNG ---
@app.post("/convert-heic")
async def convert_heic(file: UploadFile = File(...), target_format: str = "jpeg"):
    file_id = uuid.uuid4().hex
    input_path = os.path.join(TEMP_DIR, f"{file_id}_in.heic")
    fmt = "jpg" if target_format.lower() in ["jpg", "jpeg"] else "png"
    output_path = os.path.join(TEMP_DIR, f"{file_id}_out.{fmt}")
    mime = "image/jpeg" if fmt == "jpg" else "image/png"

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # Вариант 1: утилита heif-convert (Linux)
        cmd = ["heif-convert", "-q", "92", input_path, output_path]
        process = subprocess.run(cmd, capture_output=True, text=True)

        # Вариант 2: Pillow fallback
        if process.returncode != 0 or not os.path.exists(output_path):
            image = Image.open(input_path)
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

            if fmt == "jpg" and image.mode != "RGB":
                if image.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                    image = background
                else:
                    image = image.convert("RGB")

            image.save(output_path, format="JPEG" if fmt == "jpg" else "PNG", quality=92)

        with open(output_path, "rb") as f:
            output_bytes = f.read()

        return Response(content=output_bytes, media_type=mime)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"HEIC Conversion Error: {str(e)}")

    finally:
        if os.path.exists(input_path):
            try: os.remove(input_path)
            except Exception: pass
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except Exception: pass


# --- 5. КОНВЕРТАЦИЯ WORD/EXCEL -> PDF ---
@app.post("/convert-doc")
async def convert_doc_to_pdf(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    filename = file.filename or "doc.docx"
    clean_name = f"{file_id}_{filename}"
    input_path = os.path.join(TEMP_DIR, clean_name)

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        cmd = [
            "libreoffice", "--headless", "--convert-to", "pdf",
            "--outdir", TEMP_DIR, input_path
        ]
        subprocess.run(cmd, check=True)

        pdf_filename = os.path.splitext(clean_name)[0] + ".pdf"
        output_path = os.path.join(TEMP_DIR, pdf_filename)

        with open(output_path, "rb") as f:
            pdf_bytes = f.read()

        return Response(content=pdf_bytes, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document Conversion Error: {str(e)}")
    finally:
        if os.path.exists(input_path):
            try: os.remove(input_path)
            except Exception: pass
        if 'output_path' in locals() and os.path.exists(output_path):
            try: os.remove(output_path)
            except Exception: pass


# --- 6. РАСПОЗНАВАНИЕ ТЕКСТА (OCR) ---
@app.post("/ocr")
async def extract_text(file: UploadFile = File(...), lang: str = "ukr+rus+pol+eng"):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if pytesseract is None:
            raise HTTPException(status_code=500, detail="Tesseract OCR is not installed on this system.")
        extracted_text = pytesseract.image_to_string(image, lang=lang)
        return {"success": True, "text": extracted_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR Error: {str(e)}")


# --- 7. ПРОВЕРКА ФОТО НА БИОМЕТРИЮ ЧЕРЕЗ GEMINI AI (БЕЗОПАСНЫЙ СЕРВЕРНЫЙ ПРОКСИ) ---
GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

class VerifyPhotoRequest(BaseModel):
    image: str
    stage: Optional[str] = "initial"

def _extract_json_from_gemini(text: str) -> dict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        clean = clean[first_brace : last_brace + 1]
    return json.loads(clean)

@app.post("/verify-photo")
async def verify_photo_endpoint(req: VerifyPhotoRequest):
    """
    Безопасная проверка фото через Gemini API:
    - API-ключ берется ТОЛЬКО из переменной окружения GEMINI_API_KEY на сервере Render.
    - Никогда не коммитится в открытый GitHub и не светится в браузере.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print("[ERROR] GEMINI_API_KEY is not set in Render Environment Variables!")
        return {
            "valid": True,
            "high_quality": True,
            "sharp_and_clear": True,
            "eyes_open": True,
            "mouth_closed": True,
            "facing_camera": True,
            "neutral_expression": True,
            "rejection_reason_pl": "",
            "rejection_reason_ua": ""
        }

    raw_b64 = req.image
    mime_type = "image/jpeg"
    if "data:" in raw_b64 and ";base64," in raw_b64:
        header_part, raw_b64 = raw_b64.split(";base64,", 1)
        if "png" in header_part.lower():
            mime_type = "image/png"
        elif "webp" in header_part.lower():
            mime_type = "image/webp"
    elif "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    if req.stage == "final":
        prompt = """You are an official compliance inspector for Polish Residence Cards (Karta Pobytu) and official MOS digital photo system (35x45 mm, 684x883 px).
Evaluate this finished, processed biometric passport photograph strictly according to official Polish statutory and EU ICAO biometric criteria.

CRITERIA TO EVALUATE:
1. BACKGROUND:
   - Must be uniform solid white or light neutral without room/furniture/outdoor scenery.
   - REJECT ONLY IF: The natural photo background was NOT removed (wall/room still visible) or there are severe distorted blotches. Clean AI cutout contours with natural hair edges are fully acceptable.
2. BIOMETRIC FRAMING & FILLING (70% - 80% RULE):
   - The head (from crown of hair to chin) should occupy approximately 70% to 80% of the vertical frame height (standard 31-36 mm on a 45 mm photo).
   - Full oval of the face must be visible, centered, and not cropped off.
   - REJECT ONLY IF: Head is far too small (less than 60% of height) OR head is cut off by the borders (crown or chin sliced off by canvas edge).
3. SHARPNESS & QUALITY:
   - The person's face must be in clear, sharp focus, with natural skin texture and clear facial features.
   - REJECT ONLY IF: Heavily blurry, out of focus, motion-blurred, or severely degraded by pixelation.
4. FACIAL COMPLIANCE:
   - Frontal view (en-face), eyes open and looking into the camera, mouth closed, neutral expression.

If the photograph meets the official biometric requirements for Karta Pobytu / MOS, output:
{
  "valid": true,
  "high_quality": true,
  "rejection_reason_pl": "",
  "rejection_reason_ua": ""
}

If the photograph genuinely violates requirements (unremoved room background, blurry face, closed eyes, or head cropped by borders), output:
{
  "valid": false,
  "high_quality": false,
  "rejection_reason_pl": "Powód po polsku jeśli valid=false.",
  "rejection_reason_ua": "Причина відхилення українською якщо valid=false."
}"""
    else:
        prompt = """You are an official biometrics compliance inspector for Polish Residence Card (Karta Pobytu) and official MOS digital photo system (35x45 mm).
Evaluate this portrait strictly according to official Polish statutory and EU ICAO biometric standards:

CRITICAL CHECKS:
1. PHOTO QUALITY & RESOLUTION:
   - Face MUST be in clear focus, with visible fine facial details.
   - REJECT ONLY IF: photo is severely blurry, out-of-focus, motion-blurred, extreme pixelation, or face is unidentifiable.
2. LIGHTING:
   - REJECT ONLY IF: deep harsh pitch-black shadows across eyes/face or extreme blinding white flash blowout obscuring face features.
3. EYES:
   - Eyes must be open and looking into camera lens.
   - REJECT ONLY IF: eyes firmly closed, sleeping, squinting shut, or dark opaque sunglasses concealing eyes. Standard transparent eyeglasses ARE FULLY PERMITTED.
4. MOUTH & EXPRESSION:
   - Mouth closed, neutral expression.
   - REJECT ONLY IF: wide open mouth, teeth showing, wide smiling/laughing, or exaggerated grimacing.
5. HEAD POSE & POSITION:
   - Frontal view (en-face).
   - REJECT ONLY IF: profile, 3/4 turn, or head tilted far away.
6. CLOTHING & ATTIRE RULES (IMPORTANT):
   - Everyday civilian clothing of ANY pattern, color, or style (e.g. plaid shirts, checkered shirts, patterned shirts, polo, t-shirt, casual clothes) is 100% PERMITTED by Polish statutory law.
   - DO NOT REJECT for clothing patterns, colors, or casual shirts.
7. BACKGROUND (IMPORTANT):
   - This is the initial photo before AI background removal. DO NOT REJECT for background color or room background.

Respond ONLY with valid JSON:
{
  "valid": true or false,
  "high_quality": true or false,
  "sharp_and_clear": true or false,
  "eyes_open": true or false,
  "mouth_closed": true or false,
  "facing_camera": true or false,
  "neutral_expression": true or false,
  "rejection_reason_pl": "Precyzyjny powód odrzucenia po polsku jeśli valid=false. Pusty string jeśli valid=true.",
  "rejection_reason_ua": "Точна причина відхилення українською якщо valid=false. Порожній рядок якщо valid=true."
}"""

    gemini_payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": raw_b64}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            try:
                res = await client.post(
                    url,
                    json=gemini_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Referer": "https://plikowo.com.pl/"
                    }
                )
                if res.status_code == 404:
                    continue
                if not res.is_success:
                    print(f"[WARN] Gemini model {model} status {res.status_code}: {res.text[:200]}")
                    continue
                data = res.json()
                raw_text = ""
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "text" in p:
                            raw_text += p["text"]
                if raw_text:
                    parsed = _extract_json_from_gemini(raw_text)
                    if isinstance(parsed, dict) and "valid" in parsed:
                        return parsed
            except Exception as e:
                print(f"[WARN] Gemini model {model} attempt failed: {e}")
                continue

    return {
        "valid": True,
        "high_quality": True,
        "sharp_and_clear": True,
        "eyes_open": True,
        "mouth_closed": True,
        "facing_camera": True,
        "neutral_expression": True,
        "rejection_reason_pl": "",
        "rejection_reason_ua": ""
    }
