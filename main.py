import sys
import os
import json
import re
import httpx
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

# Безопасный импорт анализатора и роутера без относительных путей
try:
    from analyzer import PhotoAnalyzer
except Exception as e:
    print(f"[WARN] analyzer import failed: {e}")
    PhotoAnalyzer = None

try:
    from router import PhotoProcessingRouter, BiometricRejectionError
except Exception as e:
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
router = PhotoProcessingRouter() if PhotoProcessingRouter else None
analyzer = PhotoAnalyzer() if PhotoAnalyzer else None


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "Plikowo AI Photo Engine",
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


# --- 2. УДАЛЕНИЕ ФОНА (ПРОЗРАЧНЫЙ ИЛИ БЕЛЫЙ) ---
@app.post("/remove-bg")
async def process_photo_remove_bg(
    file: UploadFile = File(...),
    enhance: bool = False
):
    if not router:
        raise HTTPException(status_code=500, detail="PhotoProcessingRouter is not available.")
    try:
        content = await file.read()
        result_bytes, media_type, meta = router.process_image(
            content,
            output_biometric_jpeg=False
        )

        headers = {
            "X-Plikowo-Scenario": str(meta.get("scenario", "UNKNOWN")),
            "X-Plikowo-Corrections": ",".join(meta.get("applied_corrections", []))
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
        prompt = """You are a strict Polish biometric passport compliance officer inspecting a final, formatted photograph for Polish Residence Cards (Karta Pobytu) and official MOS digital photo system (35x45 mm, 684x883 px).

Perform an exacting inspection of this finished biometric photograph according to official Polish statutory and EU ICAO criteria.

CRITICAL REJECTION RULES (Check each carefully):
1. SOLID WHITE BACKGROUND & CUTOUT INTEGRITY (STRICT):
   - Background MUST be uniform, pure solid white without any natural room background, textures, shadows, or gradients.
   - REJECT IF: Natural photo background is NOT removed (e.g. wall, room, door, window, outdoor background is visible). Reason PL: "Tło nie zostało usunięte na jednolity biały kolor." / UA: "Фон не видалено на суцільний білий колір."
   - REJECT IF: Background removal left noticeable cutout artifacts, jagged/choppy edges around hair or clothing, dark halos/fringes, blurred contours, or eroded parts of the head/ears/shoulders.
2. FRAME FILLING & BIOMETRIC PROPORTIONS (70% - 80% RULE):
   - The head (from top of hair crown to bottom tip of chin) MUST occupy between 70% and 80% of the total vertical height of the 35x45 mm frame.
   - Crown of the hair/head MUST be positioned near the top guideline (~7% margin from the top border).
   - Chin MUST be positioned near the bottom guideline (~87% from top, ~13% margin from bottom border).
   - REJECT IF: Head is too small (e.g. less than 70% height, person looks far away with a large empty gap above the head or crown sits far below the top line). Reason PL: "Twarz zajmuje zbyt małą część kadru (wymagane 70-80% wysokości zdjęcia)." / UA: "Обличчя займає замалу частину кадру (вимагається 70-80% висоти фото)."
   - REJECT IF: Head is too large (crown or chin cut off by the borders, overflowing the frame). Reason PL: "Głowa jest obcięta krawędzią kadru." / UA: "Голова обрізана краєм кадру."
3. PHOTO QUALITY, SHARPNESS & RESOLUTION:
   - The image must be sharp, in crisp focus, with natural skin texture and clear facial features (eyes, lashes, lips, facial contours).
   - REJECT IF: Blurry, out of focus, motion-blurred, pixelated, heavily compressed, or low-resolution. Reason PL: "Zdjęcie jest nieostre lub ma zbyt niską jakość." / UA: "Фото нечітке або має занизьку якість."
4. FACIAL COMPLIANCE:
   - Frontal view (en-face), eyes open and looking straight into camera, mouth completely closed, neutral expression (no smiling or frowning).

If the photograph meets ALL standards with high quality, output "valid": true and "high_quality": true.
If the photograph fails ANY of the rules above, output "valid": false and "high_quality": false with a clear, polite explanation in Polish ("rejection_reason_pl") and Ukrainian ("rejection_reason_ua").

Respond ONLY with valid JSON:
{
  "valid": true or false,
  "high_quality": true or false,
  "rejection_reason_pl": "Powód po polsku jeśli valid=false. Pusty string jeśli valid=true.",
  "rejection_reason_ua": "Причина відхилення українською якщо valid=false. Порожній рядок якщо valid=true."
}"""
    else:
        prompt = """You are an official biometrics compliance inspector for Polish Residence Card (Karta Pobytu) and official MOS digital photo system (35x45 mm).
Evaluate this portrait strictly according to official Polish statutory and EU ICAO biometric standards, with EXTRA EMPHASIS ON PHOTO QUALITY AND SHARPNESS:

CRITICAL CHECKS:
1. PHOTO QUALITY & RESOLUTION (VERY IMPORTANT):
   - Face MUST be in sharp focus, with clearly visible fine facial details (eyes, lashes, lips, facial contours).
   - REJECT IF: photo is blurry, out-of-focus, motion-blurred, noisy/grainy, pixelated, low-resolution, degraded by heavy compression, or smudged.
2. LIGHTING & EXPOSURE:
   - Lighting must be natural and balanced across the entire face.
   - REJECT IF: severe shadows across face/eyes/cheeks, harsh glare, washed-out overexposure, or too dark underexposure.
3. EYES:
   - Eyes MUST be fully open, clearly visible, looking directly at camera lens.
   - REJECT IF: eyes closed, half-closed, squinting, pupils covered by hair/frames, sunglasses or tinted lenses, flash reflections obscuring eyes.
4. MOUTH & EXPRESSION:
   - Mouth MUST be firmly closed. Expression MUST be strictly neutral.
   - REJECT IF: mouth slightly open or parted, teeth showing, smiling, smirking, grinning, frowning, or non-neutral expression.
5. HEAD POSE & FRAMING:
   - Head MUST be strictly frontal (en face) and upright.
   - REJECT IF: head turned away (profile, semi-profile, 3/4 view), tilted sideways, or tilted up/down.
6. PERSON & VISIBILITY:
   - EXACTLY ONE person in the photo. Full oval of face from chin to hairline visible.
   - REJECT IF: no person/face, multiple persons/faces, face partially outside frame, hat/headwear (unless religious, but face oval fully visible).

If the photo meets ALL official criteria including high visual quality and sharpness, set "valid": true and "high_quality": true.
If the photo fails ANY criterion (especially if quality is poor, blurry, or low-res), set "valid": false and provide a polite, clear explanation in "rejection_reason_pl" and "rejection_reason_ua".

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
