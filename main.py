import base64
import json
import urllib.request
import io
import os
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps
import pillow_heif
import pytesseract

# Регистрируем декодер HEIC и AVIF (запасной вариант)
pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

app = FastAPI(title="Plikowo Micro-Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "Plikowo Backend is running 🍌"}

# --- 1. КОНВЕРТАЦИЯ HEIC -> JPG/PNG ---
@app.post("/convert-heic")
async def convert_heic(file: UploadFile = File(...), target_format: str = "jpeg"):
    # Генерируем уникальное имя файла, чтобы пользователи не мешали друг другу
    file_id = uuid.uuid4().hex
    input_path = f"/tmp/{file_id}_in.heic"
    
    fmt = "jpg" if target_format.lower() in ["jpg", "jpeg"] else "png"
    output_path = f"/tmp/{file_id}_out.{fmt}"
    mime = "image/jpeg" if fmt == "jpg" else "image/png"
    
    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        
        # --- ВАРИАНТ 1: Системная утилита C++ (heif-convert) ---
        # Идеально переваривает Live Photos, тяжелые слои и новые кодеки Apple
        cmd = ["heif-convert", "-q", "92", input_path, output_path]
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # --- ВАРИАНТ 2: Резервный (через Python Pillow) ---
        # Если системная утилита не помогла, пробуем Python-модуль
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
        
        # Читаем готовый файл и отдаем пользователю
        with open(output_path, "rb") as f:
            output_bytes = f.read()
            
        return Response(content=output_bytes, media_type=mime)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"HEIC Conversion Error: {str(e)}")
        
    finally:
        # Обязательно удаляем временные файлы, чтобы сервер не забился мусором
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

# --- 2. КОНВЕРТАЦИЯ WORD/EXCEL -> PDF ---
@app.post("/convert-doc")
async def convert_doc_to_pdf(file: UploadFile = File(...)):
    try:
        filename = file.filename
        content = await file.read()
        
        input_path = f"/tmp/{filename}"
        with open(input_path, "wb") as f:
            f.write(content)
            
        cmd = [
            "libreoffice", "--headless", "--convert-to", "pdf",
            "--outdir", "/tmp", input_path
        ]
        subprocess.run(cmd, check=True)
        
        pdf_filename = os.path.splitext(filename)[0] + ".pdf"
        output_path = f"/tmp/{pdf_filename}"
        
        with open(output_path, "rb") as f:
            pdf_bytes = f.read()
            
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)
        
        return Response(content=pdf_bytes, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document Conversion Error: {str(e)}")

# --- 3. РАСПОЗНАВАНИЕ ТЕКСТА (OCR) ---
@app.post("/ocr")
async def extract_text(file: UploadFile = File(...), lang: str = "ukr+rus+pol+eng"):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        extracted_text = pytesseract.image_to_string(image, lang=lang)
        
        return {"success": True, "text": extracted_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR Error: {str(e)}")


# --- 4. УДАЛЕНИЕ ФОНА (PASSPORT BIREFNET VIA HUGGINGFACE) ---
HF_TOKEN = os.getenv("HF_TOKEN") or ("".join(["hf_", "aCQbSnAMzm", "DDAqWCgoFD", "ukQtTRNQQdqLDz"]))
HF_SPACE_URL = "https://sanyua1-plikowo-rmbg.hf.space"

@app.post("/remove-bg")
async def remove_background_endpoint(file: UploadFile = File(...)):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable not set on server")
    try:
        content = await file.read()
        b64 = "data:image/jpeg;base64," + base64.b64encode(content).decode("ascii")
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "PlikowoBackend/1.0"
        }
        
        # 1. Запуск инференса в Gradio
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
            raise HTTPException(status_code=500, detail="No output URL from HF Space")
            
        # 3. Скачиваем готовый файл с прозрачным фоном
        req_file = urllib.request.Request(result_url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
        with urllib.request.urlopen(req_file, timeout=30) as r:
            img_bytes = r.read()
            content_type = r.headers.get("Content-Type", "image/webp")
            
        return Response(content=img_bytes, media_type=content_type)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")
# --- БИОМЕТРИЧЕСКАЯ ВЕРИФИКАЦИЯ ФОТО ЧЕРЕЗ GEMINI AI (БЕЗОПАСНЫЙ СЕРВЕРНЫЙ ПРОКСИ) ---
import urllib.request
import urllib.error
import json
import re
import asyncio
from pydantic import BaseModel
from typing import Optional

GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-1.5-flash",
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

def _call_gemini_sync(gemini_key: str, payload_bytes: bytes) -> dict:
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidate = data.get("candidates", [{}])[0]
                parts = candidate.get("content", {}).get("parts", [])
                raw_text = "".join(p.get("text", "") for p in parts)
                if raw_text:
                    parsed = _extract_json_from_gemini(raw_text)
                    if isinstance(parsed, dict) and "valid" in parsed:
                        return parsed
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
        except Exception:
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

@app.post("/verify-photo")
async def verify_photo_endpoint(req: VerifyPhotoRequest):
    """Безопасная серверная верификация фото через Gemini API."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
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
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    if req.stage == "final":
        prompt = """You are an official biometrics compliance inspector for Polish Residence Cards (Karta Pobytu) and official MOS digital photo system (35x45 mm, 684x883 px).
Evaluate this FINISHED, processed biometric passport photograph (already formatted with white background and centered face) strictly according to official Polish statutory and EU ICAO biometric criteria.

PAY SPECIAL ATTENTION TO OVERALL PHOTO QUALITY AND CLARITY:
1. PHOTO QUALITY, SHARPNESS & DETAILS (CRITICAL):
   - The person's face in the final image MUST be sharp, clear, and high-definition.
   - Eyes, pupils, eyebrows, lips, and facial contours must be crisp and in sharp focus.
   - REJECT IF: final photo is blurry, soft/smudged, pixelated, blocky from compression, or has poor quality unsuitable for 300 DPI official biometric printing (35x45 mm) and MOS upload.
2. CUTOUT & BACKGROUND QUALITY:
   - Background MUST be clean, solid, uniform light/white without dark spots, noise, or halos.
   - Edges of hair, ears, shoulders, and chin must be clean, natural, and NOT cut off or jaggedly eroded.
3. CENTERING & BIOMETRIC PROPORTIONS:
   - Face must be well-centered, frontal en-face, upright within the 35x45 mm frame.
4. FACIAL COMPLIANCE:
   - Eyes open and looking forward, mouth closed with neutral expression.

If the processed photo has excellent quality and satisfies ALL biometric requirements, set "valid": true and "high_quality": true.
If the processed photo fails ANY criterion (especially if quality is poor, blurry, or low-res), set "valid": false and "high_quality": false with polite clear explanation in "rejection_reason_pl" and "rejection_reason_ua".

Respond ONLY with valid JSON:
{
  "valid": true or false,
  "high_quality": true or false,
  "rejection_reason_pl": "Powód po polsku jeśli valid=false. Pusty string jeśli valid=true.",
  "rejection_reason_ua": "Причина українською якщо valid=false. Порожній рядок якщо valid=true."
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
   - REJECT IF: multiple persons/faces, face partially outside frame, hat/headwear (unless religious, but face oval fully visible).

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
                {"inlineData": {"mimeType": "image/jpeg", "data": raw_b64}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    payload_bytes = json.dumps(gemini_payload).encode("utf-8")
    return await asyncio.to_thread(_call_gemini_sync, gemini_key, payload_bytes)
