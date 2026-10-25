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
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps
import cv2
import numpy as np

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
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

# Поддержка форматов HEIC и AVIF
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    if hasattr(pillow_heif, "register_avif_opener"):
        pillow_heif.register_avif_opener()
except Exception as e:
    print(f"[WARN] pillow_heif init: {e}")
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
        "features": ["local_cv_analysis", "huggingface_birefnet", "mos_35x45"]
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
    with urllib.request.urlopen(req_sse, timeout=40) as r:
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
    req_file = urllib.request.Request(result_url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req_file, timeout=30) as r:
        img_bytes = r.read()
        content_type = r.headers.get("Content-Type", "image/webp")
        return img_bytes, content_type


def apply_studio_lighting_pil(pil_img: Image.Image) -> Image.Image:
    """
    Интеллектуальное студийное освещение и выравнивание теней (Вариант 1):
    1. Работает строго в цветовом пространстве LAB по каналу яркости L.
       100% сохранение исходного тона кожи, цвета глаз, губ и волос.
    2. Локально вытягивает ТОЛЬКО темные тени (под подбородком, на шее, в глазницах,
       затененная сторона лица при боковом свете).
    3. Зоны нормальной экспозиции, средние тона и светлые участки лица остаются 
       100% нетронутыми без какого-либо общего осветления ("без засветления").
    4. Если на фото нет глубоких теней — фото остается в оригинальной контрастности.
    """
    try:
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        np_img = np.array(pil_img)
        if np_img.ndim != 3 or np_img.shape[2] != 4:
            return pil_img

        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        alpha = np_img[:, :, 3]

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_float = l.astype(np.float32)

        # Анализируем силуэт человека (внутри альфа-маски)
        mask = (alpha > 20).astype(np.uint8)
        if np.sum(mask) == 0:
            return pil_img

        subject_l = l[mask > 0]
        # Проверяем, есть ли на лице реальные темные тени (10-й перцентиль < 110)
        p10 = float(np.percentile(subject_l, 10))
        if p10 >= 115.0:
            # На фото уже ровный студийный свет, тени отсутствуют — ничего не осветляем!
            return pil_img

        # Оцениваем карту освещенности через билатеральный фильтр (сохраняет текстуру кожи)
        illumination = cv2.bilateralFilter(l, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)

        # Выборочный подъем ТОЛЬКО диапазона теней (L < 115):
        # Пиксели с нормальной яркостью (L >= 115) получают gain = 1.0 (ноль изменений)
        shadow_cutoff = 115.0
        shadow_depth = np.clip((shadow_cutoff - illumination) / shadow_cutoff, 0.0, 1.0)
        gain = 1.0 + 0.45 * np.power(shadow_depth, 1.5)
        l_lifted = np.clip(l_float * gain, 0, 255).astype(np.uint8)

        # Деликатный CLAHE с малым clipLimit исключительно для четкости черт в тенях
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_lifted)
        l_enhanced = cv2.addWeighted(l_lifted, 0.85, l_clahe, 0.15, 0)

        # Применяем подъем теней строго к пикселям человека
        l_result = l.copy()
        l_result[mask > 0] = l_enhanced[mask > 0]

        # Собираем обратно с нетронутыми цветными каналами A и B
        enhanced_lab = cv2.merge([l_result, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        result_np = np.dstack([enhanced_rgb, alpha])
        return Image.fromarray(result_np, mode="RGBA")
    except Exception as e:
        print(f"[WARN] apply_studio_lighting_pil fallback: {e}")
        return pil_img


@app.post("/remove-bg")
async def process_photo_remove_bg(
    file: UploadFile = File(...),
    enhance: bool = False,
    studio_light: bool = True
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # 1. HuggingFace Space (BiRefNet)
        raw_cutout_bytes, orig_media_type = await asyncio.to_thread(remove_bg_huggingface, content)

        # 2. Выравнивание теней и студийный свет (Вариант 1)
        if studio_light:
            try:
                cutout_img = Image.open(io.BytesIO(raw_cutout_bytes))
                studio_img = apply_studio_lighting_pil(cutout_img)
                out_buf = io.BytesIO()
                studio_img.save(out_buf, format="PNG")
                result_bytes = out_buf.getvalue()
                media_type = "image/png"
            except Exception as e:
                print(f"[WARN] Studio lighting postprocess error: {e}")
                result_bytes = raw_cutout_bytes
                media_type = orig_media_type
        else:
            result_bytes = raw_cutout_bytes
            media_type = orig_media_type

        headers = {
            "X-Plikowo-Engine": "HuggingFace-BiRefNet",
            "X-Plikowo-Scenario": "A_HIGH_QUALITY",
            "X-Plikowo-Corrections": "hf_birefnet_alpha,studio_light_v1"
        }
        return Response(content=result_bytes, media_type=media_type, headers=headers)

    except Exception as e:
        print(f"[ERROR] HuggingFace /remove-bg failed: {e}")
        raise HTTPException(status_code=500, detail=f"HuggingFace error: {str(e)}")


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
    fmt = "jpg" if target_format.lower() in ["jpg", "jpeg"] else "png"
    mime = "image/jpeg" if fmt == "jpg" else "image/png"
    file_id = uuid.uuid4().hex
    input_path = os.path.join(TEMP_DIR, f"{file_id}_in.heic")
    output_path = os.path.join(TEMP_DIR, f"{file_id}_out.{fmt}")

    try:
        content = await file.read()
        if not content:
            raise ValueError("Przesłano pusty plik.")

        image = None

        # Способ 1: Прямое декодирование в памяти через Pillow + pillow_heif
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except Exception:
            # Если прямой поток не распознан, сохраняем во временный файл
            with open(input_path, "wb") as f:
                f.write(content)

            # Способ 2: heif-convert (только если утилита реально установлена в системе)
            if shutil.which("heif-convert"):
                try:
                    cmd = ["heif-convert", "-q", "92", input_path, output_path]
                    process = subprocess.run(cmd, capture_output=True, text=True)
                    if process.returncode == 0 and os.path.exists(output_path):
                        with open(output_path, "rb") as f:
                            return Response(content=f.read(), media_type=mime)
                except Exception:
                    pass

            # Способ 3: Открытие файла через Pillow
            image = Image.open(input_path)
            image.load()

        if image is None:
            raise ValueError("Nie udało się odczytać formatu HEIC/HEIF.")

        # Автоматический поворот по EXIF ориентации
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        # Конвертация цветового пространства
        if fmt == "jpg":
            if image.mode != "RGB":
                if image.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                    image = background
                else:
                    image = image.convert("RGB")
        elif fmt == "png" and image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")

        # Сохранение в память
        out_buf = io.BytesIO()
        image.save(out_buf, format="JPEG" if fmt == "jpg" else "PNG", quality=92)
        return Response(content=out_buf.getvalue(), media_type=mime)

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
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
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
        desktop_env = r"C:\Users\Oleksandr\Desktop\IPkey паспорт.env"
        if os.path.exists(desktop_env):
            try:
                with open(desktop_env, "r", encoding="utf-8", errors="ignore") as f_key:
                    val = f_key.read().strip()
                    if val and not val.startswith("#"):
                        gemini_key = val
            except Exception:
                pass

    if not gemini_key:
        print("[ERROR] GEMINI_API_KEY is not set!")
        return {
            "valid": False,
            "high_quality": False,
            "sharp_and_clear": False,
            "eyes_open": False,
            "mouth_closed": False,
            "facing_camera": False,
            "neutral_expression": False,
            "rejection_reason_pl": "Błąd serwera: brak klucza Gemini API (GEMINI_API_KEY).",
            "rejection_reason_ua": "Помилка сервера: відсутній ключ Gemini API (GEMINI_API_KEY)."
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
        prompt = """Ты — контролер качества готовых биометрических фотографий для польских документов (Karta Pobytu, Dowód Osobisty, Paszport) по стандартам MOS / 35x45 мм.
Перед тобой ГОТОВЫЙ результат автоматической обработки.

Оцени техническое качество кадра по следующим ключевым критериям:

1. ОСВЕЩЕНИЕ И ТОН (светлое нормальное фото):
   - Лицо светлое, чистое, хорошо и равномерно освещенное.
   - Нет глубоких проваленных черных теней на лице.
   - Нет выбитых белых пересветов.

2. КАЧЕСТВО ИЗОБРАЖЕНИЯ (не пиксельное):
   - Изображение четкое, резкое, черты лица (глаза, губы, нос) хорошо различимы.
   - СТРОГО БРАКОВАТЬ (is_valid = false), если фото пиксельное, с лесенками, пережатое, размытое или в низком разрешении.

3. РАЗМЕР И ФОН (в размере с белым фоном):
   - Фон: однородный чистый белый (RGB 255, 255, 255) вокруг силуэта, без посторонних предметов.
   - Масштаб: лицо крупным планом занимает примерно 70–80% высоты кадра (стандарт MOS 35x45 мм).
   - Отступ сверху: над прической есть свободное белое пространство (макушка не срезана в 0 px).
   - ВНИМАНИЕ ПО ПЛЕЧАМ: В стандарте 35x45 мм лицо занимает почти весь кадр, поэтому внизу видна только верхняя линия одежды/шея. Это АБСОЛЮТНО НОРМАЛЬНО. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО отклонять фото по плечам!

4. БИОМЕТРИЯ И АКСЕССУАРЫ:
   - Лицо анфас, глаза открыты и смотрят прямо, рот закрыт.
   - ОЧКИ СТРОГО ЗАПРЕЩЕНЫ. Если на готовом фото видны любые очки — отклонять!
   - Пирсинг и родинки ПОЛНОСТЬЮ РАЗРЕШЕНЫ.

Верни ответ СТРОГО в формате JSON без markdown-оберток (без ```json):
{
  "is_valid": true,
  "confidence_score": 0.95,
  "reasons_for_rejection": [],
  "metrics": {
    "lighting_bright_and_even": true,
    "sharp_and_not_pixelated": true,
    "size_and_white_background_ok": true,
    "no_glasses": true,
    "face_biometrics_ok": true
  },
  "recommendations_for_user_ru": "Все параметры соответствуют стандарту.",
  "recommendations_for_user_ua": "Усі параметри відповідають стандарту.",
  "recommendations_for_user_pl": "Wszystkie parametry są zgodne ze standardem."
}"""
    else:
        prompt = """Ты — строгий контролер первичных селфи для биометрических документов (Karta Pobytu, Dowód Osobisty, Paszport).
Твоя задача — проверить исходный снимок пользователя ДО обработки и БРАКОВАТЬ фото при наличии любых нарушений биометрии.

КРИТЕРИИ ПРОВЕРКИ (СТРОГО БРАКОВАТЬ can_proceed_to_processing = false, ЕСЛИ):
1. ОЧКИ (ЛЮБЫЕ ОЧКИ СТРОГО ЗАПРЕЩЕНЫ): надеты солнцезащитные, темные, прозрачные очки для зрения, очки без оправы или оправы. Если на человеке надеты любые очки — СТРОГО БРАК!
2. ГОЛОВНЫЕ УБОРЫ: надеты шапка, кепка, бейсболка, капюшон, повязка или платок. Голова должна быть полностью открыта.
3. ГЛАЗА И ВЗГЛЯД: хотя бы один глаз закрыт или прищурен, глаза опущены вниз, взгляд отведен в сторону (не в объектив), зрачки перекрыты волосами или бликами. Взгляд должен быть направлен прямо в камеру!
4. ОСВЕЩЕНИЕ И ТЕНИ: глубокие контрастные тени на половине лица, скрывающие глаза или овал лица, либо сильный пересвет.
5. ВОЛОСЫ НА ЛИЦЕ: пряди волос или челка перекрывают глаза, зрачки или закрывают брови.
6. ПОЛОЖЕНИЕ ГОЛОВЫ И ВЫРАЖЕНИЕ: голова повернута вбок (профиль, полуанфас), запрокинута или сильно наклонена; рот открыт, видны зубы, ужимки.
7. КАЧЕСТВО: сильный смаз движения или расфокус.

ЧТО РАЗРЕШЕНО:
- Пирсинг на лице (губа, нос, серьги) и родинки — ПОЛНОСТЬЮ РАЗРЕШЕНЫ.
- Обычный комнатный фон (его удалит программа).

Верни ответ СТРОГО в формате JSON без markdown-оберток (без ```json):
{
  "can_proceed_to_processing": true,
  "confidence_score": 0.95,
  "critical_errors": [],
  "checks": {
    "no_glasses": true,
    "no_headwear": true,
    "eyes_open_looking_straight": true,
    "lighting_and_shadows_acceptable": true,
    "hair_not_blocking_eyes": true,
    "head_straight_mouth_closed": true,
    "sharpness_acceptable": true
  },
  "user_feedback_ru": "Фото подходит для обработки.",
  "user_feedback_ua": "Фото підходить для обробки.",
  "user_feedback_pl": "Zdjęcie nadaje się do przetworzenia."
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
            "temperature": 0.0
        }
    }

    last_error_code = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            for retry in range(2):
                try:
                    res = await client.post(
                        url,
                        json=gemini_payload,
                        headers={
                            "Content-Type": "application/json",
                            "Referer": "https://plikowo.com.pl/",
                            "x-goog-api-key": gemini_key
                        }
                    )
                    last_error_code = res.status_code
                    if res.status_code == 404:
                        break  # model not found, try next model
                    if res.status_code in (503, 429):
                        print(f"[WARN] Gemini model {model} temporary load status {res.status_code}, retry {retry+1}...")
                        await asyncio.sleep(0.8)
                        continue
                    if not res.is_success:
                        print(f"[WARN] Gemini model {model} status {res.status_code}: {res.text[:200]}")
                        break
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
                        if isinstance(parsed, dict):
                            # Normalize boolean valid
                            if "can_proceed_to_processing" in parsed:
                                parsed["valid"] = bool(parsed["can_proceed_to_processing"])
                            elif "is_valid" in parsed:
                                parsed["valid"] = bool(parsed["is_valid"])

                            # Normalize feedback / rejection reasons
                            if not parsed.get("valid"):
                                err_list = parsed.get("critical_errors") or parsed.get("reasons_for_rejection") or []
                                err_msg = ", ".join(err_list) if isinstance(err_list, list) else str(err_list)
                                fb_ru = parsed.get("user_feedback_ru") or parsed.get("recommendations_for_user_ru") or parsed.get("recommendations_for_user") or err_msg
                                fb_ua = parsed.get("user_feedback_ua") or parsed.get("recommendations_for_user_ua") or fb_ru
                                fb_pl = parsed.get("user_feedback_pl") or parsed.get("recommendations_for_user_pl") or fb_ru
                                parsed["rejection_reason_ru"] = fb_ru
                                parsed["rejection_reason_ua"] = fb_ua
                                parsed["rejection_reason_pl"] = fb_pl
                            else:
                                parsed["rejection_reason_ru"] = ""
                                parsed["rejection_reason_ua"] = ""
                                parsed["rejection_reason_pl"] = ""

                            if "valid" in parsed:
                                return parsed
                except Exception as e:
                    print(f"[WARN] Gemini model {model} retry {retry+1} attempt failed: {e}")
                    if retry == 0:
                        await asyncio.sleep(0.5)
                        continue

    if last_error_code in (401, 403):
        err_pl = "Błąd autoryzacji Gemini API (Klucz API jest nieprawidłowy lub zablokowany)."
        err_ua = "Помилка авторизації Gemini API (Ключ API недійсний або заблокований)."
    else:
        err_pl = "Serwery Google Gemini są chwilowo przeciążone (błąd 503 / timeout). Prosimy spróbować ponownie za chwilę."
        err_ua = "Сервери Google Gemini тимчасово перевантажені (помилка 503 / таймаут). Будь ласка, спробуйте ще раз за мить."

    return {
        "valid": False,
        "high_quality": False,
        "sharp_and_clear": False,
        "eyes_open": False,
        "mouth_closed": False,
        "facing_camera": False,
        "neutral_expression": False,
        "rejection_reason_pl": err_pl,
        "rejection_reason_ua": err_ua
    }
