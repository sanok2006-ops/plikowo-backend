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
