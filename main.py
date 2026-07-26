import io
import os
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import pillow_heif
import pytesseract

# Регистрируем декодер HEIC и AVIF (для самых новых iPhone) в библиотеке Pillow
pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

app = FastAPI(title="Plikowo Micro-Backend")

# Разрешаем веб-сайту делать запросы к серверу (CORS)
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
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        output_stream = io.BytesIO()
        fmt = "JPEG" if target_format.lower() in ["jpg", "jpeg"] else "PNG"
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        
        # ФИКС: Если картинка имеет прозрачность (RGBA) или другой режим, 
        # библиотека Pillow выдаст Ошибку 400 при попытке сохранить её в формат JPEG.
        # Поэтому принудительно конвертируем в обычный RGB перед сохранением в JPG.
        if fmt == "JPEG" and image.mode != "RGB":
            # Создаем белый фон, чтобы прозрачные участки не стали черными
            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                image = background
            else:
                image = image.convert("RGB")
        
        image.save(output_stream, format=fmt, quality=92)
        output_stream.seek(0)
        
        return Response(content=output_stream.getvalue(), media_type=mime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"HEIC Conversion Error: {str(e)}")

# --- 2. КОНВЕРТАЦИЯ WORD/EXCEL -> PDF ---
@app.post("/convert-doc")
async def convert_doc_to_pdf(file: UploadFile = File(...)):
    try:
        filename = file.filename
        content = await file.read()
        
        # Сохраняем временный файл
        input_path = f"/tmp/{filename}"
        with open(input_path, "wb") as f:
            f.write(content)
            
        # Запускаем LibreOffice в консоли для конвертации в PDF
        cmd = [
            "libreoffice", "--headless", "--convert-to", "pdf",
            "--outdir", "/tmp", input_path
        ]
        subprocess.run(cmd, check=True)
        
        pdf_filename = os.path.splitext(filename)[0] + ".pdf"
        output_path = f"/tmp/{pdf_filename}"
        
        with open(output_path, "rb") as f:
            pdf_bytes = f.read()
            
        # Удаляем временные файлы
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
        
        # Распознаем текст с помощью Tesseract OCR
        extracted_text = pytesseract.image_to_string(image, lang=lang)
        
        return {"success": True, "text": extracted_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR Error: {str(e)}")
