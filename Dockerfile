FROM python:3.10-slim

# Установка системных зависимостей: LibreOffice, OCR (Tesseract) и системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    tesseract-ocr \
    tesseract-ocr-ukr \
    tesseract-ocr-rus \
    tesseract-ocr-pol \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
