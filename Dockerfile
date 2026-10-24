FROM python:3.10-slim

# Установка системных зависимостей: LibreOffice и OCR (Tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    tesseract-ocr \
    tesseract-ocr-ukr \
    tesseract-ocr-rus \
    tesseract-ocr-pol \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
