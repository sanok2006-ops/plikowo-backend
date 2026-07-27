FROM python:3.10-slim

# Устанавливаем системные утилиты
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    tesseract-ocr \
    tesseract-ocr-ukr \
    tesseract-ocr-rus \
    tesseract-ocr-pol \
    tesseract-ocr-eng \
    libheif-examples \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка весов нейросети U2-Net
RUN python -c "from rembg import new_session; new_session('u2net')"

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
