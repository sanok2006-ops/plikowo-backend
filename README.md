# Plikowo Backend

Мікросервіс на FastAPI для Plikowo:
- **Обробка та вилучення фону (HuggingFace BiRefNet + ZeroGPU + Pillow)**: `/remove-bg`
  1. Салонна автокорекція кольору та експозиції (захист від блідості/пересвітів).
  2. Сегментація силуету за допомогою моделі BiRefNet на базі HuggingFace ZeroGPU.
  3. Салонна доводка краю волосся в Pillow без кукольних спотворень.
- **Біометричне фото MOS / Karta Pobytu**: `/biometric-mos-photo` (35x45 мм, 684x883 px)
- **Конвертація HEIC/HEVC**: `/convert-heic` (Pillow + pillow-heif)
- **Конвертація документів**: `/convert-doc`
- **Розпізнавання тексту (OCR)**: `/ocr`

## Налаштування на Render
1. В налаштуваннях сервісу на Render (Environment Variables):
   - `HF_TOKEN` = ваш HuggingFace токен доступу
   - `HF_SPACE_URL` = URL HuggingFace Space (наприклад, `https://sanyua1-plikowo-rmbg.hf.space`)
   - `GEMINI_API_KEY` = ключ Google Gemini для ШІ-помічника
