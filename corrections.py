"""
Plikowo Corrections (Модуль салонной локальной коррекции по стандарту MOS Foto)
Реализует стабильный профессиональный алгоритм, аналогичный mosfoto.com:
1. Защита естественного тона кожи (Skin-Preserving Color Tone) — сохраняет естественное тепло лица на чисто белом фоне (без бледности и без красноты).
2. Студийная глубина освещения (Studio Depth) — глубокий чистый черный цвет в зрачках и ресницах (без серого мыла).
3. Бережное подавление шума сенсора (Bilateral Filter).
4. Устранение каймы на полупрозрачных волосках (Alpha Defringing) для безупречного белого фона.
5. Натуральная оптика: естественный контраст (+3%), мягкий румянец (+4%) и деликатная резкость взгляда (radius=0.8, 35%).
   Никакого перешарпа (120%), никакого пластика и никаких кукольных стеклянных глаз!
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def skin_preserving_color_grading(bgr_img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Интеллектуальная салонная цветокоррекция с защитой тона кожи:
    - Детектирует область кожи в YCrCb (133 <= Cr <= 173, 77 <= Cb <= 127).
    - Защищает гемоглобин и естественный теплый тон (R - B).
    - Предотвращает эффект 'бледного лица' на чисто белом фоне.
    """
    img_float = bgr_img.astype(np.float32)
    h, w = bgr_img.shape[:2]

    # Анализ кожи в YCrCb
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    skin_count = np.sum(skin_mask)

    if skin_count > (w * h * 0.03):
        skin_r = float(np.mean(img_float[:, :, 2][skin_mask]))
        skin_b = float(np.mean(img_float[:, :, 0][skin_mask]))
        warmth = skin_r - skin_b

        # Если кожа слишком холодная / синеватая (warmth < 28):
        # Деликатно подогреваем тон лица (+4% красного, -4% синего)
        if warmth < 28.0:
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 1.04, 0, 255)
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 0.96, 0, 255)
        # Если свет лампы дал чрезмерную желтизну (warmth > 75):
        elif warmth > 75.0:
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 0.97, 0, 255)
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 1.03, 0, 255)

    return np.clip(img_float, 0, 255).astype(np.uint8)


# Обратная совместимость для существующих вызовов
correct_white_balance = skin_preserving_color_grading


def studio_lighting_enhancement(bgr_img: np.ndarray, clip_limit: float = 1.5) -> np.ndarray:
    """
    Студийная доводка освещения и динамического диапазона:
    - Углубляет черный цвет в зрачках, ресницах и волосах (устраняет серость и мыльность).
    - Мягко нормализует света и тени без пересвечивания кожи.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)

    # Вычисляем темную и светлую опорные точки
    p05 = np.percentile(l_float, 0.5)
    p995 = np.percentile(l_float, 99.5)

    if p995 > p05 + 40:
        l_norm = np.clip((l_float - p05) / (p995 - p05) * 255.0, 0, 255)
        # Деликатное смешивание: 30% нормализации + 70% оригинала (исключает резкость)
        l_out = (0.30 * l_norm + 0.70 * l_float).astype(np.uint8)
    else:
        l_out = l

    lab_corrected = cv2.merge([l_out, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


# Обратная совместимость
adaptive_light_correction = studio_lighting_enhancement


def denoise_skin_preserving(bgr_img: np.ndarray) -> np.ndarray:
    """
    Деликатное подавление шума сенсора без замыливания текстуры пор кожи и ресниц.
    """
    return cv2.bilateralFilter(bgr_img, d=5, sigmaColor=18, sigmaSpace=18)


def defringe_alpha_edges(pil_rgba: Image.Image) -> Image.Image:
    """
    Устраняет темную или цветную кайму на полупрозрачных волосках
    по границе альфа-канала при наложении на чисто белый фон.
    """
    if pil_rgba.mode != "RGBA":
        return pil_rgba

    np_img = np.array(pil_rgba).copy()
    alpha = np_img[:, :, 3]
    edge_mask = (alpha > 5) & (alpha < 240)

    if np.sum(edge_mask) > 0:
        weight = (255.0 - alpha[edge_mask].astype(np.float32)) / 255.0
        for c in range(3):
            np_img[:, :, c][edge_mask] = np.clip(
                np_img[:, :, c][edge_mask].astype(np.float32) + 25.0 * weight,
                0, 255
            ).astype(np.uint8)

    return Image.fromarray(np_img)


def apply_studio_postprocessing(pil_rgba: Image.Image) -> Image.Image:
    """
    Салонная фото-доводка силуэта перед наложением на чисто белый фон (стандарт Mosfoto):
    1. Насыщенность (+4%): естественный живой оттенок кожи на белом фоне (без бледности и без перенасыщения).
    2. Мягкий контраст (+3%): естественная глубина теней без жесткости.
    3. Деликатная резкость (radius=0.8, 35%, threshold=2):
       Четкий живой взгляд, ресницы и волоски БЕЗ перешарпа (120%), БЕЗ зернистости и БЕЗ стеклянных глаз!
    """
    if pil_rgba.mode != "RGBA":
        pil_rgba = pil_rgba.convert("RGBA")

    # Устраняем краевую кайму на волосах
    cleaned_rgba = defringe_alpha_edges(pil_rgba)

    r, g, b, a = cleaned_rgba.split()
    rgb = Image.merge("RGB", (r, g, b))

    # 1. Цвет / насыщенность (защита от бледности)
    vibrant = ImageEnhance.Color(rgb).enhance(1.04)

    # 2. Мягкий естественный контраст
    contrasted = ImageEnhance.Contrast(vibrant).enhance(1.03)

    # 3. Деликатная салонная резкость (ясный живой взгляд без жесткости)
    final_rgb = contrasted.filter(ImageFilter.UnsharpMask(radius=0.8, percent=35, threshold=2))

    r_out, g_out, b_out = final_rgb.split()
    return Image.merge("RGBA", (r_out, g_out, b_out, a))
