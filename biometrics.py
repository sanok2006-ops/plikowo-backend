"""
Plikowo Biometrics (Модуль биометрических стандартов MOS / Karta Pobytu / ICAO)
Реализует официальные требования польского портала MOS (Moduł Obsługi Spraw)
и Urząd Wojewódzki для документов Karta Pobytu, MOS, Dowód Osobisty:
- Пропорции 35 x 45 мм (базовое цифровое разрешение 684 x 883 px, 500 DPI)
- Чисто белый равномерный фон (RGB 255, 255, 255)
- Размер головы (от подбородка до макушки): 70–80% высоты фотографии (31,5–36 мм)
- Глаза расположены на биометрической горизонтальной оси
- Свободное поле над макушкой головы (ок. 7-10% высоты кадра)
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image


def generate_mos_biometric_photo(
    silhouette_rgba: Image.Image,
    target_width: int = 684,
    target_height: int = 883,
    head_ratio: float = 0.75,
    face_box: Optional[Tuple[int, int, int, int]] = None
) -> Image.Image:
    """
    Накладывает силуэт на чисто белый фон (RGB 255, 255, 255)
    и кадрирует под официальный стандарт MOS (35x45 мм, 684x883 px).
    """
    if silhouette_rgba.mode != "RGBA":
        silhouette_rgba = silhouette_rgba.convert("RGBA")

    w, h = silhouette_rgba.size
    alpha = np.array(silhouette_rgba.split()[3])

    # 1. Находим границы силуэта человека по альфа-каналу
    y_indices, x_indices = np.where(alpha > 75)
    if len(y_indices) == 0:
        # Резервный случай: белый фон и центрированное масштабирование
        out = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        resized = silhouette_rgba.convert("RGB").resize((target_width, target_height), Image.Resampling.LANCZOS)
        out.paste(resized, (0, 0))
        return out

    top_hair_y = int(np.min(y_indices))
    bottom_y = int(np.max(y_indices))
    min_x = int(np.min(x_indices))
    max_x = int(np.max(x_indices))
    center_x = (min_x + max_x) // 2

    # 2. Оцениваем биометрический размер головы
    # Если переданы координаты лица от анализатора:
    if face_box:
        fx, fy, fw, fh = face_box
        chin_y = fy + fh
        estimated_head_h = max(int(fh * 1.25), chin_y - top_hair_y)
    else:
        # Стандартная анатомическая пропорция силуэта портрета: голова ~ 35-45% от верхней границы до плеч
        total_visible_h = bottom_y - top_hair_y
        estimated_head_h = int(total_visible_h * 0.55)

    estimated_head_h = max(estimated_head_h, int(h * 0.25))

    # Целевая высота головы: 75% от 883 px = 662 px
    target_head_px = int(target_height * head_ratio)
    scale = target_head_px / float(estimated_head_h)

    # Ограничиваем масштаб разумными пределами
    scale = np.clip(scale, 0.45, 2.5)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    scaled_silhouette = silhouette_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 3. Вычисляем позицию на полотне 684x883 px
    # Макушка головы должна быть на расстоянии ~5.5% от верхнего края (ок. 45-50 px)
    top_margin_px = int(target_height * 0.055)
    scaled_top_hair = int(top_hair_y * scale)
    paste_y = top_margin_px - scaled_top_hair

    # Центрируем лицо по горизонтали
    scaled_center_x = int(center_x * scale)
    paste_x = (target_width // 2) - scaled_center_x

    # 4. Создаем чисто белый фон (RGB 255, 255, 255)
    canvas = Image.new("RGBA", (target_width, target_height), (255, 255, 255, 255))
    canvas.paste(scaled_silhouette, (paste_x, paste_y), mask=scaled_silhouette)

    return canvas.convert("RGB")


def crop_and_resize_to_biometric_standard(
    img_bgr_or_pil,
    target_w: int = 684,
    target_h: int = 883
) -> Image.Image:
    """
    Крок 1. Валідація, кадрування та ресайз (Локально)
    1. Пошук обличчя через OpenCV Haar Cascades
    2. Кадрування: обличчя займає 70-80% кадру (макушка по верхній лінії 7%, підборіддя по нижній 87%)
    3. Зіниці / центр обличчя центровані по горизонталі
    4. Приведення до цільової роздільної здатності (684x883 px, пропорція 7:9)
    """
    import cv2

    if isinstance(img_bgr_or_pil, Image.Image):
        img_bgr = cv2.cvtColor(np.array(img_bgr_or_pil), cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_bgr_or_pil

    h, w = img_bgr.shape[:2]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_bgr, 1.1, 4)

    if len(faces) == 0:
        # Fallback на центральне кадрування з пропорцією 7:9
        target_ratio = target_w / target_h
        cur_ratio = w / h
        if cur_ratio > target_ratio:
            crop_w = int(h * target_ratio)
            crop_h = h
            x1 = (w - crop_w) // 2
            y1 = 0
        else:
            crop_w = w
            crop_h = int(w / target_ratio)
            x1 = 0
            y1 = (h - crop_h) // 2
        cropped = img_bgr[y1:y1+crop_h, x1:x1+crop_w]
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    # Використовуємо головне обличчя
    fx, fy, fw, fh = faces[0]

    # Спроба знайти зіниці для точного центрування
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    roi_gray = cv2.cvtColor(img_bgr[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.15, 3)

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        pupil_x = fx + (eyes[0][0] + eyes[0][2]//2 + eyes[-1][0] + eyes[-1][2]//2) // 2
    else:
        pupil_x = fx + fw // 2

    # Анатомічні межі голови
    chin_y = fy + int(fh * 1.02)
    crown_y = fy - int(fh * 0.40)
    head_h = chin_y - crown_y

    # Стандарт ліній: макушка на 7%, підборіддя на 87% (голови ~80% висоти кадру)
    target_head_ratio = 0.80
    crop_h = int(head_h / target_head_ratio)
    crop_w = int(crop_h * (target_w / target_h))

    crop_x1 = pupil_x - crop_w // 2
    crop_y1 = int(crown_y - 0.07 * crop_h)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    pad_top = max(0, -crop_y1)
    pad_bottom = max(0, crop_y2 - h)
    pad_left = max(0, -crop_x1)
    pad_right = max(0, crop_x2 - w)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        padded = cv2.copyMakeBorder(img_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        crop_y1 += pad_top
        crop_y2 += pad_top
        crop_x1 += pad_left
        crop_x2 += pad_left
        cropped = padded[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        cropped = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

