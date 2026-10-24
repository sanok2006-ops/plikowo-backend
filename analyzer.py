"""
Plikowo Photo Analyzer (Модуль автоматического анализа качества фото)
Выполняет быстрый (~10-20 мс) локальный анализ входного фото средствами компьютерного зрения:
- Разрешение, пропорции и формат
- Резкость (Laplacian variance, Sobel gradients) по всему кадру и отдельно по лицу
- Экспозиция, динамический диапазон и контраст
- Баланс белого и цветовой сдвиг (LAB color cast detection)
- Цифровой шум и артефакты сжатия
- Обнаружение лиц и ключевых ориентиров (глаза, пропорции, наклон)
- Проверка биометрической пригодности (MOS / Karta Pobytu / ICAO)
- Определение оптимального сценария обработки (A, B, C, D, E, F)
"""

import io
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps


class Scenario(str, Enum):
    A_HIGH_QUALITY = "A_HIGH_QUALITY"      # Отличное качество: только удаление фона и легкий микроконтраст
    B_POOR_LIGHTING = "B_POOR_LIGHTING"    # Проблемы со светом: коррекция баланса белого и экспозиции перед сегментацией
    C_NOISE_OR_LOWRES = "C_NOISE_OR_LOWRES"# Шум / низкое разрешение: бережный апскейл Real-ESRGAN (без перерисовки лица)
    D_MILD_BLUR = "D_MILD_BLUR"            # Легкая нерезкость: адаптивное повышение резкости
    E_DISTRACTING_BG = "E_DISTRACTING_BG"  # Сложный фон: стандартная сегментация RMBG
    F_REJECT = "F_REJECT"                  # Непригодный исходник: отказ с рекомендацией загрузить лучшее фото


class PhotoAnalyzer:
    def __init__(self):
        # Загружаем каскады Хаара для детекции лица и глаз
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")

    def pil_to_cv2(self, pil_img: Image.Image) -> np.ndarray:
        """Конвертирует PIL Image в BGR NumPy массив OpenCV с учетом EXIF."""
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def analyze_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Анализирует изображение из байтов."""
        pil_img = Image.open(io.BytesIO(image_bytes))
        return self.analyze_pil(pil_img)

    def analyze_pil(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Основной метод анализа."""
        cv_img = self.pil_to_cv2(pil_img)
        h, w = cv_img.shape[:2]
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 1. Разрешение и геометрия
        total_pixels = w * h
        aspect_ratio = w / h if h > 0 else 1.0

        # 2. Резкость (Laplacian variance и Sobel)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mean = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

        # 3. Экспозиция и контраст
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        p5, p95 = np.percentile(gray, [5, 95])
        dynamic_range = float(p95 - p5)
        overexposed_ratio = float(np.mean(gray > 245))
        underexposed_ratio = float(np.mean(gray < 25))

        # 4. Баланс белого и цветовой сдвиг (LAB пространство)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        mean_l = float(np.mean(lab[:, :, 0]))
        mean_a = float(np.mean(lab[:, :, 1]))
        mean_b = float(np.mean(lab[:, :, 2]))

        # Нейтральный серый/белый в LAB: a=128, b=128
        color_cast_warm = mean_b - 128.0   # > 6 = желтоватый/теплый свет ламп
        color_cast_cool = 128.0 - mean_b   # > 6 = синеватый/холодный свет
        color_cast_tint = mean_a - 128.0   # > 6 = пурпурный, < -6 = зеленоватый

        # 5. Оценка шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_residual = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
        noise_level = float(np.mean(noise_residual))

        # 6. Детекция лица и глаз
        min_face_dim = max(60, int(min(w, h) * 0.10))
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_face_dim, min_face_dim)
        )

        face_count = len(detected_faces)
        face_info: Optional[Dict[str, Any]] = None
        rejection_reasons: List[Dict[str, str]] = []

        if face_count == 0:
            rejection_reasons.append({
                "code": "no_face_detected",
                "uk": "Обличчя не виявлено. Будь ласка, зробіть чітке портретне фото анфас.",
                "pl": "Nie wykryto twarzy. Proszę zrobić wyraźne zdjęcie portretowe na wprost.",
                "ru": "Лицо не обнаружено. Пожалуйста, сделайте четкое портретное фото анфас."
            })
        elif face_count > 1:
            rejection_reasons.append({
                "code": "multiple_faces",
                "uk": "На фото виявлено декілька людей. На фото для документів повинна бути лише одна особа.",
                "pl": "Wykryto kilka osób. Na zdjęciu do dokumentu może znajdować się tylko jedna osoba.",
                "ru": "На фото обнаружено несколько человек. На фото для документов должен быть только один человек."
            })
        else:
            # Одно лицо найдено — анализируем его детально
            fx, fy, fw, fh = detected_faces[0]
            face_crop_gray = gray[fy:fy+fh, fx:fx+fw]
            face_lap_var = float(cv2.Laplacian(face_crop_gray, cv2.CV_64F).var())
            face_rel_h = fh / h
            face_rel_w = fw / w

            # Проверяем глаза внутри области лица
            face_upper = face_crop_gray[:int(fh * 0.65), :]
            eyes = self.eye_cascade.detectMultiScale(face_upper, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            eye_count = len(eyes)

            # Проверяем наклон (если обнаружено ровно 2 глаза)
            roll_angle = 0.0
            if eye_count == 2:
                e1, e2 = eyes[0], eyes[1]
                # Сортируем слева направо
                if e1[0] > e2[0]:
                    e1, e2 = e2, e1
                dx = (e2[0] + e2[2]/2) - (e1[0] + e1[2]/2)
                dy = (e2[1] + e2[3]/2) - (e1[1] + e1[3]/2)
                if dx > 0:
                    roll_angle = float(np.degrees(np.arctan2(dy, dx)))

            # Проверяем обрезку краями кадра
            margin_x = min(fx, w - (fx + fw))
            margin_y = min(fy, h - (fy + fh))
            is_cut_off = (fx < w * 0.015) or ((fx + fw) > w * 0.985) or (fy < h * 0.015)

            if is_cut_off:
                rejection_reasons.append({
                    "code": "face_cut_off",
                    "uk": "Обличчя обрізане краями кадру. Потрібно більше простору навколо голови.",
                    "pl": "Twarz jest obcięta krawędziami kadru. Potrzeba więcej przestrzeni wokół głowy.",
                    "ru": "Лицо обрезано краями кадра. Нужно больше пространства вокруг головы."
                })

            if face_rel_h < 0.12:
                rejection_reasons.append({
                    "code": "face_too_small",
                    "uk": "Обличчя занадто далеко від камери. Зробіть фото ближче.",
                    "pl": "Twarz znajduje się za daleko od aparatu. Zrób zdjęcie z mniejszej odległości.",
                    "ru": "Лицо слишком далеко от камеры. Сделайте фото ближе."
                })

            if abs(roll_angle) > 18.0:
                rejection_reasons.append({
                    "code": "face_tilted",
                    "uk": "Голова занадто нахилена. Тримайте голову рівно та дивіться прямо в камеру.",
                    "pl": "Głowa jest zbyt pochylona. Proszę trzymać głowę prosto i patrzeć w obiektyw.",
                    "ru": "Голова слишком наклонена. Держите голову ровно и смотрите прямо в камеру."
                })

            face_info = {
                "box": (int(fx), int(fy), int(fw), int(fh)),
                "face_width_px": int(fw),
                "face_height_px": int(fh),
                "face_relative_height": float(face_rel_h),
                "face_sharpness_lap": float(face_lap_var),
                "eyes_detected": int(eye_count),
                "roll_angle_deg": float(roll_angle),
                "margin_x": int(margin_x),
                "margin_y": int(margin_y)
            }

        # Проверка на критическую размытость всего кадра
        if lap_var < 18.0:
            rejection_reasons.append({
                "code": "severe_blur",
                "uk": "Фото занадто розмите або не у фокусі. Зробіть чіткіший знімок при хорошому світлі.",
                "pl": "Zdjęcie jest zbyt rozmyte lub nieostre. Zrób wyraźniejsze zdjęcie przy dobrym świetle.",
                "ru": "Фото слишком размыто или не в фокусе. Сделайте более четкий снимок при хорошем свете."
            })

        # Проверка на критическую темноту
        if mean_brightness < 35.0:
            rejection_reasons.append({
                "code": "extreme_darkness",
                "uk": "Фото занадто темне. Неможливо розрізнити риси обличчя для біометрії.",
                "pl": "Zdjęcie jest zbyt ciemne. Nie można rozpoznać rysów twarzy do biometrii.",
                "ru": "Фото слишком темное. Невозможно различить черты лица для биометрии."
            })

        # Определение статуса пригодности
        biometric_valid = (len(rejection_reasons) == 0)

        # 7. Выбор оптимального сценария обработки (Router Logic)
        if not biometric_valid:
            recommended_scenario = Scenario.F_REJECT
        else:
            # Оцениваем потребность в световой коррекции
            needs_light_fix = (
                mean_brightness < 90.0 or mean_brightness > 190.0
                or abs(color_cast_warm) > 10.0 or abs(color_cast_cool) > 10.0
                or abs(color_cast_tint) > 10.0 or dynamic_range < 110.0
            )

            # Оцениваем потребность в апскейле (MOS стандарт 684x883 px)
            fw = face_info["face_width_px"] if face_info else 0
            needs_upscale = (w < 600 or h < 750 or fw < 160)

            # Оцениваем потребность в оптической резкости (без генеративных кукол)
            face_lap = face_info["face_sharpness_lap"] if face_info else lap_var
            needs_sharpening = (face_lap < 90.0 or lap_var < 100.0)

            if needs_light_fix and not needs_upscale:
                recommended_scenario = Scenario.B_POOR_LIGHTING
            elif needs_upscale:
                recommended_scenario = Scenario.C_NOISE_OR_LOWRES
            elif needs_sharpening:
                recommended_scenario = Scenario.D_MILD_BLUR
            else:
                recommended_scenario = Scenario.A_HIGH_QUALITY

        return {
            "resolution": {"width": w, "height": h, "aspect_ratio": round(aspect_ratio, 3), "megapixels": round(total_pixels / 1e6, 2)},
            "sharpness": {"laplacian_variance": round(lap_var, 1), "sobel_mean": round(sobel_mean, 1)},
            "exposure": {
                "mean_brightness": round(mean_brightness, 1),
                "contrast_std": round(contrast_std, 1),
                "dynamic_range": round(dynamic_range, 1),
                "overexposed_ratio": round(overexposed_ratio, 3),
                "underexposed_ratio": round(underexposed_ratio, 3)
            },
            "color_cast": {
                "mean_l": round(mean_l, 1),
                "mean_a": round(mean_a, 1),
                "mean_b": round(mean_b, 1),
                "warm_shift": round(color_cast_warm, 1),
                "cool_shift": round(color_cast_cool, 1),
                "tint_shift": round(color_cast_tint, 1)
            },
            "noise_level": round(noise_level, 2),
            "face": face_info,
            "face_count": face_count,
            "biometric_valid": biometric_valid,
            "rejection_reasons": rejection_reasons,
            "recommended_scenario": recommended_scenario.value
        }
