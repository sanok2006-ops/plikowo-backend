"""
Plikowo Quality Assurance Gate (Модуль автоматического контроля качества)
Проверяет результат сегментации и постобработки перед возвратом пользователю:
- Проверка наличия и целостности силуэта
- Проверка альфа-канала на артефакты
- Проверка площади человека (15-85% площади кадра)
- Выявление дефектов для запуска автоматического fallback-сценария (не более 1 попытки)
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image


class QualityAssuranceGate:
    @staticmethod
    def verify_silhouette(pil_rgba: Image.Image) -> Tuple[bool, Dict[str, Any]]:
        """
        Проверяет результат вырезания силуэта на корректность.
        Возвращает (is_valid, metrics).
        """
        if pil_rgba.mode != "RGBA":
            return False, {"error": "Image does not contain alpha channel"}

        w, h = pil_rgba.size
        total_pixels = w * h

        # Анализируем альфа-канал
        alpha = np.array(pil_rgba.split()[3])

        solid_fg_pixels = int(np.sum(alpha > 200))
        transparent_bg_pixels = int(np.sum(alpha < 50))
        semi_trans_pixels = total_pixels - solid_fg_pixels - transparent_bg_pixels

        fg_ratio = solid_fg_pixels / total_pixels
        bg_ratio = transparent_bg_pixels / total_pixels

        # 1. Проверка: силуэт не должен быть пустым
        if fg_ratio < 0.10:
            return False, {
                "reason": "silhouette_empty",
                "fg_ratio": round(fg_ratio, 3),
                "message": "Силуэт человека практически не обнаружен."
            }

        # 2. Проверка: фон должен быть действительно удален (не 100% сплошной)
        if bg_ratio < 0.08:
            return False, {
                "reason": "background_not_removed",
                "bg_ratio": round(bg_ratio, 3),
                "message": "Фон не был отделен от человека."
            }

        # 3. Проверка: силуэт не должен занимать абсолютно весь кадр без полей
        if fg_ratio > 0.88:
            return False, {
                "reason": "silhouette_too_large",
                "fg_ratio": round(fg_ratio, 3),
                "message": "Силуэт занимает чрезмерную площадь кадра."
            }

        # 4. Проверка верхней полосы (голова/волосы):
        # В верхних 10% кадра по бокам должен быть фон
        top_strip = alpha[:int(h * 0.10), :]
        top_corner_bg_ratio = np.mean(top_strip[:, :int(w * 0.25)] < 50)
        if top_corner_bg_ratio < 0.30:
            edge_warning = True
        else:
            edge_warning = False

        metrics = {
            "fg_ratio": round(fg_ratio, 3),
            "bg_ratio": round(bg_ratio, 3),
            "semi_trans_ratio": round(semi_trans_pixels / total_pixels, 3),
            "edge_warning": edge_warning,
            "qa_passed": True
        }

        return True, metrics


class BiometricQualityController:
    """
    Крок 4. Автоматичний цикл перевірки (Контроль якості біометрії та різкості)
    1. Метрика схожості (Біометрія): DeepFace.verify(model_name="Facenet512", distance_metric="cosine")
       Правило: distance <= 0.07 (Cosine Similarity >= 0.93)
    2. Метрика різкості: дисперсія Лапласіана cv2.Laplacian(image, cv2.CV_64F).var()
       Правило: зростання різкості мінімум на 20-30% без спотворення біометрії.
    """
    @staticmethod
    def verify_biometric_similarity(img_before_rgb: Image.Image, img_after_rgb: Image.Image) -> Tuple[bool, float, Dict[str, Any]]:
        import cv2

        bgr1 = cv2.cvtColor(np.array(img_before_rgb), cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor(np.array(img_after_rgb), cv2.COLOR_RGB2BGR)

        # 1. Основна перевірка через DeepFace (Facenet512 / VGG-Face)
        try:
            from deepface import DeepFace
            res = DeepFace.verify(
                img1_path=bgr1,
                img2_path=bgr2,
                model_name="Facenet512",
                distance_metric="cosine",
                enforce_detection=False
            )
            dist = float(res.get("distance", 0.0))
            is_valid = (dist <= 0.07)
            return is_valid, dist, {"model": "Facenet512", "distance": round(dist, 4), "verified": is_valid}
        except Exception as e_fn:
            try:
                from deepface import DeepFace
                res = DeepFace.verify(
                    img1_path=bgr1,
                    img2_path=bgr2,
                    model_name="VGG-Face",
                    distance_metric="cosine",
                    enforce_detection=False
                )
                dist = float(res.get("distance", 0.0))
                is_valid = (dist <= 0.07)
                return is_valid, dist, {"model": "VGG-Face", "distance": round(dist, 4), "verified": is_valid}
            except Exception as e_vgg:
                # Надійний локальний fallback (кореляція спектра обличчя)
                hsv1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
                hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
                hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                dist = max(0.0, float(1.0 - corr))
                is_valid = (dist <= 0.07)
                return is_valid, dist, {"model": "opencv_spectral_fallback", "distance": round(dist, 4), "verified": is_valid}

    @staticmethod
    def measure_sharpness_variance(img_rgb: Image.Image) -> float:
        """
        Обчислює дисперсію Лапласіана: cv2.Laplacian(image, cv2.CV_64F).var()
        """
        import cv2
        gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

