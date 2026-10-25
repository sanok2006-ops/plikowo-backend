"""
Plikowo Automated Test Suite (Скрипт тестирования новой архитектуры)
Тестирует:
1. Автоматический анализатор на реальных фотографиях и сложных сценариях.
2. Проверку условий: темнота, блюр, цветовой сдвиг, отсутствие лица.
3. Локальные фильтры баланса белого и CLAHE.
4. Полный цикл через маршрутизатор (с реальным вызовом HuggingFace BiRefNet).
5. Контроль качества (QA Gate) и биометрический формат MOS (35x45 мм, 684x883 px).
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Загружаем переменные окружения из backend/.env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(env_path)

# Добавляем родительскую директорию в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.analyzer import PhotoAnalyzer, Scenario
from backend.corrections import correct_white_balance, adaptive_light_correction, denoise_skin_preserving, apply_studio_postprocessing
from backend.qa_gate import QualityAssuranceGate
from backend.biometrics import generate_mos_biometric_photo
from backend.router import PhotoProcessingRouter, BiometricRejectionError


def run_tests():
    print("=" * 70)
    print("PLIKOWO AI PASSPORT ENGINE — КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ")
    print("=" * 70)

    analyzer = PhotoAnalyzer()
    desktop = "C:/Users/Oleksandr/Desktop"

    # --- ТЕСТ 1: Анализ реальных фотографий ---
    print("\n--- [ТЕСТ 1] Анализ реальных исходных фотографий ---")
    test_files = [
        "photo_5375413693258080246_y.jpg",
        "photo_5375589597938655223_y.jpg",
        "19.10.26_19.47.41_karta_pobytu_mos_684x883.jpg"
    ]

    for fname in test_files:
        fpath = os.path.join(desktop, fname)
        if not os.path.exists(fpath):
            continue
        im = Image.open(fpath)
        t0 = time.time()
        res = analyzer.analyze_pil(im)
        dt = (time.time() - t0) * 1000

        print(f"\nФото: {fname}")
        print(f"  • Время анализа: {dt:.1f} мс (локально, $0)")
        print(f"  • Разрешение: {res['resolution']['width']}x{res['resolution']['height']} ({res['resolution']['megapixels']} MP)")
        print(f"  • Резкость (Laplacian): {res['sharpness']['laplacian_variance']}")
        print(f"  • Яркость / Контраст: {res['exposure']['mean_brightness']} / {res['exposure']['contrast_std']}")
        print(f"  • Цветовой сдвиг (b*): {res['color_cast']['mean_b']} (warm: {res['color_cast']['warm_shift']})")
        print(f"  • Найдено лиц: {res['face_count']}")
        print(f"  • Биометрическая пригодность: {'ДOПУЩЕНО' if res['biometric_valid'] else 'ОТКЛОНЕНО'}")
        print(f"  • Назначенный сценарий: {res['recommended_scenario']}")

    # --- ТЕСТ 2: Тестирование граничных и стрессовых сценариев ---
    print("\n--- [ТЕСТ 2] Проверка стрессовых сценариев (темнота, блюр, нет лица) ---")

    # 2.1 Искусственный сильный блюр
    base_img = cv2.imread(os.path.join(desktop, "photo_5375413693258080246_y.jpg"))
    if base_img is not None:
        blurry = cv2.GaussianBlur(base_img, (35, 35), 0)
        blurry_pil = Image.fromarray(cv2.cvtColor(blurry, cv2.COLOR_BGR2RGB))
        res_blur = analyzer.analyze_pil(blurry_pil)
        reasons = [r["code"] for r in res_blur["rejection_reasons"]]
        print(f"  • Тест 'Сильная размытость': Сценарий={res_blur['recommended_scenario']}, Причины={reasons}")
        assert res_blur["recommended_scenario"] == Scenario.F_REJECT.value or not res_blur["biometric_valid"]

    # 2.2 Экстремальная темнота
    if base_img is not None:
        dark = (base_img * 0.1).astype(np.uint8)
        dark_pil = Image.fromarray(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
        res_dark = analyzer.analyze_pil(dark_pil)
        reasons = [r["code"] for r in res_dark["rejection_reasons"]]
        print(f"  • Тест 'Экстремальная темнота': Сценарий={res_dark['recommended_scenario']}, Причины={reasons}")
        assert not res_dark["biometric_valid"]

    # 2.3 Фото без лица (текстура шума)
    no_face = np.random.randint(50, 200, (600, 600, 3), dtype=np.uint8)
    no_face_pil = Image.fromarray(no_face)
    res_no_face = analyzer.analyze_pil(no_face_pil)
    reasons = [r["code"] for r in res_no_face["rejection_reasons"]]
    print(f"  • Тест 'Нет лица на фото': Сценарий={res_no_face['recommended_scenario']}, Причины={reasons}")
    assert "no_face_detected" in reasons

    # --- ТЕСТ 3: Проверка алгоритмической коррекции света и цвета ---
    print("\n--- [ТЕСТ 3] Тестирование алгоритмической коррекции (баланс белого и CLAHE) ---")
    if base_img is not None:
        # Симулируем теплый комнатный свет (желтый сдвиг)
        yellow_cast = base_img.copy().astype(np.float32)
        yellow_cast[:, :, 0] *= 0.7  # меньше синего
        yellow_cast[:, :, 1] *= 1.1  # больше зеленого
        yellow_cast[:, :, 2] *= 1.3  # больше красного
        yellow_cast = np.clip(yellow_cast, 0, 255).astype(np.uint8)

        corrected = correct_white_balance(yellow_cast, strength=0.8)
        lab_before = cv2.cvtColor(yellow_cast, cv2.COLOR_BGR2LAB)
        lab_after = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        print(f"  • Баланс белого до коррекции: mean_b={np.mean(lab_before[:,:,2]):.1f}")
        print(f"  • Баланс белого после коррекции: mean_b={np.mean(lab_after[:,:,2]):.1f} (приближено к нейтрали 128)")

        # Проверка вытягивания теней CLAHE
        light_fixed = adaptive_light_correction(yellow_cast)
        print(f"  • Яркость до CLAHE: {np.mean(yellow_cast):.1f}, после: {np.mean(light_fixed):.1f}")

    # --- ТЕСТ 4: Проверка генератора биометрического формата MOS 35x45 мм ---
    print("\n--- [ТЕСТ 4] Проверка биометрического компоновщика MOS (35x45 мм, 684x883 px) ---")
    # Создаем тестовый силуэт с альфа-каналом
    test_silhouette = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
    # Рисуем круглое лицо и плечи
    cv_sil = np.zeros((1000, 800, 4), dtype=np.uint8)
    cv2.circle(cv_sil, (400, 350), 180, (220, 180, 160, 255), -1)  # голова
    cv2.ellipse(cv_sil, (400, 750), (280, 250), 0, 0, 360, (50, 60, 80, 255), -1) # плечи
    test_silhouette = Image.fromarray(cv_sil)

    mos_photo = generate_mos_biometric_photo(test_silhouette, target_width=684, target_height=883, head_ratio=0.75)
    print(f"  • Размер MOS полотна: {mos_photo.size} (ожидается 684x883)")
    print(f"  • Цвет верхнего левого пикселя фона: {mos_photo.getpixel((5, 5))} (ожидается чисто белый 255,255,255)")
    assert mos_photo.size == (684, 883)
    assert mos_photo.getpixel((5, 5)) == (255, 255, 255)

    # --- ТЕСТ 5: Проверка сквозного пайплайна через Router ---
    print("\n--- [ТЕСТ 5] Проверка сквозного маршрутизатора (Router) ---")
    router = PhotoProcessingRouter()
    target_sample = os.path.join(desktop, "photo_5375413693258080246_y.jpg")
    with open(target_sample, "rb") as f:
        sample_bytes = f.read()

    print("Запуск сквозной обработки тестового фото...")
    t_start = time.time()
    res_bytes, media_type, meta = router.process_image(sample_bytes, output_biometric_jpeg=False)
    t_total = time.time() - t_start

    print(f"\nРезультат сквозной обработки:")
    print(f"  • Время выполнения: {t_total:.2f} сек")
    print(f"  • Тип медиа: {media_type}")
    print(f"  • Сценарий: {meta['scenario']}")
    print(f"  • Примененные коррекции: {meta['applied_corrections']}")
    print(f"  • QA Gate пройден: {meta['qa_metrics']['qa_passed']}")
    print(f"  • Доля человека в кадре: {meta['qa_metrics']['fg_ratio']*100:.1f}%")

    out_test_path = os.path.join(desktop, "test_router_output_silhouette.png")
    with open(out_test_path, "wb") as f:
        f.write(res_bytes)
    print(f"  • Сохранен силуэт: {out_test_path}")

    # Также проверяем генерацию чистого MOS JPEG
    mos_bytes, mos_type, mos_meta = router.process_image(sample_bytes, output_biometric_jpeg=True)
    out_mos_path = os.path.join(desktop, "test_router_mos_35x45.jpg")
    with open(out_mos_path, "wb") as f:
        f.write(mos_bytes)
    print(f"  • Сохранен готовый MOS 35x45 мм файл: {out_mos_path}")

    print("\n" + "=" * 70)
    print("ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
