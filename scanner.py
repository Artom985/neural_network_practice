"""Сканирование каталога и преобразование файлов в единый массив."""

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .exceptions import DataFileError

logger = logging.getLogger(__name__)

def find_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Рекурсивно находит файлы с указанными расширениями.

    :param root: корневая директория
    :param extensions: кортеж расширений, напр. ('.jpg', '.png')
    :return: отсортированный список путей
    """
    if not root.exists():
        raise DataFileError(f"Директория не найдена: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    logger.info("Найдено %d файлов в %s", len(files), root)
    return sorted(files)

def load_image(path: Path, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Читает изображение через PIL, приводит к RGB и размеру size.

    Нормализация: значения пикселей делятся на 255 (диапазон [0, 1]).
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize(size)
            array = np.asarray(img, dtype=np.float32) / 255.0
    except OSError as exc:
        raise DataFileError(f"Не удалось прочитать {path}: {exc}") from exc
    return array

def prepare_dataset(root: Path, extensions: tuple[str, ...],
                    size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Готовит единый массив (N, H, W, C) из всех изображений каталога."""
    started = time.perf_counter()
    files = find_files(root, extensions)
    if not files:
        raise DataFileError("Не найдено ни одного файла с заданными расширениями.")

    arrays = [load_image(path, size) for path in files]
    dataset = np.stack(arrays, axis=0)
    elapsed = time.perf_counter() - started

    logger.info("Массив: %s | размер: %.2f МБ | время: %.2f с",
                dataset.shape, dataset.nbytes / 1e6, elapsed)
    return dataset
