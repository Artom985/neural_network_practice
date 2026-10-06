"""Аугментации изображений (выполняются в отдельных процессах)."""

import numpy as np

def random_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Случайное горизонтальное отражение."""
    return np.fliplr(image) if rng.random() < 0.5 else image

def random_rotation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Случайный поворот на 90/180/270 градусов."""
    k = int(rng.integers(0, 4))
    return np.rot90(image, k)

def add_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Добавление гауссова шума."""
    noise = rng.normal(0.0, 0.02, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)

def augment(image: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Применяет полный конвейер аугментаций к одному изображению."""
    rng = np.random.default_rng(seed)
    result = random_flip(image, rng)
    result = random_rotation(result, rng)
    return add_noise(result, rng)
