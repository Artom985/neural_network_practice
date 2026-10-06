"""Загрузка, нормализация и разбиение датасета."""

from pathlib import Path

import numpy as np

from .exceptions import DataFileError, MismatchedDataError

class DatasetManager:
    """Управляет CSV-датасетом: загрузка, нормализация, split."""

    def __init__(self) -> None:
        self.features: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self.feature_names: list[str] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def load_csv(self, path: str | Path) -> None:
        """Читает CSV (последний столбец — целевая переменная)."""
        csv_path = Path(path)
        if not csv_path.exists():
            raise DataFileError(f"Файл не найден: {csv_path}")
        try:
            raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                                dtype=float, encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise DataFileError(f"Ошибка чтения CSV: {exc}") from exc

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        self.features = raw[:, :-1]
        self.targets = raw[:, -1:]

    def normalize(self, mode: str = "standard") -> None:
        """Нормализует признаки: 'standard' или 'minmax'."""
        if self.features is None:
            raise MismatchedDataError("Сначала загрузите данные.")
        if mode == "standard":
            self._mean = self.features.mean(axis=0)
            std = self.features.std(axis=0)
            std[std == 0] = 1.0
            self._std = std
            self.features = (self.features - self._mean) / self._std
        elif mode == "minmax":
            lo, hi = self.features.min(axis=0), self.features.max(axis=0)
            span = np.where(hi - lo == 0, 1.0, hi - lo)
            self.features = (self.features - lo) / span
        else:
            raise MismatchedDataError(f"Неизвестный режим нормализации: {mode}")

    def split(self, test_ratio: float = 0.2, seed: int = 42):
        """Делит выборку на train/test в указанной пропорции."""
        if not 0.0 < test_ratio < 1.0:
            raise MismatchedDataError("test_ratio должен быть в (0, 1).")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.features.shape[0])
        cut = int(len(indices) * (1 - test_ratio))
        train_idx, test_idx = indices[:cut], indices[cut:]
        # Транспонируем: сеть ожидает (n_features, n_samples)
        x_train = self.features[train_idx].T
        y_train = self.targets[train_idx].T
        x_test = self.features[test_idx].T
        y_test = self.targets[test_idx].T
        return x_train, y_train, x_test, y_test
