"""Конфигурация: значения по умолчанию + переменные окружения."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # читает .env, если он есть

@dataclass
class Settings:
    """Настройки приложения без захардкоженных путей."""

    data_dir: str = os.getenv("DATA_DIR", "data/input")
    output_dir: str = os.getenv("OUTPUT_DIR", "data/output")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def resolve_output(self, filename: str) -> str:
        """Строит полный путь к выходному файлу."""
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, filename)

settings = Settings()
