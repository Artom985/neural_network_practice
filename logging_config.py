"""Настройка логирования с записью в файл и консоль."""

import logging
from pathlib import Path

def setup_logging(level: str = "INFO") -> None:
    """Инициализирует корневой логгер.

    :param level: строковый уровень логирования (DEBUG/INFO/WARNING)
    """
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log", encoding="utf-8"),
        ],
    )
