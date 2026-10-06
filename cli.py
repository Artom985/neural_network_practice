"""CLI-утилита: команды prepare и doctor."""

import argparse
import logging
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import settings
from .logging_config import setup_logging
from .scanner import prepare_dataset

logger = logging.getLogger(__name__)

def cmd_prepare(args: argparse.Namespace) -> int:
    """Команда prepare: собрать массив из файлов и сохранить в .npy."""
    root = Path(args.path or settings.data_dir)
    extensions = tuple(e if e.startswith(".") else f".{e}"
                       for e in args.ext.split(","))
    dataset = prepare_dataset(root, extensions)
    output = Path(args.output or settings.resolve_output("dataset.npy"))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, dataset)
    print(f"Сохранено: {output} | shape={dataset.shape}")
    return 0

def _run_external(command: list[str]) -> str:
    """Запускает внешнюю команду через subprocess и возвращает её вывод."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip() or "(пусто)"
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"недоступно ({exc})"

def cmd_doctor(args: argparse.Namespace) -> int:
    """Команда doctor: проверка окружения."""
    print("=== Проверка окружения ===")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Рабочая директория: {Path.cwd()}")

    for module in ("numpy", "PIL", "dotenv", "torch"):
        try:
            __import__(module)
            print(f"Библиотека {module}: OK")
        except ImportError:
            print(f"Библиотека {module}: НЕ УСТАНОВЛЕНА")

    print(f"CUDA (nvidia-smi): {_run_external(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    """Собирает парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(prog="prep", description="Подготовка данных для нейросети")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="собрать массив из файлов")
    p_prepare.add_argument("--path", help="директория с данными")
    p_prepare.add_argument("--ext", default=".jpg,.png", help="расширения через запятую")
    p_prepare.add_argument("--output", help="выходной .npy файл")
    p_prepare.set_defaults(func=cmd_prepare)

    p_doctor = sub.add_parser("doctor", help="проверить окружение")
    p_doctor.set_defaults(func=cmd_doctor)
    return parser

def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(settings.log_level)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
