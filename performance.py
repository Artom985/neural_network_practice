"""Сравнение последовательной и многопроцессорной аугментации."""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .augment import augment
from .scanner import load_image

def run_sequential(files: list[Path]) -> float:
    """Измеряет время последовательной обработки."""
    started = time.perf_counter()
    for path in files:
        _ = augment(load_image(path))
    return time.perf_counter() - started

def run_parallel(files: list[Path], n_workers: int = 4) -> float:
    """Измеряет время многопроцессорной обработки."""
    started = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        pool.map(_augment_path, files)
    return time.perf_counter() - started

def _augment_path(path: Path) -> np.ndarray:
    """Вспомогательная функция для Pool.map (должна быть picklable)."""
    return augment(load_image(path))

def save_report(seq_time: float, par_time: float, n_files: int,
                output: str = "performance_report.json") -> None:
    """Сохраняет отчёт о производительности в JSON."""
    report = {
        "files_processed": n_files,
        "sequential_seconds": round(seq_time, 4),
        "parallel_seconds": round(par_time, 4),
        "speedup": round(seq_time / par_time, 2) if par_time > 0 else None,
    }
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
