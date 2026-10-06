"""Демонстрация producer-consumer и сравнение производительности."""

import logging
from pathlib import Path

from part2_prepare.app.logging_config import setup_logging
from part3_parallel.app.scanner import find_files     # ← если scanner общий
from part3_parallel.app.performance import run_sequential, run_parallel, save_report
from part3_parallel.app.producer_consumer import ParallelAugmenter

setup_logging("INFO")

files = find_files(Path("data/input"), (".png", ".jpg"))
print(f"Найдено файлов: {len(files)}\n")

# --- производительность ---
seq = run_sequential(files)
par = run_parallel(files, n_workers=4)
print(f"Последовательно: {seq:.2f} с")
print(f"4 процесса:      {par:.2f} с")
print(f"Ускорение:       {seq/par:.2f}x\n")

save_report(seq, par, len(files))
