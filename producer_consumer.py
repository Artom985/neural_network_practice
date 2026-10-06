"""Producer-consumer: потоки читают файлы, процессы аугментируют."""

import logging
import queue
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .augment import augment
from .scanner import load_image

logger = logging.getLogger(__name__)

_SENTINEL = None

class ParallelAugmenter:
    """Организует многопоточное чтение и многопроцессорную аугментацию."""

    def __init__(self, n_readers: int = 4, n_workers: int = 4) -> None:
        self.n_readers = n_readers
        self.n_workers = n_workers
        self.path_queue: queue.Queue = queue.Queue()
        self.array_queue: queue.Queue = queue.Queue(maxsize=64)
        self.results: list[np.ndarray] = []
        self.counter = 0
        self.lock = threading.Lock()

    # ---------- producer ----------
    def _producer(self, files: list[Path]) -> None:
        """Сканирует список файлов и кладёт пути в очередь."""
        for path in files:
            self.path_queue.put(path)
        for _ in range(self.n_readers):
            self.path_queue.put(_SENTINEL)

    # ---------- consumers (потоки) ----------
    def _reader(self) -> None:
        """Читает файлы из path_queue и кладёт массивы в array_queue."""
        while True:
            path = self.path_queue.get()
            if path is _SENTINEL:
                self.path_queue.task_done()
                break
            try:
                array = load_image(path)
                self.array_queue.put(array)
            except Exception as exc:  # логируем и продолжаем
                logger.error("Ошибка чтения %s: %s", path, exc)
            finally:
                self.path_queue.task_done()

    # ---------- процессная аугментация ----------
    def _collect(self, augmented: np.ndarray) -> None:
        """Callback: сохраняет результат и обновляет счётчик потокобезопасно."""
        with self.lock:
            self.results.append(augmented)
            self.counter += 1

    def run(self, files: list[Path]) -> list[np.ndarray]:
        """Запускает полный конвейер и корректно завершает все потоки/процессы."""
        started = time.perf_counter()
        producer = threading.Thread(target=self._producer, args=(files,), daemon=True)
        readers = [threading.Thread(target=self._reader, daemon=True)
                   for _ in range(self.n_readers)]
        producer.start()
        for reader in readers:
            reader.start()

        with Pool(processes=self.n_workers) as pool:
            while True:
                try:
                    array = self.array_queue.get(timeout=0.5)
                except queue.Empty:
                    if all(not r.is_alive() for r in readers):
                        break
                    continue
                pool.apply_async(augment, (array,), callback=self._collect)

        producer.join(timeout=5)
        for reader in readers:
            reader.join(timeout=5)

        elapsed = time.perf_counter() - started
        logger.info("Обработано %d файлов за %.2f с", self.counter, elapsed)
        return self.results
