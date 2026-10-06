"""Консольный интерфейс для обучения и эксплуатации MLP."""

from pathlib import Path

import numpy as np

from .exceptions import NeuralNetError
from .manager import DatasetManager
from .models import NeuralNetwork

def _read_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Ошибка: введите целое число.")

def _read_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Ошибка: введите число.")

def create_network() -> NeuralNetwork:
    """Пункт меню 1: создание сети."""
    sizes = input("Размеры слоёв через запятую (напр. 4,8,1): ")
    layer_sizes = [int(s) for s in sizes.split(",") if s.strip()]
    activation = input("Активация (sigmoid/relu): ").strip() or "sigmoid"
    net = NeuralNetwork(layer_sizes, activation)
    print(f"Создана сеть {layer_sizes} на активации '{activation}'.")
    return net

def main() -> None:
    """Точка входа консольного приложения."""
    net: NeuralNetwork | None = None
    dataset = DatasetManager()

    while True:
        print("\n=== Меню ===")
        print("1 - Создать сеть")
        print("2 - Загрузить CSV")
        print("3 - Обучить сеть")
        print("4 - Предсказание")
        print("5 - Сохранить веса")
        print("6 - Загрузить веса")
        print("7 - Показать график ошибки")
        print("0 - Выход")
        choice = input("Ваш выбор: ").strip()

        try:
            if choice == "1":
                net = create_network()
            elif choice == "2":
                path = input("Путь к CSV: ").strip() or "data/input/data.csv"
                dataset.load_csv(path)
                mode = input("Нормализация (standard/minmax): ").strip() or "standard"
                dataset.normalize(mode)
                print(f"Загружено {dataset.features.shape[0]} строк.")
            elif choice == "3":
                if net is None:
                    raise NeuralNetError("Сначала создайте сеть (п. 1).")
                lr = _read_float("Скорость обучения: ")
                epochs = _read_int("Число эпох: ")
                batch = _read_int("Размер батча: ")
                x_tr, y_tr, x_te, y_te = dataset.split()
                net.train(x_tr, y_tr, lr, epochs, batch)
                pred = net.predict(x_te)
                print(f"Test MSE: {float(np.mean((pred - y_te) ** 2)):.6f}")
            elif choice == "4":
                if net is None:
                    raise NeuralNetError("Сначала создайте/загрузите сеть.")
                values = input("Признаки через запятую: ")
                x = np.array([[float(v) for v in values.split(",")]]).T
                print("Предсказание:", net.predict(x).ravel())
            elif choice == "5":
                if net is None:
                    raise NeuralNetError("Нет сети для сохранения.")
                net.save_weights(input("Файл для сохранения: ").strip())
                print("Веса сохранены.")
            elif choice == "6":
                net = NeuralNetwork.load_weights(input("Файл весов: ").strip())
                print("Веса загружены.")
            elif choice == "7":
                if net is None or not net.loss_history:
                    raise NeuralNetError("Нет истории ошибок.")
                import matplotlib.pyplot as plt
                plt.plot(net.loss_history)
                plt.xlabel("Эпоха")
                plt.ylabel("MSE")
                plt.grid(True)
                plt.show()
            elif choice == "0":
                print("Выход.")
                break
            else:
                print("Неизвестный пункт меню.")
        except (NeuralNetError, ValueError) as exc:
            print(f"[Ошибка] {exc}")

if __name__ == "__main__":
    main()
