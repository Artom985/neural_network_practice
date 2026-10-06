"""Полносвязная нейронная сеть с настраиваемой активацией."""

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .exceptions import InvalidLayerSizeError, UnknownActivationError, WeightsIOError

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Сигмоида, устойчивая к переполнению."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Производная сигмоиды, выраженная через саму функцию."""
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU."""
    return np.maximum(0.0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Производная ReLU."""
    return (x > 0).astype(float)

ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}

class NeuralNetwork:
    """Полносвязная сеть прямого распространения.

    :param layer_sizes: список размеров слоёв, напр. [4, 8, 3]
    :param activation: имя функции активации ('sigmoid' или 'relu')
    :param seed: зерно генератора случайных чисел
    """

    def __init__(self, layer_sizes: list[int], activation: str = "sigmoid",
                 seed: int = 42) -> None:
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError("Нужно минимум 2 слоя (вход и выход).")
        if any(size <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоёв должны быть > 0.")
        if activation not in ACTIVATIONS:
            raise UnknownActivationError(f"Неизвестная активация: {activation}")

        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.activation, self.activation_derivative = ACTIVATIONS[activation]
        self.loss_history: list[float] = []

        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Инициализация Xavier — стабильнее при глубоких сетях
            scale = np.sqrt(2.0 / (in_size + out_size))
            self.weights.append(rng.normal(0.0, scale, size=(out_size, in_size)))
            self.biases.append(np.zeros((out_size, 1)))

    # ---------- прямой проход ----------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямое распространение.

        :param x: массив (n_features, n_samples)
        :return: выход сети той же формы по строкам
        """
        if x.shape[0] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Ожидалось {self.layer_sizes[0]} признаков, получено {x.shape[0]}."
            )
        self._pre_activations = []
        self._activations = [x]
        a = x
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            self._pre_activations.append(z)
            a = self.activation(z)
            self._activations.append(a)
        return a

    # ---------- обратный проход ----------
    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        """Накопление градиентов (вызывается после forward)."""
        n_samples = x.shape[1]
        delta = (self._activations[-1] - y) * self.activation_derivative(
            self._pre_activations[-1]
        )
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)

        for layer in reversed(range(len(self.weights))):
            grad_w[layer] = delta @ self._activations[layer].T / n_samples
            grad_b[layer] = np.sum(delta, axis=1, keepdims=True) / n_samples
            if layer > 0:
                delta = (self.weights[layer].T @ delta) * \
                    self.activation_derivative(self._pre_activations[layer - 1])

        self._grad_w, self._grad_b = grad_w, grad_b

    # ---------- обучение ----------
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              learning_rate: float = 0.1, epochs: int = 100,
              batch_size: int = 16) -> list[float]:
        """Обучение мини-батчами с градиентным спуском."""
        if x_train.shape[1] != y_train.shape[1]:
            raise MismatchedDataError("Число примеров в X и y не совпадает.")
        n_samples = x_train.shape[1]
        self.loss_history = []

        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            x_shuffled, y_shuffled = x_train[:, perm], y_train[:, perm]
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                xb = x_shuffled[:, start:start + batch_size]
                yb = y_shuffled[:, start:start + batch_size]
                output = self.forward(xb)
                epoch_loss += float(np.mean((output - yb) ** 2))
                self.backward(xb, yb)
                for layer in range(len(self.weights)):
                    self.weights[layer] -= learning_rate * self._grad_w[layer]
                    self.biases[layer] -= learning_rate * self._grad_b[layer]

            mean_loss = epoch_loss / max(1, len(range(0, n_samples, batch_size)))
            self.loss_history.append(mean_loss)
            if epoch % max(1, epochs // 10) == 0:
                print(f"Эпоха {epoch:4d} | loss = {mean_loss:.6f}")
        return self.loss_history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Предсказание без сохранения промежуточных состояний."""
        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.activation(w @ a + b)
        return a

    # ---------- сериализация ----------
    def save_weights(self, path: str | Path) -> None:
        """Сохраняет веса и смещения в JSON-файл."""
        try:
            payload = {
                "layer_sizes": self.layer_sizes,
                "activation": self.activation_name,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
            }
            Path(path).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            raise WeightsIOError(f"Не удалось сохранить веса: {exc}") from exc

    @classmethod
    def load_weights(cls, path: str | Path) -> "NeuralNetwork":
        """Загружает сеть из JSON-файла."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WeightsIOError(f"Не удалось прочитать веса: {exc}") from exc

        net = cls(payload["layer_sizes"], payload["activation"])
        net.weights = [np.array(w) for w in payload["weights"]]
        net.biases = [np.array(b) for b in payload["biases"]]
        return net
