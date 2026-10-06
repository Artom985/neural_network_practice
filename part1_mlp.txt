"""Быстрая проверка прямой и обратной связи без реальных данных."""

import numpy as np
from part1_mlp.app.models import NeuralNetwork

# XOR — классическая задача, сеть должна выучить её
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)   # (2 признака, 4 примера)
y = np.array([[0, 1, 1, 0]], dtype=float)   # XOR

net = NeuralNetwork([2, 8, 1], activation="sigmoid", seed=42)
net.train(X, y, learning_rate=0.5, epochs=3000, batch_size=4)

pred = net.predict(X).ravel()
for inp, out, target in zip(X.T, pred, y.ravel()):
    print(f"X={inp} -> {out:.4f} (ожидалось {target})")

assert np.allclose(pred.round(), y.ravel()), "Сеть не выучила XOR!"
print("\n[OK] XOR выучен, forward/backward работают корректно.")
