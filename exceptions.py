"""Собственные исключения для приложения нейронной сети."""

class NeuralNetError(Exception):
    """Базовое исключение приложения."""

class InvalidLayerSizeError(NeuralNetError):
    """Некорректный размер слоя сети."""

class UnknownActivationError(NeuralNetError):
    """Запрошена неизвестная функция активации."""

class MismatchedDataError(NeuralNetError):
    """Несовместимость размерностей входных данных."""

class DataFileError(NeuralNetError):
    """Проблема при чтении файла данных."""

class WeightsIOError(NeuralNetError):
    """Ошибка при сохранении или загрузке весов."""
