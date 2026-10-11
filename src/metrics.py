"""Метрики Prometheus, которые отдаёт /metrics."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

#: Количество обращений к эндпоинтам с разбивкой по коду ответа
REQUESTS = Counter(
    "predict_requests_total",
    "Количество запросов к сервису",
    ["endpoint", "status"],
)

#: Время обработки запроса на предсказание
LATENCY = Histogram(
    "predict_latency_seconds",
    "Время расчёта предсказания",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

#: Распределение выдаваемых вероятностей — помогает заметить дрейф модели
PROBABILITY = Histogram(
    "predict_purchase_probability",
    "Распределение вероятностей покупки в ответах сервиса",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

#: Нашёлся ли товар в справочнике (иначе используются значения по умолчанию)
ITEM_LOOKUP = Counter(
    "predict_item_lookup_total",
    "Результат поиска товара в справочнике фичей",
    ["result"],
)

#: Загружена ли модель: 1 — да, 0 — сервис отвечает без модели
MODEL_LOADED = Gauge(
    "model_loaded",
    "Загружена ли модель при старте сервиса",
)

#: Качество загруженной модели (PR-AUC на отложенной выборке)
MODEL_PR_AUC = Gauge(
    "model_pr_auc",
    "PR-AUC загруженной модели",
    ["version"],
)


def set_model_metrics(version: str, pr_auc: float) -> None:
    """Публикует версию модели, её PR-AUC и признак успешной загрузки."""
    MODEL_PR_AUC.labels(version=version).set(pr_auc)
    MODEL_LOADED.set(1)
