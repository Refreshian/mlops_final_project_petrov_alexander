"""Схемы ответов сервиса."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Ответ на запрос вероятности покупки."""

    itemid: int = Field(description="Идентификатор товара")
    hour: int = Field(ge=0, le=23, description="Час суток из запроса")
    weekday: int = Field(ge=0, le=6, description="День недели, 0 — понедельник")
    purchase_probability: float = Field(
        ge=0.0, le=1.0, description="Вероятность покупки товара после просмотра"
    )
    item_found: bool = Field(description="Нашёлся ли товар в справочнике фичей")
    model_version: str = Field(description="Версия загруженной модели")


class HealthResponse(BaseModel):
    """Ответ проверки работоспособности сервиса."""

    status: str = Field(description="ok, если модель загружена")
    model_version: str | None = Field(default=None, description="Версия модели")
    items: int | None = Field(default=None, description="Размер справочника товаров")
    pr_auc: float | None = Field(default=None, description="PR-AUC модели на отложенной выборке")
    error: str | None = Field(default=None, description="Причина, если модель не загрузилась")


class VersionResponse(BaseModel):
    """Версия сервиса и модели."""

    service: str = Field(description="Версия кода сервиса")
    model_version: str | None = Field(default=None, description="Версия модели из ranker.pkl")
    model_trained_at: str | None = Field(default=None, description="Когда модель обучена")
