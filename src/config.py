"""Настройки путей: бакет, префиксы и имена артефактов в S3."""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Бакет стенда курса (переопределяется переменной окружения S3_BUCKET)
DEFAULT_BUCKET = "r-mlops-bucket-9-1-1-65671054"

#: Корневой префикс проекта в бакете
DEFAULT_PREFIX = "petrov_alexander_project"

#: Файлы исходного датасета RetailRocket
RAW_FILES = (
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
)

#: Справочник фичей товаров — по нему работает сервис
ITEM_FEATURES_FILE = "item_features.parquet"

#: Обучающая выборка: строка на каждый просмотр товара
TRAINING_SAMPLE_FILE = "data_for_training.parquet"

#: Обученная модель
MODEL_FILE = "ranker.pkl"


@dataclass(frozen=True)
class Settings:
    """Пути хранения данных и артефактов модели."""

    bucket: str
    prefix: str
    endpoint_url: str

    @property
    def raw_prefix(self) -> str:
        return f"{self.prefix}/raw_data"

    @property
    def processed_prefix(self) -> str:
        return f"{self.prefix}/processed_data"

    @property
    def models_prefix(self) -> str:
        return f"{self.prefix}/models"

    def raw_key(self, filename: str) -> str:
        return f"{self.raw_prefix}/{filename}"

    def processed_key(self, filename: str) -> str:
        return f"{self.processed_prefix}/{filename}"

    def model_key(self, filename: str) -> str:
        return f"{self.models_prefix}/{filename}"


def get_settings() -> Settings:
    """Читает настройки из переменных окружения.

    Значения по умолчанию соответствуют стенду курса, поэтому для запуска
    на нём достаточно задать только ключи доступа.
    """
    return Settings(
        bucket=os.environ.get("S3_BUCKET", DEFAULT_BUCKET),
        prefix=os.environ.get("S3_PREFIX", DEFAULT_PREFIX),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net"),
    )
