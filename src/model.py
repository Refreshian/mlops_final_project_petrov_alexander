"""Загрузка артефактов модели из S3 и расчёт предсказаний."""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src import storage
from src.config import ITEM_FEATURES_FILE, MODEL_FILE, get_settings
from src.features import build_request_row

logger = logging.getLogger(__name__)

#: Каталог с артефактами внутри контейнера
DEFAULT_MODEL_DIR = "/app/artifacts"


@dataclass
class PurchaseRanker:
    """Модель вероятности покупки и справочник фичей товаров."""

    model: Any
    feature_columns: list[str]
    item_features: pd.DataFrame
    version: str
    trained_at: str
    metrics: dict[str, float]

    @property
    def items(self) -> int:
        """Количество товаров в справочнике."""
        return int(len(self.item_features))

    @classmethod
    def load(
        cls,
        model_dir: str | Path | None = None,
        download: bool = True,
    ) -> PurchaseRanker:
        """Загружает модель и справочник, при необходимости скачивая их из S3."""
        directory = (
            Path(model_dir)
            if model_dir is not None
            else Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
        )
        model_path = directory / MODEL_FILE
        item_features_path = directory / ITEM_FEATURES_FILE

        if download and not (model_path.exists() and item_features_path.exists()):
            cls._download_artifacts(directory)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели {model_path}. Проверьте MODEL_DIR и доступ к S3."
            )
        if not item_features_path.exists():
            raise FileNotFoundError(f"Не найден справочник фичей товаров {item_features_path}")

        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)

        item_features = pd.read_parquet(item_features_path).set_index("itemid").sort_index()

        return cls(
            model=artifact["model"],
            feature_columns=list(artifact["feature_columns"]),
            item_features=item_features,
            version=str(artifact.get("version", "unknown")),
            trained_at=str(artifact.get("trained_at", "")),
            metrics=dict(artifact.get("metrics", {})),
        )

    @staticmethod
    def _download_artifacts(directory: Path) -> None:
        """Скачивает модель и справочник фичей из S3."""
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            logger.warning("Ключи S3 не заданы, скачивание артефактов пропущено")
            return

        settings = get_settings()
        targets = (
            (settings.model_key(MODEL_FILE), MODEL_FILE),
            (settings.processed_key(ITEM_FEATURES_FILE), ITEM_FEATURES_FILE),
        )
        for key, filename in targets:
            storage.download_file(key, directory / filename, settings)
            logger.info("Скачано из S3: s3://%s/%s", settings.bucket, key)

    def predict(self, itemid: int, hour: int, weekday: int) -> tuple[float, bool]:
        """Возвращает вероятность покупки и признак того, что товар есть в справочнике."""
        row, found = build_request_row(self.item_features, itemid, hour, weekday)
        probability = float(self.model.predict_proba(row[self.feature_columns])[0, 1])
        return probability, found
