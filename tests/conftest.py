"""Общие фикстуры тестов.

Тесты работают на небольшом сэмпле реальных данных из tests/data: артефакты
собираются и модель обучается прямо в фикстурах, поэтому ни S3, ни заранее
подготовленные файлы для прогона не нужны.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import ITEM_FEATURES_FILE
from src.features import build_item_features, build_training_sample, read_events
from src.main import create_app
from src.train_model import TrainParams, run_training

DATA_DIR = Path(__file__).parent / "data"

#: Каталог для артефактов тестового прогона (создаётся и удаляется фикстурой)
BUILD_DIR = Path(__file__).parent / "_build"

#: Маленькая модель: тестам важно, что сервис отвечает, а не качество
TEST_PARAMS = TrainParams(iterations=50, depth=4)


@pytest.fixture(scope="session")
def build_dir() -> Iterator[Path]:
    """Каталог, куда тесты складывают промежуточные артефакты."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    yield BUILD_DIR
    shutil.rmtree(BUILD_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def events() -> pd.DataFrame:
    """События из сэмпла исходных данных."""
    return read_events(DATA_DIR / "events.csv", chunk_size=20_000)


@pytest.fixture(scope="session")
def item_features(events: pd.DataFrame) -> pd.DataFrame:
    """Справочник фичей товаров, посчитанный по сэмплу."""
    return build_item_features(
        events,
        [
            DATA_DIR / "item_properties_part1.csv",
            DATA_DIR / "item_properties_part2.csv",
        ],
    )


@pytest.fixture(scope="session")
def sample(events: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    """Обучающая выборка по сэмплу."""
    return build_training_sample(events, item_features)


@pytest.fixture(scope="session")
def artifacts_dir(
    sample: pd.DataFrame,
    item_features: pd.DataFrame,
    build_dir: Path,
) -> Path:
    """Каталог с ranker.pkl и item_features.parquet — как их видит сервис."""
    output_dir = build_dir / "artifacts"
    sample_path = build_dir / "data_for_training.parquet"
    sample.to_parquet(sample_path, index=False)

    run_training(sample_path, output_dir, TEST_PARAMS)
    item_features.to_parquet(output_dir / ITEM_FEATURES_FILE, index=False)
    return output_dir


@pytest.fixture(scope="session")
def known_itemid(item_features: pd.DataFrame) -> int:
    """Товар, который точно есть в справочнике."""
    return int(item_features["itemid"].iloc[0])


@pytest.fixture()
def client(artifacts_dir: Path) -> Iterator[TestClient]:
    """Клиент сервиса с загруженной моделью."""
    app = create_app(model_dir=artifacts_dir, download_model=False)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def broken_client(build_dir: Path) -> Iterator[TestClient]:
    """Клиент сервиса без артефактов: модель не загружается."""
    empty_dir = build_dir / "empty"
    empty_dir.mkdir(exist_ok=True)
    app = create_app(model_dir=empty_dir, download_model=False)
    with TestClient(app) as test_client:
        yield test_client
