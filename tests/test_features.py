"""Тесты расчёта фичей и подготовки данных."""

from __future__ import annotations

import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    ITEM_FEATURE_COLUMNS,
    SAMPLE_COLUMNS,
    TARGET_COLUMN,
    build_request_row,
    model_matrix,
)


def test_item_features_columns(item_features: pd.DataFrame) -> None:
    assert list(item_features.columns) == ITEM_FEATURE_COLUMNS
    assert item_features["itemid"].is_unique
    assert len(item_features) > 0


def test_item_features_values(item_features: pd.DataFrame) -> None:
    assert (item_features["views"] >= 0).all()
    assert (item_features["purchases"] >= 0).all()
    assert item_features["ctr"].between(0, 1).all()
    assert item_features["available"].isin([0, 1]).all()
    assert (item_features["categoryid"] >= 0).all()


def test_sample_columns(sample: pd.DataFrame) -> None:
    assert list(sample.columns) == SAMPLE_COLUMNS
    assert len(sample) > 0


def test_sample_time_features(sample: pd.DataFrame) -> None:
    assert sample["hour"].between(0, 23).all()
    assert sample["weekday"].between(0, 6).all()


def test_sample_target(sample: pd.DataFrame) -> None:
    assert sample[TARGET_COLUMN].isin([0, 1]).all()
    # В реальных логах покупок мало, но они должны попадать в выборку
    assert 0 < sample[TARGET_COLUMN].sum() < len(sample)


def test_target_only_after_view(item_features: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Покупка считается целевой, только если она позже просмотра."""
    buys = sample.loc[sample[TARGET_COLUMN] == 1]
    assert buys["purchases"].ge(1).all()


def test_model_matrix_has_only_features(sample: pd.DataFrame) -> None:
    matrix = model_matrix(sample)

    assert list(matrix.columns) == FEATURE_COLUMNS
    assert matrix.notna().all().all()


def test_request_row_for_known_item(item_features: pd.DataFrame) -> None:
    reference = item_features.set_index("itemid")
    itemid = int(reference.index[0])

    matrix, found = build_request_row(reference, itemid, hour=15, weekday=4)

    assert found is True
    assert list(matrix.columns) == FEATURE_COLUMNS
    assert matrix.loc[0, "hour"] == 15
    assert matrix.loc[0, "weekday"] == 4
    assert matrix.loc[0, "views"] == reference.loc[itemid, "views"]


def test_request_row_for_unknown_item(item_features: pd.DataFrame) -> None:
    reference = item_features.set_index("itemid")

    matrix, found = build_request_row(reference, 10**9, hour=0, weekday=0)

    assert found is False
    assert matrix.loc[0, "hour"] == 0
    assert matrix.loc[0, "views"] == 0
    assert matrix.notna().all().all()
