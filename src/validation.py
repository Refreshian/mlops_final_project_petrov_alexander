"""Pandera-схемы для проверки данных на входе и выходе обработки."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column

from src.features import FEATURE_COLUMNS, PURCHASE_EVENTS, TARGET_COLUMN, VIEW_EVENT

#: Схема исходных событий: то, что ожидаем увидеть в events.csv
events_schema = pa.DataFrameSchema(
    columns={
        "timestamp": Column("int64", required=True, checks=Check.ge(0)),
        "visitorid": Column("int64", required=True, checks=Check.ge(0)),
        "event": Column(
            "str",
            required=True,
            checks=Check.isin([VIEW_EVENT, *PURCHASE_EVENTS, "addtocart"]),
        ),
        "itemid": Column("int64", required=True, checks=Check.ge(0)),
        "transactionid": Column("float64", required=False, nullable=True),
    },
    strict=False,
    ordered=False,
)

#: Схема справочника фичей товаров
item_features_schema = pa.DataFrameSchema(
    columns={
        "itemid": Column("int64", required=True, checks=Check.ge(0)),
        "views": Column("int64", required=True, checks=Check.ge(0)),
        "purchases": Column("int64", required=True, checks=Check.ge(0)),
        "ctr": Column("float64", required=True, checks=Check.in_range(0, 1)),
        "categoryid": Column("int64", required=True, checks=Check.ge(0)),
        "available": Column("int64", required=True, checks=Check.isin([0, 1])),
    },
    strict=True,
    ordered=False,
    coerce=True,
    drop_invalid_rows=True,
)

#: Схема обучающей выборки: проверяем признаки и таргет
training_sample_schema = pa.DataFrameSchema(
    columns={
        "itemid": Column("int64", required=True, checks=Check.ge(0)),
        "hour": Column("int64", required=True, checks=Check.in_range(0, 23)),
        "weekday": Column("int64", required=True, checks=Check.in_range(0, 6)),
        "views": Column("int64", required=True, checks=Check.ge(0)),
        "purchases": Column("int64", required=True, checks=Check.ge(0)),
        "ctr": Column("float64", required=True, checks=Check.in_range(0, 1)),
        "categoryid": Column("int64", required=True, checks=Check.ge(0)),
        "available": Column("int64", required=True, checks=Check.isin([0, 1])),
        TARGET_COLUMN: Column("int64", required=True, checks=Check.isin([0, 1])),
    },
    strict=False,
    ordered=False,
    coerce=True,
    drop_invalid_rows=True,
)


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Проверяет исходные события по схеме events_schema."""
    checked: pd.DataFrame = events_schema.validate(events, lazy=True)
    return checked


def validate_item_features(item_features: pd.DataFrame) -> pd.DataFrame:
    """Проверяет справочник фичей товаров."""
    checked: pd.DataFrame = item_features_schema.validate(item_features, lazy=True)
    return checked


def validate_training_sample(sample: pd.DataFrame) -> pd.DataFrame:
    """Проверяет обучающую выборку."""
    checked: pd.DataFrame = training_sample_schema.validate(sample, lazy=True)
    return checked


#: Список признаков модели — используется тестами схем
FEATURE_COLUMN_NAMES = tuple(FEATURE_COLUMNS)
