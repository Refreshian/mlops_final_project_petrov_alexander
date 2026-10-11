"""Расчёт фичей товаров и обучающей выборки.

Сервис отвечает на вопрос «какова вероятность, что товар купят после
просмотра», поэтому данные готовятся в два шага:

* **фичи товаров** — накопленная популярность (`views`, `purchases`, `ctr`)
  и свойства из item_properties (`categoryid`, `available`); это же
  справочник, по которому работает сервис;
* **обучающая выборка** — строка на каждый просмотр, целевая переменная равна
  единице, если тот же посетитель купил этот товар в течение 24 часов после
  просмотра.

Исходные данные RetailRocket: `events.csv` и `item_properties_part*.csv`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

#: Событие просмотра товара
VIEW_EVENT = "view"

#: События покупки (в датасете RetailRocket встречается только `transaction`)
PURCHASE_EVENTS = ("transaction", "purchase")

#: Свойства товаров, которые попадают в фичи
SIMPLE_PROPERTIES = ("categoryid", "available")

#: Признаки, которые видит модель
FEATURE_COLUMNS = [
    "views",
    "purchases",
    "ctr",
    "hour",
    "weekday",
    "categoryid",
    "available",
]

#: Целевая переменная
TARGET_COLUMN = "target"

#: Колонки справочника фичей товаров
ITEM_FEATURE_COLUMNS = ["itemid", "views", "purchases", "ctr", "categoryid", "available"]

#: Колонки обучающей выборки
SAMPLE_COLUMNS = [
    "timestamp",
    "visitorid",
    "itemid",
    "hour",
    "weekday",
    "views",
    "purchases",
    "ctr",
    "categoryid",
    "available",
    TARGET_COLUMN,
]

#: Окно, в течение которого покупка считается следствием просмотра
DEFAULT_WINDOW_HOURS = 24

#: Размер чанка при чтении больших csv (item_properties занимает ~850 МБ)
CHUNK_SIZE = 1_000_000

#: Значения по умолчанию для товара, которого нет в справочнике
DEFAULT_ITEM_FEATURES: dict[str, float] = {
    "views": 0.0,
    "purchases": 0.0,
    "ctr": 0.0,
    "categoryid": 0.0,
    "available": 0.0,
}


def read_events(path: str | Path, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Читает события посетителей, оставляя только нужные колонки."""
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=["timestamp", "visitorid", "event", "itemid"],
        dtype={"timestamp": "int64", "visitorid": "int64", "itemid": "int64", "event": "str"},
        chunksize=chunk_size,
    )
    for chunk in reader:
        chunk["event"] = chunk["event"].astype("str").str.lower()
        frames.append(chunk)

    if not frames:
        raise ValueError(f"Файл событий пуст: {path}")

    events = pd.concat(frames, ignore_index=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    return events


def read_item_properties(
    paths: Sequence[str | Path],
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Возвращает последние значения свойств `categoryid` и `available` по товарам.

    Файлы свойств отсортированы по времени, поэтому актуальным считается
    последнее встреченное значение. В памяти держим по одной записи на пару
    «товар + свойство», а не весь файл целиком.
    """
    latest: pd.DataFrame | None = None
    columns = ["timestamp", "itemid", "property", "value"]

    for path in paths:
        reader = pd.read_csv(
            path,
            usecols=columns,
            dtype={"timestamp": "int64", "itemid": "int64", "property": "str", "value": "str"},
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk = chunk[chunk["property"].isin(SIMPLE_PROPERTIES)]
            if chunk.empty:
                continue
            tail_index = chunk.drop_duplicates(subset=["itemid", "property"], keep="last").index
            chunk = chunk.loc[tail_index, columns]
            if latest is None:
                latest = chunk
            else:
                latest = pd.concat([latest, chunk], ignore_index=True).drop_duplicates(
                    subset=["itemid", "property"], keep="last"
                )

    if latest is None:
        return pd.DataFrame(columns=["itemid", *SIMPLE_PROPERTIES])

    wide = latest.pivot(index="itemid", columns="property", values="value").reset_index()
    for property_name in SIMPLE_PROPERTIES:
        if property_name not in wide.columns:
            wide[property_name] = np.nan
        wide[property_name] = pd.to_numeric(wide[property_name], errors="coerce")

    return wide[["itemid", *SIMPLE_PROPERTIES]]


def build_item_features(
    events: pd.DataFrame,
    properties_paths: Sequence[str | Path],
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Собирает справочник фичей товаров: популярность и свойства."""
    views = events.loc[events["event"] == VIEW_EVENT].groupby("itemid").size().rename("views")
    purchases = (
        events.loc[events["event"].isin(PURCHASE_EVENTS)]
        .groupby("itemid")
        .size()
        .rename("purchases")
    )

    item_features = pd.concat([views, purchases], axis=1).fillna(0).astype("int64")
    item_features["ctr"] = (
        item_features["purchases"] / item_features["views"].clip(lower=1)
    ).round(6)

    properties = read_item_properties(properties_paths, chunk_size=chunk_size)
    item_features = item_features.reset_index().merge(properties, on="itemid", how="left")

    # У товара без свойств оставляем нули
    item_features["categoryid"] = item_features["categoryid"].fillna(0).astype("int64")
    item_features["available"] = (
        item_features["available"].fillna(0).clip(lower=0, upper=1).astype("int64")
    )

    return item_features[ITEM_FEATURE_COLUMNS].sort_values("itemid").reset_index(drop=True)


def build_training_sample(
    events: pd.DataFrame,
    item_features: pd.DataFrame,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Собирает обучающую выборку: строка на просмотр, таргет — покупка после него."""
    views = events.loc[events["event"] == VIEW_EVENT, ["timestamp", "visitorid", "itemid"]]
    views = views.reset_index(drop=True)
    if sample_size is not None and len(views) > sample_size:
        views = views.sample(n=sample_size, random_state=42).reset_index(drop=True)

    purchases = (
        events.loc[events["event"].isin(PURCHASE_EVENTS), ["timestamp", "visitorid", "itemid"]]
        .rename(columns={"timestamp": "purchase_time"})
        .reset_index(drop=True)
    )

    # Соединяем просмотры с покупками того же товара тем же посетителем и
    # проверяем, попала ли покупка в окно после просмотра
    views = views.assign(view_id=np.arange(len(views), dtype="int64"))
    joined = views.merge(purchases, on=["visitorid", "itemid"], how="left")
    within_window = (joined["purchase_time"] > joined["timestamp"]) & (
        joined["purchase_time"] <= joined["timestamp"] + pd.Timedelta(hours=window_hours)
    )
    target = within_window.groupby(joined["view_id"]).any().astype("int8")

    sample = views.assign(target=target.reindex(views["view_id"]).fillna(False).to_numpy())
    sample = sample.drop(columns=["view_id"])
    sample["hour"] = sample["timestamp"].dt.hour.astype("int64")
    sample["weekday"] = sample["timestamp"].dt.dayofweek.astype("int64")

    sample = sample.merge(item_features, on="itemid", how="left")
    for column in ("views", "purchases", "categoryid", "available"):
        sample[column] = sample[column].fillna(0).astype("int64")
    sample["ctr"] = sample["ctr"].fillna(0.0).astype("float64")

    return sample[SAMPLE_COLUMNS].reset_index(drop=True)


def model_matrix(sample: pd.DataFrame) -> pd.DataFrame:
    """Матрица признаков для модели."""
    matrix = sample[FEATURE_COLUMNS].astype("float64")
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_request_row(
    item_features: pd.DataFrame,
    itemid: int,
    hour: int,
    weekday: int,
) -> tuple[pd.DataFrame, bool]:
    """Готовит строку признаков для запроса к сервису.

    `item_features` — справочник товаров с индексом по `itemid`. Для известного
    товара берутся его фичи, для неизвестного — значения по умолчанию; час и
    день недели всегда подставляются из запроса. Второй элемент результата
    показывает, нашёлся ли товар в справочнике.
    """
    if itemid in item_features.index:
        row = item_features.loc[itemid].to_dict()
        found = True
    else:
        row = dict(DEFAULT_ITEM_FEATURES)
        found = False

    row["hour"] = float(hour)
    row["weekday"] = float(weekday)
    for column in FEATURE_COLUMNS:
        row.setdefault(column, DEFAULT_ITEM_FEATURES.get(column, 0.0))

    matrix = pd.DataFrame([row])[FEATURE_COLUMNS].astype("float64")
    return matrix, found


def describe_sample(sample: pd.DataFrame) -> dict[str, float]:
    """Краткая сводка по выборке — для логов и метрик."""
    return {
        "rows": float(len(sample)),
        "items": float(sample["itemid"].nunique()),
        "visitors": float(sample["visitorid"].nunique()),
        "positives": float(int(sample[TARGET_COLUMN].sum())),
        "positive_rate": round(float(sample[TARGET_COLUMN].mean()), 6),
    }
