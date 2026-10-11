"""DAG подготовки данных для сервиса вероятности покупки.

Пайплайн:

    check_raw_files_in_s3 -> download_raw_files -> process_files
        -> upload_processed_data -> check_processed_files_in_s3

Из событий RetailRocket собираются два артефакта: справочник фичей товаров
(`item_features.parquet`) и обучающая выборка (`data_for_training.parquet`).
Оба выгружаются в S3 в префикс `processed_data`. Расписание — раз в сутки.

Доступ к S3 берётся из соединения Airflow (Admin -> Connections, тип
Amazon Web Services), ключи в коде DAG не хранятся.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

from src.config import (
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    ITEM_FEATURES_FILE,
    RAW_FILES,
    TRAINING_SAMPLE_FILE,
)
from src.features import build_item_features, build_training_sample, describe_sample, read_events
from src.validation import validate_events, validate_item_features, validate_training_sample

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
#: Идентификатор соединения Airflow с ключами доступа к S3
AWS_CONN_ID = "s3_conn"

BUCKET_NAME = os.environ.get("S3_BUCKET", DEFAULT_BUCKET)
PROJECT_PREFIX = os.environ.get("S3_PREFIX", DEFAULT_PREFIX)

RAW_PREFIX = f"{PROJECT_PREFIX}/raw_data"
PROCESSED_PREFIX = f"{PROJECT_PREFIX}/processed_data"

RAW_KEYS = [f"{RAW_PREFIX}/{name}" for name in RAW_FILES]
PROCESSED_KEYS = [
    f"{PROCESSED_PREFIX}/{ITEM_FEATURES_FILE}",
    f"{PROCESSED_PREFIX}/{TRAINING_SAMPLE_FILE}",
]

PROPERTY_FILES = ("item_properties_part1.csv", "item_properties_part2.csv")

LOCAL_DIR = Path("/tmp/preprocessing")
RAW_DIR = LOCAL_DIR / "raw"
OUTPUT_DIR = LOCAL_DIR / "processed"

#: Сколько строк событий проверяем схемой: файл целиком читать не нужно
VALIDATION_SAMPLE_ROWS = 200_000

#: Окно, в течение которого покупка считается следствием просмотра
WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "24"))


# ---------------------------------------------------------------------------
# Шаг 1. Загрузка исходных файлов из S3
# ---------------------------------------------------------------------------
def download_raw_files() -> dict[str, int]:
    """Скачивает файлы исходного датасета в локальный каталог воркера."""
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sizes: dict[str, int] = {}
    for name in RAW_FILES:
        key = f"{RAW_PREFIX}/{name}"
        local_path = RAW_DIR / name
        s3_hook.get_key(key=key, bucket_name=BUCKET_NAME).download_file(str(local_path))
        sizes[name] = local_path.stat().st_size
        print(f"[load] s3://{BUCKET_NAME}/{key} -> {local_path} ({sizes[name]} байт)")

    return sizes


# ---------------------------------------------------------------------------
# Шаг 2. Проверка схемы и сборка фичей
# ---------------------------------------------------------------------------
def process_files() -> dict[str, float]:
    """Проверяет события и собирает фичи товаров с обучающей выборкой."""
    sample = pd.read_csv(
        RAW_DIR / "events.csv",
        nrows=VALIDATION_SAMPLE_ROWS,
        dtype={"timestamp": "int64", "visitorid": "int64", "itemid": "int64", "event": "str"},
    )
    validate_events(sample)
    print(f"[transform] Схема событий проверена на {len(sample)} строках")

    events = read_events(RAW_DIR / "events.csv")
    item_features = build_item_features(events, [RAW_DIR / name for name in PROPERTY_FILES])
    training_sample = build_training_sample(events, item_features, window_hours=WINDOW_HOURS)

    item_features = validate_item_features(item_features)
    training_sample = validate_training_sample(training_sample)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    item_features.to_parquet(OUTPUT_DIR / ITEM_FEATURES_FILE, index=False)
    training_sample.to_parquet(OUTPUT_DIR / TRAINING_SAMPLE_FILE, index=False)

    summary = describe_sample(training_sample)
    print(f"[transform] Товаров в справочнике: {len(item_features)}")
    print(f"[transform] Строк в обучающей выборке: {int(summary['rows'])}")
    print(f"[transform] Доля положительных примеров: {summary['positive_rate']:.4%}")

    return summary


# ---------------------------------------------------------------------------
# Шаг 3. Выгрузка артефактов в S3
# ---------------------------------------------------------------------------
def upload_processed_data() -> list[str]:
    """Загружает справочник фичей и обучающую выборку в S3."""
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    uploaded = []

    for filename in (ITEM_FEATURES_FILE, TRAINING_SAMPLE_FILE):
        key = f"{PROCESSED_PREFIX}/{filename}"
        s3_hook.load_file(
            filename=str(OUTPUT_DIR / filename),
            key=key,
            bucket_name=BUCKET_NAME,
            replace=True,
        )
        uploaded.append(key)
        print(f"[load] Загружено в S3: s3://{BUCKET_NAME}/{key}")

    return uploaded


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner": "mlops-student",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="preprocessing_data",
    description="Подготовка фичей товаров и обучающей выборки: S3 -> S3",
    start_date=datetime(2025, 5, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "preprocessing", "s3", "pandera"],
    default_args=default_args,
) as dag:

    # Шаг 1. Ждём появления исходных файлов в S3
    check_raw_files_in_s3 = S3KeySensor(
        task_id="check_raw_files_in_s3",
        bucket_name=BUCKET_NAME,
        bucket_key=RAW_KEYS,
        aws_conn_id=AWS_CONN_ID,
        timeout=600,
        poke_interval=30,
        mode="poke",
    )

    # Шаг 2. Скачиваем исходные данные
    download_raw = PythonOperator(
        task_id="download_raw_files",
        python_callable=download_raw_files,
    )

    # Шаг 3. Валидация и сборка фичей с обучающей выборкой
    process = PythonOperator(
        task_id="process_files",
        python_callable=process_files,
    )

    # Шаг 4. Выгружаем артефакты в S3
    upload_processed = PythonOperator(
        task_id="upload_processed_data",
        python_callable=upload_processed_data,
    )

    # Шаг 5. Проверяем, что артефакты появились в S3
    check_processed_files_in_s3 = S3KeySensor(
        task_id="check_processed_files_in_s3",
        bucket_name=BUCKET_NAME,
        bucket_key=PROCESSED_KEYS,
        aws_conn_id=AWS_CONN_ID,
        timeout=300,
        poke_interval=15,
        mode="poke",
    )

    (
        check_raw_files_in_s3
        >> download_raw
        >> process
        >> upload_processed
        >> check_processed_files_in_s3
    )
