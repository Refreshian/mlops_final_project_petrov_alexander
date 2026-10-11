"""Подготовка данных: исходные файлы из S3 -> фичи товаров и обучающая выборка.

Локальный запуск на уже скачанных данных:

    python -m src.prepare_features --events ecom_raw/events.csv \\
        --properties ecom_raw/item_properties_part1.csv ecom_raw/item_properties_part2.csv

Полный цикл (скачать из S3, обработать, выгрузить обратно):

    python -m src.prepare_features --from-s3 --upload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src import storage
from src.config import (
    ITEM_FEATURES_FILE,
    TRAINING_SAMPLE_FILE,
    get_settings,
)
from src.features import (
    DEFAULT_WINDOW_HOURS,
    build_item_features,
    build_training_sample,
    describe_sample,
    read_events,
)

#: Файлы свойств товаров по умолчанию
DEFAULT_PROPERTIES = ("item_properties_part1.csv", "item_properties_part2.csv")


def prepare(
    events_path: str | Path,
    properties_paths: list[str | Path],
    output_dir: str | Path,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    sample_size: int | None = None,
) -> tuple[Path, Path]:
    """Считает фичи товаров и обучающую выборку, сохраняет их в parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = read_events(events_path)
    print(f"[prepare] Событий прочитано: {len(events)}")
    print(
        f"[prepare] Покупок в логах: {int(events['event'].isin(['transaction', 'purchase']).sum())}"
    )

    item_features = build_item_features(events, properties_paths)
    sample = build_training_sample(
        events,
        item_features,
        window_hours=window_hours,
        sample_size=sample_size,
    )

    item_features_path = output_dir / ITEM_FEATURES_FILE
    sample_path = output_dir / TRAINING_SAMPLE_FILE
    item_features.to_parquet(item_features_path, index=False)
    sample.to_parquet(sample_path, index=False)

    summary = describe_sample(sample)
    positive_rate = summary["positive_rate"]
    print(f"[prepare] Товаров в справочнике: {len(item_features)}")
    print(f"[prepare] Строк в обучающей выборке: {int(summary['rows'])}")
    print(f"[prepare] Положительных примеров: {int(summary['positives'])} ({positive_rate:.4%})")
    print(f"[prepare] Окно покупки: {window_hours} ч")
    print(f"[prepare] Сохранено: {item_features_path} и {sample_path}")
    return item_features_path, sample_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовка данных для ранкера покупок")
    parser.add_argument("--events", default=os.environ.get("EVENTS_FILE", "ecom_raw/events.csv"))
    parser.add_argument(
        "--properties",
        nargs="*",
        default=[f"ecom_raw/{name}" for name in DEFAULT_PROPERTIES],
    )
    parser.add_argument(
        "--output-dir", default=os.environ.get("PROCESSED_DATA_DIR", "data/processed")
    )
    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--sample", type=int, default=None, help="ограничить число просмотров")
    parser.add_argument("--from-s3", action="store_true", help="скачать исходные файлы из S3")
    parser.add_argument("--upload", action="store_true", help="выгрузить результат в S3")
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = Path(args.events).parent

    if args.from_s3:
        downloaded = storage.download_prefix(settings.raw_prefix, raw_dir, settings)
        print(f"[prepare] Из S3 скачано файлов: {len(downloaded)}")

    item_features_path, sample_path = prepare(
        events_path=args.events,
        properties_paths=args.properties,
        output_dir=args.output_dir,
        window_hours=args.window_hours,
        sample_size=args.sample,
    )

    if args.upload:
        for path in (item_features_path, sample_path):
            key = settings.processed_key(path.name)
            storage.upload_file(path, key, settings)
            print(f"[prepare] Выгружено: s3://{settings.bucket}/{key}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
