"""Шаг пайплайна ClearML: подготовка выборок для обучения.

Скачивает обучающую выборку из S3, делит её на train/validation/test и
публикует части артефактами задачи, чтобы шаги обучения не ходили в S3.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
from clearml import Task

PROJECT_NAME = "Ecommerce Purchase Predictor"
TASK_NAME = "load_data"


def bootstrap_repo(repo_dir: str) -> None:
    """Добавляет каталог репозитория в sys.path, чтобы импортировать пакет src."""
    if repo_dir and (Path(repo_dir) / "src").is_dir():
        sys.path.insert(0, repo_dir)


def main() -> int:
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.data_processing,
    )
    params = task.connect(
        {"repo_dir": "", "validation_size": 0.2, "test_size": 0.2, "random_state": 42}
    )
    bootstrap_repo(str(params["repo_dir"]))

    from src import storage
    from src.config import TRAINING_SAMPLE_FILE, get_settings
    from src.features import TARGET_COLUMN, describe_sample
    from src.train_model import split_sample

    settings = get_settings()
    work_dir = Path(tempfile.mkdtemp(prefix="purchase_pipeline_"))
    local_sample = Path(
        storage.download_file(
            settings.processed_key(TRAINING_SAMPLE_FILE),
            work_dir / TRAINING_SAMPLE_FILE,
            settings,
        )
    )

    sample = pd.read_parquet(local_sample)
    summary = describe_sample(sample)
    print(
        f"[load] Выборка: {int(summary['rows'])} строк, "
        f"положительных {summary['positive_rate']:.4%}"
    )
    print(f"[load] Товаров: {int(summary['items'])}, посетителей: {int(summary['visitors'])}")

    train, validation, test = split_sample(
        sample,
        validation_size=float(params["validation_size"]),
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
    )
    print(f"[load] train {len(train)}, validation {len(validation)}, test {len(test)}")
    print(f"[load] Доля положительных в train: {train[TARGET_COLUMN].mean():.4%}")

    for name, part in (("train", train), ("validation", validation), ("test", test)):
        path = work_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        task.upload_artifact(name, str(path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
