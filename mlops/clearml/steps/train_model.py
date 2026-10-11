"""Шаг пайплайна ClearML: обучение CatBoost с заданными гиперпараметрами.

Выборки берутся из артефактов шага `load_data`, метрики уходят в ClearML,
готовая модель публикуется артефактом `model`.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import pandas as pd
from clearml import Task

PROJECT_NAME = "Ecommerce Purchase Predictor"
TASK_NAME = "train_model"


def bootstrap_repo(repo_dir: str) -> None:
    """Добавляет каталог репозитория в sys.path, чтобы импортировать пакет src."""
    if repo_dir and (Path(repo_dir) / "src").is_dir():
        sys.path.insert(0, repo_dir)


def main() -> int:
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
    )
    params = task.connect(
        {
            "repo_dir": "",
            "load_data_task_id": "",
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
        }
    )
    bootstrap_repo(str(params["repo_dir"]))

    from src.config import MODEL_FILE
    from src.features import FEATURE_COLUMNS
    from src.train_model import (
        MODEL_VERSION,
        TrainParams,
        evaluate,
        select_threshold,
        train_model,
    )

    train_params = TrainParams(
        iterations=int(params["iterations"]),
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_seed=int(params["random_seed"]),
    )
    print(f"[train] Гиперпараметры: {train_params.as_dict()}")

    load_task = Task.get_task(task_id=str(params["load_data_task_id"]))
    train = pd.read_parquet(load_task.artifacts["train"].get_local_copy())
    validation = pd.read_parquet(load_task.artifacts["validation"].get_local_copy())
    test = pd.read_parquet(load_task.artifacts["test"].get_local_copy())

    model = train_model(train, validation, train_params)
    threshold = select_threshold(model, validation)
    metrics = evaluate(model, test, threshold)

    logger = task.get_logger()
    for name, value in metrics.items():
        logger.report_scalar(title="metrics", series=name, iteration=0, value=value)
        print(f"[train] {name}: {value}")

    model_path = Path(tempfile.mkdtemp(prefix="purchase_pipeline_")) / MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_columns": FEATURE_COLUMNS,
                "metrics": metrics,
                "params": train_params.as_dict(),
                "version": MODEL_VERSION,
            },
            handle,
        )

    task.upload_artifact("model", str(model_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
