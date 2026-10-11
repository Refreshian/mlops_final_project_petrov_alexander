"""Шаг пайплайна ClearML: подбор гиперпараметров через Optuna.

Optuna ищет комбинацию гиперпараметров CatBoost, максимизирующую PR-AUC на
валидационной выборке. Лучшие параметры и обученная на train+validation
модель публикуются в ClearML, метрики каждого испытания видны в графиках.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import optuna
import pandas as pd
from clearml import Task

PROJECT_NAME = "Ecommerce Purchase Predictor"
TASK_NAME = "tune_hyperparameters"

#: Сколько испытаний Optuna делает по умолчанию
DEFAULT_TRIALS = 15


def bootstrap_repo(repo_dir: str) -> None:
    """Добавляет каталог репозитория в sys.path, чтобы импортировать пакет src."""
    if repo_dir and (Path(repo_dir) / "src").is_dir():
        sys.path.insert(0, repo_dir)


def main() -> int:
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.optimizer,
    )
    params = task.connect(
        {"repo_dir": "", "load_data_task_id": "", "n_trials": DEFAULT_TRIALS, "random_seed": 42}
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

    seed = int(params["random_seed"])
    load_task = Task.get_task(task_id=str(params["load_data_task_id"]))
    train = pd.read_parquet(load_task.artifacts["train"].get_local_copy())
    validation = pd.read_parquet(load_task.artifacts["validation"].get_local_copy())
    test = pd.read_parquet(load_task.artifacts["test"].get_local_copy())

    logger = task.get_logger()

    def objective(trial: optuna.Trial) -> float:
        """Обучает модель с предложенными параметрами и возвращает PR-AUC."""
        train_params = TrainParams(
            iterations=trial.suggest_int("iterations", 300, 900, step=100),
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_seed=seed,
        )
        model = train_model(train, validation, train_params, verbose=0)
        metrics = evaluate(model, validation)
        pr_auc = metrics["pr_auc"]
        print(f"[tune] trial {trial.number}: pr_auc={pr_auc}, params={train_params.as_dict()}")
        for name, value in metrics.items():
            logger.report_scalar(
                title="trials",
                series=name,
                iteration=trial.number,
                value=value,
            )
        return pr_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(params["n_trials"]))

    print(f"[tune] Лучшие параметры: {study.best_params}")
    print(f"[tune] PR-AUC на валидации: {study.best_value}")

    best_params = TrainParams(
        iterations=int(study.best_params["iterations"]),
        depth=int(study.best_params["depth"]),
        learning_rate=float(study.best_params["learning_rate"]),
        l2_leaf_reg=float(study.best_params["l2_leaf_reg"]),
        random_seed=seed,
    )

    # Финальную модель обучаем на train + validation, метрики считаем на test
    full_train = pd.concat([train, validation], ignore_index=True)
    model = train_model(full_train, validation, best_params, verbose=0)
    threshold = select_threshold(model, validation)
    metrics = evaluate(model, test, threshold)
    print(f"[tune] Метрики финальной модели на test: {metrics}")

    for name, value in metrics.items():
        logger.report_scalar(title="best_model", series=name, iteration=0, value=value)
    logger.report_text(f"Лучшие гиперпараметры: {best_params.as_dict()}")

    model_path = Path(tempfile.mkdtemp(prefix="purchase_pipeline_")) / MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_columns": FEATURE_COLUMNS,
                "metrics": metrics,
                "params": best_params.as_dict(),
                "version": MODEL_VERSION,
            },
            handle,
        )

    task.upload_artifact("model", str(model_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
