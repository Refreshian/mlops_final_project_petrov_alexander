"""Шаг пайплайна ClearML: выбор лучшей модели и выгрузка её в S3.

Сравнивает PR-AUC моделей из предыдущих шагов, берёт лучшую и выгружает
`ranker.pkl` в S3. Справочник фичей товаров (`item_features.parquet`) уже
лежит в S3: его готовит DAG предобработки.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

from clearml import Task

PROJECT_NAME = "Ecommerce Purchase Predictor"
TASK_NAME = "select_best_model"

#: Метрика, по которой выбираем модель: положительный класс очень редкий
SELECTION_METRIC = "pr_auc"


def bootstrap_repo(repo_dir: str) -> None:
    """Добавляет каталог репозитория в sys.path, чтобы импортировать пакет src."""
    if repo_dir and (Path(repo_dir) / "src").is_dir():
        sys.path.insert(0, repo_dir)


def main() -> int:
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.custom,
    )
    params = task.connect({"repo_dir": "", "train_task_ids": ""})
    bootstrap_repo(str(params["repo_dir"]))

    from src import storage
    from src.config import MODEL_FILE, get_settings

    candidate_ids = [item for item in str(params["train_task_ids"]).split(",") if item]
    if not candidate_ids:
        raise ValueError("Не передан список задач с обученными моделями")

    best_score = -1.0
    best_artifact: dict[str, object] | None = None
    best_task_name = ""

    for task_id in candidate_ids:
        candidate_task = Task.get_task(task_id=task_id)
        model_path = Path(candidate_task.artifacts["model"].get_local_copy())
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)

        score = float(artifact["metrics"][SELECTION_METRIC])
        print(
            f"[select] {candidate_task.name}: {SELECTION_METRIC}={score}, "
            f"параметры={artifact.get('params')}"
        )
        if score > best_score:
            best_score, best_artifact, best_task_name = score, artifact, candidate_task.name

    if best_artifact is None:
        raise RuntimeError("Не удалось прочитать ни одну обученную модель")

    print(f"[select] Лучшая модель: {best_task_name} с {SELECTION_METRIC}={best_score}")

    model_out = Path(tempfile.mkdtemp(prefix="purchase_pipeline_")) / MODEL_FILE
    with model_out.open("wb") as handle:
        pickle.dump(best_artifact, handle)

    settings = get_settings()
    key = settings.model_key(MODEL_FILE)
    storage.upload_file(model_out, key, settings)
    print(f"[select] Выгружено: s3://{settings.bucket}/{key}")

    logger = task.get_logger()
    logger.report_scalar(title="best_model", series=SELECTION_METRIC, iteration=0, value=best_score)
    logger.report_text(f"Выбрана модель шага {best_task_name}")
    task.upload_artifact("model", str(model_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
