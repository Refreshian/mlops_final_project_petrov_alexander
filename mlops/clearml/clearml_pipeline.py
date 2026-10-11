"""ML-пайплайн ClearML: обучение модели вероятности покупки.

Схема пайплайна:

    load_data ──┬─> train_catboost_shallow ──┐
                ├─> train_catboost_medium ───┤
                ├─> train_catboost_deep ─────┼─> select_best_model
                └─> tune_hyperparameters ────┘

* `load_data`            — берёт обучающую выборку из S3 и делит её на train/validation/test;
* `train_catboost_*`     — параллельные эксперименты с разными гиперпараметрами;
* `tune_hyperparameters` — подбор гиперпараметров через Optuna по PR-AUC;
* `select_best_model`    — сравнивает PR-AUC всех моделей и выгружает лучшую в S3 как `ranker.pkl`.

Запуск (из корня репозитория, с ключами S3 в переменных окружения):

    python mlops/clearml/clearml_pipeline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

from clearml import PipelineController, Task

PROJECT_NAME = "Ecommerce Purchase Predictor"
PIPELINE_NAME = "purchase_ranker_training"
PIPELINE_VERSION = "1.0.0"

#: Очередь ClearML, на которой работают агенты
QUEUE = os.environ.get("CLEARML_QUEUE", "default")

#: Каталог репозитория — шаги добавляют его в sys.path, чтобы импортировать пакет src
REPO_DIR = str(Path(__file__).resolve().parents[2])

STEPS_DIR = Path(__file__).resolve().parent / "steps"

#: Сколько испытаний Optuna запускает по умолчанию
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "15"))

#: Пакеты, которые агент ставит в окружение шагов
PACKAGES = [
    "clearml",
    "pandas",
    "pyarrow",
    "numpy",
    "scikit-learn",
    "catboost",
    "optuna",
    "boto3",
]


class Experiment(TypedDict):
    """Название шага и гиперпараметры одного эксперимента."""

    step: str
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float


#: Эксперименты с разными гиперпараметрами CatBoost
EXPERIMENTS: list[Experiment] = [
    {
        "step": "train_catboost_shallow",
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.08,
        "l2_leaf_reg": 3.0,
    },
    {
        "step": "train_catboost_medium",
        "iterations": 700,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
    },
    {
        "step": "train_catboost_deep",
        "iterations": 900,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 6.0,
    },
]


def create_step_task(task_name: str, script: Path, task_type: str) -> Task:
    """Создаёт задачу-шаблон для шага пайплайна.

    Скрипт шага сохраняется на сервере ClearML как standalone-файл, поэтому
    агенту не нужен доступ к git-репозиторию — достаточно каталога с кодом.
    """
    return Task.create(
        project_name=PROJECT_NAME,
        task_name=task_name,
        script=str(script),
        working_directory=str(STEPS_DIR),
        packages=PACKAGES,
        task_type=task_type,
        force_single_script_file=True,
    )


def build_pipeline() -> PipelineController:
    """Собирает пайплайн: подготовка выборок, эксперименты, выбор лучшей модели."""
    load_task = create_step_task(
        "load_data",
        STEPS_DIR / "load_data.py",
        Task.TaskTypes.data_processing,
    )
    train_tasks = {
        experiment["step"]: create_step_task(
            experiment["step"],
            STEPS_DIR / "train_model.py",
            Task.TaskTypes.training,
        )
        for experiment in EXPERIMENTS
    }
    tune_task = create_step_task(
        "tune_hyperparameters",
        STEPS_DIR / "tune_hyperparameters.py",
        Task.TaskTypes.optimizer,
    )
    select_task = create_step_task(
        "select_best_model",
        STEPS_DIR / "select_best_model.py",
        Task.TaskTypes.custom,
    )

    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PROJECT_NAME,
        version=PIPELINE_VERSION,
        add_pipeline_tags=False,
    )
    pipe.set_default_execution_queue(QUEUE)

    # Шаг 1: данные с признаками из S3
    pipe.add_step(
        name="load_data",
        base_task_id=load_task.id,
        execution_queue=QUEUE,
        parameter_override={"General/repo_dir": REPO_DIR},
    )

    # Шаг 2: параллельные эксперименты с гиперпараметрами и подбор через Optuna
    experiment_steps = []
    for experiment in EXPERIMENTS:
        step_name = experiment["step"]
        pipe.add_step(
            name=step_name,
            base_task_id=train_tasks[step_name].id,
            parents=["load_data"],
            parameter_override={
                "General/repo_dir": REPO_DIR,
                "General/load_data_task_id": "${load_data.id}",
                "General/iterations": experiment["iterations"],
                "General/depth": experiment["depth"],
                "General/learning_rate": experiment["learning_rate"],
                "General/l2_leaf_reg": experiment["l2_leaf_reg"],
            },
            execution_queue=QUEUE,
        )
        experiment_steps.append(step_name)

    pipe.add_step(
        name="tune_hyperparameters",
        base_task_id=tune_task.id,
        parents=["load_data"],
        parameter_override={
            "General/repo_dir": REPO_DIR,
            "General/load_data_task_id": "${load_data.id}",
            "General/n_trials": OPTUNA_TRIALS,
        },
        execution_queue=QUEUE,
    )
    experiment_steps.append("tune_hyperparameters")

    # Шаг 3: сравнение метрик и выгрузка лучшей модели в S3
    pipe.add_step(
        name="select_best_model",
        base_task_id=select_task.id,
        parents=experiment_steps,
        parameter_override={
            "General/repo_dir": REPO_DIR,
            "General/train_task_ids": ",".join(f"${{{name}.id}}" for name in experiment_steps),
        },
        execution_queue=QUEUE,
    )

    return pipe


def main() -> int:
    pipe = build_pipeline()
    print(f"Запускаем пайплайн '{PIPELINE_NAME}' в проекте '{PROJECT_NAME}' на очереди '{QUEUE}'")
    # Контроллер выполняется локально, а шаги уходят на агентов очереди: так он
    # не занимает единственный слот агента и не блокирует собственные шаги.
    pipe.start_locally(run_pipeline_steps_locally=False)
    print("Пайплайн завершён")
    return 0


if __name__ == "__main__":
    sys.exit(main())
