"""Обучение CatBoost-модели вероятности покупки.

Метрика качества — PR-AUC (`average_precision`): положительный класс очень
редкий, поэтому accuracy и ROC-AUC выглядят хорошо даже у бесполезной модели.
Выборка делится на train / validation / test: на validation работает ранняя
остановка, метрики считаются на отложенном test.

Запуск:

    python -m src.train_model --sample data/processed/data_for_training.parquet --upload
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import storage
from src.config import MODEL_FILE, TRAINING_SAMPLE_FILE, get_settings
from src.features import FEATURE_COLUMNS, TARGET_COLUMN, describe_sample, model_matrix

#: Версия артефакта: попадает в ответ /version и в метки метрик
MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class TrainParams:
    """Гиперпараметры CatBoost."""

    iterations: int = 700
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_sample(
    sample: pd.DataFrame,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит выборку на train, validation и test со стратификацией по таргету."""
    train, rest = train_test_split(
        sample,
        test_size=validation_size + test_size,
        random_state=random_state,
        stratify=sample[TARGET_COLUMN],
    )
    validation, test = train_test_split(
        rest,
        test_size=test_size / (validation_size + test_size),
        random_state=random_state,
        stratify=rest[TARGET_COLUMN],
    )
    return train, validation, test


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: TrainParams,
    verbose: int = 100,
) -> CatBoostClassifier:
    """Обучает CatBoost с ранней остановкой по PR-AUC на валидации."""
    model = CatBoostClassifier(
        iterations=params.iterations,
        depth=params.depth,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.l2_leaf_reg,
        random_seed=params.random_seed,
        loss_function="Logloss",
        eval_metric="PRAUC",
        od_type="Iter",
        od_wait=50,
        use_best_model=True,
        # Служебный каталог catboost_info в репозитории не нужен
        allow_writing_files=False,
        verbose=verbose,
    )
    model.fit(
        model_matrix(train),
        train[TARGET_COLUMN],
        eval_set=(model_matrix(validation), validation[TARGET_COLUMN]),
    )
    return model


def select_threshold(model: CatBoostClassifier, validation: pd.DataFrame) -> float:
    """Подбирает порог вероятности, максимизирующий F1 на валидации.

    Модель выдаёт маленькие вероятности (покупок мало), поэтому порог 0.5
    почти ничего не относит к положительному классу и метрики выглядят
    бесполезными. Порог подбирается на валидации и применяется к test.
    """
    probabilities = model.predict_proba(model_matrix(validation))[:, 1]
    precision, recall, thresholds = precision_recall_curve(validation[TARGET_COLUMN], probabilities)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(thresholds[int(np.nanargmax(f1_scores[:-1]))])


def evaluate(
    model: CatBoostClassifier,
    test: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Считает метрики на отложенной выборке при заданном пороге."""
    probabilities = model.predict_proba(model_matrix(test))[:, 1]
    target = test[TARGET_COLUMN]
    predictions = (probabilities >= threshold).astype("int8")

    return {
        "pr_auc": round(float(average_precision_score(target, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
        "precision": round(float(precision_score(target, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(target, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(target, predictions, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(target, predictions)), 4),
        "threshold": round(threshold, 6),
        "positive_rate": round(float(target.mean()), 6),
        "test_rows": float(len(test)),
    }


def save_model(
    model: CatBoostClassifier, metrics: dict[str, float], output_dir: str | Path
) -> Path:
    """Сохраняет модель вместе с метаданными в ranker.pkl."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "version": MODEL_VERSION,
        "params": model.get_params(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = output_dir / MODEL_FILE
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)

    print(f"[train] Модель сохранена: {path}")
    return path


def run_training(
    sample_path: str | Path,
    output_dir: str | Path,
    params: TrainParams | None = None,
) -> dict[str, float]:
    """Полный цикл: читаем выборку, обучаем, считаем метрики, сохраняем модель."""
    params = params or TrainParams()
    sample = pd.read_parquet(sample_path)
    summary = describe_sample(sample)
    positive_rate = summary["positive_rate"]
    print(f"[train] Выборка: {int(summary['rows'])} строк, положительных {positive_rate:.4%}")
    print(f"[train] Гиперпараметры: {params.as_dict()}")

    train, validation, test = split_sample(sample, random_state=params.random_seed)
    print(f"[train] train {len(train)}, validation {len(validation)}, test {len(test)}")

    model = train_model(train, validation, params)
    threshold = select_threshold(model, validation)
    print(f"[train] Порог вероятности по валидации: {threshold:.6f}")
    metrics = evaluate(model, test, threshold)
    print(f"[train] Метрики на test: {metrics}")

    save_model(model, metrics, output_dir)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучение модели вероятности покупки")
    parser.add_argument("--sample", default=f"data/processed/{TRAINING_SAMPLE_FILE}")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--iterations", type=int, default=TrainParams.iterations)
    parser.add_argument("--depth", type=int, default=TrainParams.depth)
    parser.add_argument("--learning-rate", type=float, default=TrainParams.learning_rate)
    parser.add_argument("--l2-leaf-reg", type=float, default=TrainParams.l2_leaf_reg)
    parser.add_argument("--upload", action="store_true", help="выгрузить модель в S3")
    args = parser.parse_args()

    params = TrainParams(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
    )
    run_training(args.sample, args.output_dir, params)

    if args.upload:
        settings = get_settings()
        key = settings.model_key(MODEL_FILE)
        storage.upload_file(Path(args.output_dir) / MODEL_FILE, key, settings)
        print(f"[train] Выгружено: s3://{settings.bucket}/{key}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
