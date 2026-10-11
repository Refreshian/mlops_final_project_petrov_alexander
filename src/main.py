"""FastAPI-сервис вероятности покупки товара.

Эндпоинты:

* `GET /predict_purchase?itemid=&hour=&weekday=` — вероятность покупки;
* `GET /health` — состояние сервиса и загруженной модели;
* `GET /version` — версии сервиса и модели;
* `GET /metrics` — метрики Prometheus;
* `GET /docs` — интерактивная документация.

Запуск:

    python -m src.main
    # или
    uvicorn src.main:app --host 0.0.0.0 --port 32000
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src import metrics
from src.model import DEFAULT_MODEL_DIR, PurchaseRanker
from src.schemas import HealthResponse, PredictionResponse, VersionResponse

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("retail_recsys")

#: Версия кода сервиса
SERVICE_VERSION = "1.0.0"


def create_app(model_dir: str | Path | None = None, download_model: bool = True) -> FastAPI:
    """Собирает приложение; модель грузится при старте сервиса."""
    directory = (
        Path(model_dir)
        if model_dir is not None
        else Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    )
    state: dict[str, object] = {"ranker": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            ranker = PurchaseRanker.load(directory, download=download_model)
            state["ranker"] = ranker
            state["error"] = None
            metrics.set_model_metrics(ranker.version, ranker.metrics.get("pr_auc", 0.0))
            logger.info(
                "Модель версии %s загружена из %s, товаров в справочнике: %d",
                ranker.version,
                directory,
                ranker.items,
            )
        except Exception as error:  # сервис должен подняться даже без модели
            state["error"] = str(error)
            logger.error("Модель не загружена: %s", error)
        yield

    app = FastAPI(
        title="Retail Recommendation API",
        description=(
            "Сервис оценивает вероятность покупки товара после просмотра "
            "с учётом часа и дня недели."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    def current_ranker() -> PurchaseRanker | None:
        ranker = state["ranker"]
        return ranker if isinstance(ranker, PurchaseRanker) else None

    @app.get("/", tags=["system"], summary="Короткая справка о сервисе")
    def root() -> dict[str, str]:
        return {
            "message": "Retail recommender API running",
            "version": SERVICE_VERSION,
            "docs": "/docs",
        }

    @app.get("/version", tags=["system"], response_model=None, summary="Версии сервиса и модели")
    def version() -> VersionResponse:
        ranker = current_ranker()
        return VersionResponse(
            service=SERVICE_VERSION,
            model_version=ranker.version if ranker else None,
            model_trained_at=ranker.trained_at if ranker else None,
        )

    @app.get("/health", tags=["system"], response_model=None, summary="Проверка состояния")
    def health() -> HealthResponse | JSONResponse:
        ranker = current_ranker()
        if ranker is None:
            metrics.REQUESTS.labels(endpoint="/health", status="503").inc()
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="degraded",
                    error=str(state["error"] or "модель не загружена"),
                ).model_dump(),
            )

        metrics.REQUESTS.labels(endpoint="/health", status="200").inc()
        return HealthResponse(
            status="ok",
            model_version=ranker.version,
            items=ranker.items,
            pr_auc=ranker.metrics.get("pr_auc"),
        )

    @app.get(
        "/predict_purchase",
        tags=["predict"],
        response_model=None,
        summary="Вероятность покупки товара",
    )
    def predict_purchase(
        itemid: int = Query(..., ge=0, description="Идентификатор товара"),
        hour: int = Query(..., ge=0, le=23, description="Час суток, 0-23"),
        weekday: int = Query(..., ge=0, le=6, description="День недели, 0 — понедельник"),
    ) -> PredictionResponse | JSONResponse:
        ranker = current_ranker()
        if ranker is None:
            metrics.REQUESTS.labels(endpoint="/predict_purchase", status="503").inc()
            return JSONResponse(
                status_code=503, content={"detail": str(state["error"] or "модель не загружена")}
            )

        started = time.perf_counter()
        probability, found = ranker.predict(itemid, hour, weekday)
        metrics.LATENCY.labels(endpoint="/predict_purchase").observe(time.perf_counter() - started)
        metrics.PROBABILITY.observe(probability)
        metrics.ITEM_LOOKUP.labels(result="found" if found else "default").inc()
        metrics.REQUESTS.labels(endpoint="/predict_purchase", status="200").inc()

        return PredictionResponse(
            itemid=itemid,
            hour=hour,
            weekday=weekday,
            purchase_probability=round(probability, 6),
            item_found=found,
            model_version=ranker.version,
        )

    @app.get("/metrics", tags=["system"], summary="Метрики Prometheus")
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=32000, workers=1)
