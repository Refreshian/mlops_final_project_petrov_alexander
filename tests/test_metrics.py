"""Тесты метрик Prometheus."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_endpoint_exposes_service_metrics(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "predict_requests_total" in body
    assert "predict_latency_seconds" in body
    assert "model_loaded" in body


def test_model_metrics_after_start(client: TestClient) -> None:
    body = client.get("/metrics").text

    assert "model_loaded 1.0" in body
    assert "model_pr_auc{version=" in body


def test_predict_increases_request_counter(client: TestClient) -> None:
    client.get("/predict_purchase", params={"itemid": 1, "hour": 5, "weekday": 2})
    body = client.get("/metrics").text

    line = next(
        item
        for item in body.splitlines()
        if item.startswith('predict_requests_total{endpoint="/predict_purchase",status="200"}')
    )
    assert float(line.split()[-1]) >= 1.0
