"""Тесты служебных эндпоинтов: /, /health, /version, /docs."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_returns_service_info(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"]
    assert payload["version"]


def test_health_returns_ok_with_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_version"]
    assert payload["items"] > 0
    assert payload["error"] is None


def test_health_without_model(broken_client: TestClient) -> None:
    response = broken_client.get("/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["error"]


def test_version_returns_model_info(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"]
    assert payload["model_version"]
    assert payload["model_trained_at"]


def test_docs_is_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
