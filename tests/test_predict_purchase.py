"""Тесты эндпоинта /predict_purchase."""

from __future__ import annotations

from fastapi.testclient import TestClient

#: Пример запроса из задания
EXAMPLE_QUERY = {"itemid": 356475, "hour": 14, "weekday": 3}


def test_predict_purchase_returns_probability(client: TestClient) -> None:
    response = client.get("/predict_purchase", params=EXAMPLE_QUERY)

    assert response.status_code == 200
    payload = response.json()
    assert payload["itemid"] == EXAMPLE_QUERY["itemid"]
    assert payload["hour"] == EXAMPLE_QUERY["hour"]
    assert payload["weekday"] == EXAMPLE_QUERY["weekday"]
    assert 0.0 <= payload["purchase_probability"] <= 1.0
    assert payload["model_version"]


def test_predict_purchase_known_item_is_found(client: TestClient, known_itemid: int) -> None:
    response = client.get(
        "/predict_purchase", params={"itemid": known_itemid, "hour": 12, "weekday": 1}
    )

    assert response.status_code == 200
    assert response.json()["item_found"] is True


def test_predict_purchase_unknown_item_uses_defaults(client: TestClient) -> None:
    response = client.get(
        "/predict_purchase", params={"itemid": 987_654_321, "hour": 12, "weekday": 1}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["item_found"] is False
    assert 0.0 <= payload["purchase_probability"] <= 1.0


def test_predict_purchase_depends_on_hour(client: TestClient, known_itemid: int) -> None:
    morning = client.get(
        "/predict_purchase", params={"itemid": known_itemid, "hour": 4, "weekday": 1}
    ).json()
    evening = client.get(
        "/predict_purchase", params={"itemid": known_itemid, "hour": 20, "weekday": 1}
    ).json()

    assert morning["purchase_probability"] != evening["purchase_probability"]


def test_predict_purchase_hour_out_of_range(client: TestClient) -> None:
    response = client.get("/predict_purchase", params={"itemid": 1, "hour": 25, "weekday": 1})

    assert response.status_code == 422


def test_predict_purchase_weekday_out_of_range(client: TestClient) -> None:
    response = client.get("/predict_purchase", params={"itemid": 1, "hour": 10, "weekday": 9})

    assert response.status_code == 422


def test_predict_purchase_without_itemid(client: TestClient) -> None:
    response = client.get("/predict_purchase", params={"hour": 10, "weekday": 1})

    assert response.status_code == 422


def test_predict_purchase_without_model(broken_client: TestClient) -> None:
    response = broken_client.get("/predict_purchase", params=EXAMPLE_QUERY)

    assert response.status_code == 503
