"""Обмен файлами с S3-совместимым хранилищем (Yandex Object Storage).

Ключи доступа берутся из переменных окружения AWS_ACCESS_KEY_ID и
AWS_SECRET_ACCESS_KEY — так их можно не хранить в коде и репозитории.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import boto3

from src.config import Settings, get_settings


def get_client(settings: Settings | None = None) -> Any:
    """Создаёт клиент S3 с указанием endpoint_url (нужен для Yandex Object Storage)."""
    settings = settings or get_settings()
    session = boto3.session.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    return session.client("s3", endpoint_url=settings.endpoint_url)


def upload_file(
    local_path: str | Path,
    key: str,
    settings: Settings | None = None,
) -> str:
    """Загружает локальный файл в бакет и возвращает ключ объекта."""
    settings = settings or get_settings()
    get_client(settings).upload_file(str(local_path), settings.bucket, key)
    return key


def download_file(
    key: str,
    local_path: str | Path,
    settings: Settings | None = None,
) -> str:
    """Скачивает объект в локальный файл, создавая при необходимости каталоги."""
    settings = settings or get_settings()
    target = Path(local_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    get_client(settings).download_file(settings.bucket, key, str(target))
    return str(target)


def download_prefix(
    prefix: str,
    local_dir: str | Path,
    settings: Settings | None = None,
) -> list[str]:
    """Скачивает все объекты с префиксом в каталог, сохраняя структуру имён."""
    settings = settings or get_settings()
    client = get_client(settings)
    target_dir = Path(local_dir)
    downloaded: list[str] = []

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(prefix) :].lstrip("/")
            downloaded.append(download_file(key, target_dir / relative, settings))
    return downloaded


def object_exists(key: str, settings: Settings | None = None) -> bool:
    """Проверяет наличие объекта в бакете."""
    settings = settings or get_settings()
    client = get_client(settings)
    try:
        client.head_object(Bucket=settings.bucket, Key=key)
    except client.exceptions.ClientError:
        return False
    return True
