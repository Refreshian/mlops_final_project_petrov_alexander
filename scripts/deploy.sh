#!/usr/bin/env bash
# Выкатка сервиса на виртуалке: логин в реестр, обновление образа, проверка /health.
#
# Запускается из GitLab CI (стадия deploy) или вручную из корня репозитория:
#   bash scripts/deploy.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${CI_REGISTRY_USER:-}" && -n "${CI_REGISTRY_PASSWORD:-}" ]]; then
    echo "== Логин в $CI_REGISTRY =="
    echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin "$CI_REGISTRY"
fi

echo "== Обновление образа =="
docker compose pull

echo "== Перезапуск сервиса =="
docker compose up -d

echo "== Проверка /health =="
for attempt in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:8000/health" > /dev/null; then
        curl -s "http://127.0.0.1:8000/health"
        echo
        echo "Сервис отвечает"
        exit 0
    fi
    echo "Попытка ${attempt}: сервис ещё поднимается"
    sleep 10
done

echo "Сервис не ответил на /health"
docker compose logs --tail 50 api
exit 1
