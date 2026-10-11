# Финальный проект MLOps: сервис оценки вероятности покупки

[![CI](https://github.com/Refreshian/mlops_final_project_petrov_alexander/actions/workflows/ci.yml/badge.svg)](https://github.com/Refreshian/mlops_final_project_petrov_alexander/actions)

Сервис отвечает на вопрос «какова вероятность, что товар купят после просмотра»
с учётом часа и дня недели. В репозитории собран полный цикл: подготовка данных
в Airflow, обучение и подбор гиперпараметров в ClearML, FastAPI-сервис
предсказаний, выкатка на виртуалку через GitLab CI и мониторинг в
Prometheus + Loki + Grafana.

Код опубликован на GitHub: https://github.com/Refreshian/mlops_final_project_petrov_alexander. CI/CD настроен под GitLab стенда курса (`.gitlab-ci.yml`), репозиторий на GitHub — зеркало для сдачи работы.

## Задача и данные

Датасет RetailRocket — логи интернет-магазина:

| Файл | Что внутри |
| --- | --- |
| `events.csv` | события посетителей: `view`, `addtocart`, `transaction` |
| `item_properties_part1.csv`, `item_properties_part2.csv` | свойства товаров: категория, цена, наличие |
| `category_tree.csv` | дерево категорий |

Из событий собираются два артефакта:

* **`item_features.parquet`** — справочник фичей товаров: `views`, `purchases`,
  `ctr` (накопленная популярность) и свойства `categoryid`, `available`.
  По нему работает сервис;
* **`data_for_training.parquet`** — обучающая выборка: строка на каждый просмотр
  товара, `target = 1`, если тот же посетитель купил этот товар в течение
  24 часов после просмотра.

Модель — CatBoost на признаках
`views, purchases, ctr, hour, weekday, categoryid, available`.
Главная метрика — **PR-AUC**: положительный класс редкий (меньше процента
просмотров заканчиваются покупкой), поэтому accuracy и ROC-AUC обманчивы.
Выборка делится на train / validation / test: на validation работает ранняя
остановка, метрики считаются на отведённом test.

## Структура репозитория

```
.
├── config/                        конфиги инфраструктуры
│   ├── prometheus/                сбор метрик и правила алертов
│   ├── grafana/                   датасорсы и дашборд
│   ├── loki/, promtail/           сбор логов контейнеров
│   └── k8s/                       deployment, service, secret, serviceaccount
├── src/                           код приложения
│   ├── main.py                    FastAPI-сервис
│   ├── model.py                   загрузка артефактов из S3 и предсказание
│   ├── features.py                фичи товаров и обучающая выборка
│   ├── prepare_features.py        CLI подготовки данных
│   ├── train_model.py             CLI обучения CatBoost
│   ├── validation.py              Pandera-схемы данных
│   ├── metrics.py                 метрики Prometheus
│   ├── schemas.py                 схемы ответов
│   ├── config.py, storage.py      настройки и работа с S3
├── mlops/
│   ├── airflow/                   DAG предобработки и docker-compose Airflow
│   └── clearml/                   пайплайн обучения и его шаги
├── tests/                         unit-тесты
├── scripts/                       deploy.sh и вспомогательные скрипты
├── Dockerfile, build.sh           образ сервиса
├── docker-compose.yaml            сервис на виртуалке
├── docker-compose.monitoring.yml  Prometheus, Loki, Grafana, node-exporter
├── .gitlab-ci.yml                 lint, formating, test, build, save, deploy
├── pyproject.toml                 зависимости и настройки ruff, black, mypy, pytest
└── requirements.txt               зависимости (requirements-dev.txt — линтеры и тесты)
```

## Хранение данных в S3

Данных в репозитории нет: и исходные файлы, и артефакты лежат в бакете.

```
s3://<bucket>/petrov_alexander_project/raw_data/         исходные файлы RetailRocket
s3://<bucket>/petrov_alexander_project/processed_data/   item_features.parquet, data_for_training.parquet
s3://<bucket>/petrov_alexander_project/models/           ranker.pkl
```

| Переменная | Назначение | Значение по умолчанию |
| --- | --- | --- |
| `S3_BUCKET` | бакет с данными | `r-mlops-bucket-9-1-1-65671054` |
| `S3_PREFIX` | префикс проекта | `petrov_alexander_project` |
| `S3_ENDPOINT_URL` | endpoint S3-совместимого хранилища | `https://storage.yandexcloud.net` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | ключи доступа | не заданы |
| `MODEL_DIR` | каталог с артефактами в контейнере | `/app/artifacts` |
| `LOG_LEVEL` | уровень логирования сервиса | `INFO` |

## Окружение

Проект рассчитан на Python 3.12, зависимости ставятся через [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Refreshian/mlops_final_project_petrov_alexander.git
cd mlops_final_project_petrov_alexander

uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements-dev.txt
```

## Шаг 1 — подготовка фичей

Исходные файлы (их можно взять из стартового репозитория курса) кладём в
`ecom_raw/`, дальше одной командой считаем фичи и обучающую выборку:

```bash
python -m src.prepare_features \
  --events ecom_raw/events.csv \
  --properties ecom_raw/item_properties_part1.csv ecom_raw/item_properties_part2.csv
```

Если исходники уже в S3, проще так — данные скачаются, а результат выгрузится обратно:

```bash
python -m src.prepare_features --from-s3 --upload
```

На выходе — `data/processed/item_features.parquet` и `data/processed/data_for_training.parquet`.
Полезные ключи: `--window-hours` (окно покупки, по умолчанию 24 часа) и
`--sample` (ограничить число просмотров).

## Шаг 2 — обучение модели

```bash
python -m src.train_model --sample data/processed/data_for_training.parquet --upload
```

Скрипт делит выборку на train/validation/test, обучает CatBoost с ранней
остановкой по PR-AUC, печатает метрики и сохраняет `ranker.pkl`
(модель + метрики + версия + список признаков). С `--upload` модель уезжает в S3.
Гиперпараметры можно переопределить ключами `--iterations`, `--depth`,
`--learning-rate`, `--l2-leaf-reg` — этим пользуется пайплайн ClearML.

## Результаты

Обучение на полном датасете: 2 756 101 событие, 234 844 товара в справочнике,
2 664 312 строк в обучающей выборке (положительных 1.27%).

| Метрика | Значение |
| --- | --- |
| PR-AUC на отложенной выборке | 0.116 |
| ROC-AUC | 0.918 |
| Precision при пороге 0.093 | 0.134 |
| Recall при пороге 0.093 | 0.275 |

Случайный ранкер на такой выборке даёт PR-AUC ≈ 0.013 (он равен доле
положительных примеров), то есть модель примерно в девять раз лучше случайной.
Порог вероятности подбирается по валидации по максимуму F1: при пороге 0.5
модель почти ничего не относит к покупкам, и precision с recall выглядят
бессмысленно.

## Шаг 3 — сервис предсказаний

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/predict_purchase?itemid=&hour=&weekday=` | вероятность покупки товара |
| `GET` | `/health` | состояние сервиса и модели (503, если модель не загружена) |
| `GET` | `/version` | версии сервиса и модели |
| `GET` | `/metrics` | метрики Prometheus |
| `GET` | `/docs` | интерактивная документация |

Локальный запуск (модель и справочник фичей сервис ищет в `MODEL_DIR`, при
отсутствии — скачивает из S3):

```bash
MODEL_DIR=artifacts python -m src.main
# или
MODEL_DIR=artifacts uvicorn src.main:app --host 0.0.0.0 --port 32000
```

```bash
curl "http://127.0.0.1:32000/predict_purchase?itemid=356475&hour=14&weekday=3"
```

```json
{
  "itemid": 356475,
  "hour": 14,
  "weekday": 3,
  "purchase_probability": 0.018192,
  "item_found": true,
  "model_version": "1.0.0"
}
```

Если товара нет в справочнике, сервис отвечает 200 и считает по значениям по
умолчанию, а в ответе ставит `item_found: false`. Такие запросы видны в метрике
`predict_item_lookup_total`.

## Тесты и качество кода

```bash
pytest
ruff check .
black --check .
mypy
```

Тесты работают на небольшом сэмпле реальных данных из `tests/data`: модель
обучается прямо в фикстурах, поэтому ни S3, ни заранее собранные артефакты для
прогона не нужны.

## Airflow

DAG `preprocessing_data` (`mlops/airflow/dags/preprocessing_data.py`) раз в сутки
забирает исходные файлы из S3, проверяет схему событий Pandera, собирает фичи
товаров и обучающую выборку, проверяет их схемами и выгружает в S3:

```
check_raw_files_in_s3 -> download_raw_files -> process_files
    -> upload_processed_data -> check_processed_files_in_s3
```

Запуск Airflow 3 на виртуалке:

```bash
cd mlops/airflow
docker compose up -d
```

Интерфейс — `http://<ip-виртуалки>:8080`. Репозиторий монтируется в контейнеры
целиком (`/opt/mlops_project`), поэтому DAG импортирует пакет `src` из него же.

Доступ к S3 хранится в соединении Airflow, а не в коде DAG. Соединение `s3_conn`
(Admin → Connections, тип Amazon Web Services) создаётся один раз:

```bash
docker compose exec airflow-apiserver airflow connections add s3_conn \
  --conn-type aws \
  --conn-login "$AWS_ACCESS_KEY_ID" \
  --conn-password "$AWS_SECRET_ACCESS_KEY" \
  --conn-extra '{"endpoint_url": "https://storage.yandexcloud.net"}'
```

## ClearML

Пайплайн `mlops/clearml/clearml_pipeline.py`:

```
load_data ──┬─> train_catboost_shallow ──┐
            ├─> train_catboost_medium ───┤
            ├─> train_catboost_deep ─────┼─> select_best_model
            └─> tune_hyperparameters ────┘
```

* `load_data` скачивает обучающую выборку из S3 и делит её на train/validation/test;
* три шага `train_catboost_*` обучают модели с разными гиперпараметрами
  (глубина 4/6/8, learning rate 0.08/0.05/0.03, разная L2-регуляризация);
* `tune_hyperparameters` подбирает гиперпараметры через Optuna по PR-AUC на
  валидации, метрики каждого испытания видны в графиках ClearML;
* `select_best_model` сравнивает PR-AUC всех моделей и выгружает лучшую в S3 как `ranker.pkl`.

Запуск (агент ClearML должен быть поднят на той же машине, ключи S3 — в окружении):

```bash
python mlops/clearml/clearml_pipeline.py
```

Результаты и сравнение экспериментов — в веб-интерфейсе ClearML, проект
`Ecommerce Purchase Predictor`.

## Выкатка на виртуалку

На виртуалке достаточно Docker и docker compose:

```bash
git clone https://github.com/Refreshian/mlops_final_project_petrov_alexander.git
cd mlops_final_project_petrov_alexander
cp .env.example .env && nano .env        # образ из реестра и ключи S3
docker compose up -d
```

Сервис сам скачивает `ranker.pkl` и `item_features.parquet` из S3 при старте.
Контейнер слушает порт 32000, на хост он проброшен как 8000 и 32000, поэтому
проверка из задания работает как есть:

```bash
curl "http://127.0.0.1:8000/predict_purchase?itemid=356475&hour=14&weekday=3"
```

Стадия `deploy` в `.gitlab-ci.yml` делает то же самое автоматически:
синхронизирует репозиторий на виртуалку, собирает `.env` из CI/CD Variables,
запускает `scripts/deploy.sh` (обновление образа, перезапуск, ожидание `/health`).

### Kubernetes (дополнительно)

Манифесты в `config/k8s` повторяют это поведение: init-контейнер `model-loader`
запускает загрузчик сервиса и складывает артефакты в общий том, основной
контейнер поднимает API.

```bash
kubectl create secret generic s3-credentials -n <namespace> \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
kubectl apply -f config/k8s/

kubectl port-forward svc/purchase-predictor 32000:32000
curl "http://127.0.0.1:32000/predict_purchase?itemid=356475&hour=14&weekday=3"
```

## Мониторинг

```bash
docker network create mlops
GRAFANA_ADMIN_PASSWORD=... docker compose -f docker-compose.monitoring.yml up -d
```

* Prometheus (`:9090`) собирает метрики сервиса с `api:32000/metrics`;
* Promtail читает логи контейнеров и отправляет их в Loki (`:3100`);
* Grafana (`:3000`) с провижинингом датасорсов и дашбордом
  «Purchase Predictor: сервис предсказаний»;
* node-exporter (`:9100`) отдаёт метрики виртуалки.

Метрики сервиса: `predict_requests_total`, `predict_latency_seconds`,
`predict_purchase_probability`, `predict_item_lookup_total`, `model_loaded` и
`model_pr_auc` (качество загруженной модели).

Алерты — в `config/prometheus/alert.rules.yml`: `ServiceDown`, `ModelNotLoaded`,
`HighErrorRate`, `HighLatency`, `ManyUnknownItems`. Их состояние видно на
странице Prometheus → Alerts.

## CI/CD

| Стадия | Что делает |
| --- | --- |
| `lint` | `ruff check .` и `mypy` |
| `formating` | `black --check .` |
| `test` | `pytest` с отчётом в JUnit |
| `build` | сборка образа через buildkit с кэшем в реестре |
| `save` | выгрузка образа в GitLab Container Registry |
| `deploy` | выкатка на виртуалку и проверка эндпоинта (только основная ветка) |

Секреты лежат в CI/CD Variables с флагом Masked: `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `VM_HOST`, `VM_USER`, `SSH_PRIVATE_KEY`,
`GRAFANA_ADMIN_PASSWORD`. В репозитории секретов нет: `.env` в `.gitignore`,
а на виртуалке он собирается стадией `deploy`.

## Артефакты

| Артефакт | Где лежит | Что содержит |
| --- | --- | --- |
| `item_features.parquet` | `s3://<bucket>/petrov_alexander_project/processed_data/` | фичи товаров для сервиса |
| `data_for_training.parquet` | там же | обучающая выборка |
| `ranker.pkl` | `s3://<bucket>/petrov_alexander_project/models/` | модель, метрики, версия, список признаков |
| эксперименты и метрики | ClearML, проект `Ecommerce Purchase Predictor` | параметры, метрики, сравнение моделей |
| образ сервиса | GitLab Container Registry | тег коммита и `latest` |
