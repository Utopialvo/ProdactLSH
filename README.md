# FastRoLSH

Высокопроизводительная система приближенного поиска ближайших соседей с использованием комбинированного подхода FastLSH, roLSH и Product Quantization.

## О проекте

FastRoLSH — это система для эффективного приближенного поиска ближайших соседей в высокоразмерных пространствах. Она объединяет три современных подхода:
- **FastLSH**: уменьшает вычислительную сложность через интеллектуальное семплирование признаков
- **roLSH**: оптимизирует процесс поиска через адаптивный выбор радиуса
- **Product Quantization**: обеспечивает эффективное сжатие данных и ускорение поиска

Система поддерживает три режима работы (LSH-only, PQ-only, Hybrid), GPU-ускорение и интегрируется с PostgreSQL для хранения данных. Предоставляет REST API для легкой интеграции в различные приложения.

## Основные возможности

- Поддержка двух метрик расстояния: евклидовой и косинусной
- Три режима работы: LSH-only, PQ-only и гибридный режим
- Инкрементальное обновление индексов при добавлении новых данных
- Стратифицированное семплирование данных на основе LSH-бакетов
- Автоматическая оптимизация параметров на основе характеристик данных
- Поддержка диффузионных карт для улучшения качества квантования
- Комплексный REST API для управления датасетами, обработки запросов и мониторинга

## Установка и запуск

### Предварительные требования

- Python 3.10+
- PostgreSQL 16+
- Docker и Docker Compose (опционально, для запуска в контейнерах)
- Torch
- numpy
- scipy
- joblib
- scikit-learn
- matplotlib

### Установка

1. Клонируйте репозиторий:
```bash
git clone <https://github.com/Utopialvo/ProdactLSH.git>
cd fastrolsh
```

### Установите зависимости:

```bash
pip install -r requirements.txt
```
Настройте базу данных PostgreSQL и обновите строку подключения в переменной окружения DATABASE_URL
Инициализируйте базу данных, выполнив SQL-скрипт init.sql

### Запуск с помощью Docker

В проекте предоставлен docker-compose.yml для удобного запуска:

```bash
docker-compose up -d
```

Это запустит:
    Сервер PostgreSQL на порту 5432
    FastAPI приложение на порту 8000

### Использование
Создание датасета с поддержкой квантования

```bash

curl -X POST "http://localhost:8000/datasets/" \
-H "Content-Type: application/json" \
-d '{
  "name": "my_dataset",
  "dimension": 100,
  "m": 100,
  "k": 10,
  "L": 5,
  "w": 1.0,
  "distance_metric": "euclidean",
  "initial_radius": null,
  "radius_expansion": 2.0,
  "sampling_ratio": 0.1,
  "quantization_method": "hybrid",
  "pq_num_subspaces": 8,
  "pq_num_clusters": 256,
  "pq_use_diffusion": false
}'

```

Добавление данных

```bash

curl -X POST "http://localhost:8000/batches/" \
-H "Content-Type: application/json" \
-d '{
  "dataset_name": "my_dataset",
  "batch_id": "batch_1",
  "data": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
}'

```

Поиск ближайших соседей

```bash

curl -X POST "http://localhost:8000/query/" \
-H "Content-Type: application/json" \
-d '{
  "dataset_name": "my_dataset",
  "queries": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "k": 10
}'
```

Обучение модели квантования

```bash

curl -X POST "http://localhost:8000/quantization/train/" \
-H "Content-Type: application/json" \
-d '{
  "dataset_name": "my_dataset",
  "method": "pq",
  "use_diffusion": false
}'
```


Семплирование данных

```bash

curl -X POST "http://localhost:8000/sample/" \
-H "Content-Type: application/json" \
-d '{
  "dataset_name": "my_dataset",
  "strategy": "proportional",
  "size": 1000
}'
```

### API endpoints

    GET / — информация о API

    GET /health — проверка здоровья сервера и базы данных

    POST /datasets/ — создание нового датасета

    POST /batches/ — обработка батча данных

    POST /query/ — поиск ближайших соседей

    POST /sample/ — семплирование данных

    GET /datasets/ — список всех датасетов

    GET /model/state/{dataset_name} — состояние модели

    POST /model/save/ — сохранение состояния модели в файл

    POST /model/load/ — загрузка состояния модели из файл

    GET /datasets/{dataset_name}/info — информация о датасете

    GET /datasets/{dataset_name}/batches/{batch_id} — информация о батче

    POST /model/optimize/{dataset_name} — оптимизация параметров модели

    POST /quantization/train/ — обучение модели квантования

### Примеры использования

Примеры тестирования системы находятся в файле test_file.py. Они демонстрируют:

    Тестирование поиска ближайших соседей с разными методами квантования

    Тестирование кластеризации на семплированных данных

    Тестирование классификации на семплированных данных

    Тестирование регрессии на семплированных данных

    Тестирование различных метрик расстояния

    Тестирование различных параметров LSH и PQ

    Тестирование оптимизации параметров

### Структура проекта
    docs — Описание к проекту и статьи на которых он строится
    project/ — Папка с файлами проекта 

    project/main.py — основное FastAPI приложение
    project/fast_rolsh_sampler.py — реализация алгоритмов FastLSH и roLSH
    project/product_quantizer.py — реализация продуктного квантования
    project/unified_quantization.py — обобщающая реализация
    
    project/test_file.py — тесты и примеры использования
    project/init.sql — скрипт инициализации базы данных

    project/docker-compose.yml — конфигурация Docker Compose
    project/Dockerfile — Dockerfile

    project/requirements.txt — зависимости Python


### Примечание о качестве кода:
Рефакторинг кода, написание комментариев и докстрингов, а также форматирование LaTeX-документации в первоначальной версии проекта происходило с использованием генеративных моделей. Это могло повлиять на качество и стиль кода. Рекомендуется внимательно проверять код перед использованием.


###  Лицензия:
Проект распространяется под лицензией MIT. Подробнее см. в файле LICENSE.