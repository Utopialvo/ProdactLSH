-- Создание таблицы datasets
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    dimension INTEGER NOT NULL,
    m INTEGER DEFAULT 100,
    k INTEGER DEFAULT 10,
    L INTEGER DEFAULT 5,
    w FLOAT DEFAULT 1.0,
    distance_metric VARCHAR(50) DEFAULT 'euclidean',
    initial_radius FLOAT,
    radius_expansion FLOAT DEFAULT 2.0,
    sampling_ratio FLOAT DEFAULT 0.1,
    quantization_method VARCHAR(50) DEFAULT 'lsh',
    pq_num_subspaces INTEGER DEFAULT 8,
    pq_num_clusters INTEGER DEFAULT 256,
    pq_use_diffusion BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы batches
CREATE TABLE IF NOT EXISTS batches (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    batch_id VARCHAR(255) NOT NULL,
    size INTEGER NOT NULL,
    features_hash VARCHAR(64),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы lsh_tables
CREATE TABLE IF NOT EXISTS lsh_tables (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    table_id INTEGER NOT NULL,
    hash_value VARCHAR(255) NOT NULL,
    point_indices TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы quantization_models
CREATE TABLE IF NOT EXISTS quantization_models (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    method VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    model_data BYTEA,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы data_points (опционально, если нужно хранить сами данные)
CREATE TABLE IF NOT EXISTS data_points (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    point_index INTEGER NOT NULL,
    data_vector JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создание индексов
CREATE INDEX IF NOT EXISTS idx_lsh_tables_dataset ON lsh_tables(dataset_id, table_id);
CREATE INDEX IF NOT EXISTS idx_batches_dataset ON batches(dataset_id);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_quantization_models_dataset ON quantization_models(dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_points_dataset ON data_points(dataset_id, point_index);

-- Вставка начальных данных (опционально)
INSERT INTO datasets (name, dimension, quantization_method) 
VALUES ('default_dataset', 10, 'lsh') 
ON CONFLICT (name) DO NOTHING;