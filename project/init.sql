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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS batches (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    batch_id VARCHAR(255) NOT NULL,
    size INTEGER NOT NULL,
    features_hash VARCHAR(64),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lsh_tables (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    table_id INTEGER NOT NULL,
    hash_value VARCHAR(255) NOT NULL,
    point_indices TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lsh_tables_dataset ON lsh_tables(dataset_id, table_id);
CREATE INDEX IF NOT EXISTS idx_batches_dataset ON batches(dataset_id);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);

-- Таблица для хранения точек (опционально, если нужно хранить сами данные)
CREATE TABLE IF NOT EXISTS data_points (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    point_index INTEGER NOT NULL,
    data_vector JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_data_points_dataset ON data_points(dataset_id, point_index);