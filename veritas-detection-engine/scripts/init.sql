CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS fingerprints (
    id BIGSERIAL PRIMARY KEY,
    modality TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    label TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
