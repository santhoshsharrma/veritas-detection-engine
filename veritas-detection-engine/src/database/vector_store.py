import os

import psycopg
from pgvector.psycopg import register_vector


def connect():
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "veritas"),
        user=os.getenv("POSTGRES_USER", "veritas"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        autocommit=True,
    )
    register_vector(conn)
    return conn


def add_fingerprint(conn, modality: str, embedding, label: str) -> None:
    conn.execute(
        "INSERT INTO fingerprints (modality, embedding, label) VALUES (%s, %s, %s)",
        (modality, embedding, label),
    )


def nearest(conn, embedding, k: int = 5):
    return conn.execute(
        "SELECT label, embedding <=> %s AS distance FROM fingerprints "
        "ORDER BY distance LIMIT %s",
        (embedding, k),
    ).fetchall()
