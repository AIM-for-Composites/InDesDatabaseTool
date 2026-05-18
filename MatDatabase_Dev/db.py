# db.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = None
engine = None
SessionLocal = None

if all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _require_engine():
    if engine is None:
        raise RuntimeError(
            "Database is not configured. Set DB_HOST, DB_PORT, DB_NAME, DB_USER, and DB_PASSWORD."
        )

def fetch_all(query, params=None):
    _require_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return [dict(row._mapping) for row in result]

def fetch_one(query, params=None):
    _require_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        row = result.fetchone()
        return dict(row._mapping) if row else None

def execute_query(query, params=None):
    _require_engine()
    with engine.begin() as conn:
        result = conn.execute(text(query), params or {})
        return result.rowcount
