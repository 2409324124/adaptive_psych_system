from __future__ import annotations

from datetime import datetime
from pathlib import Path
import uuid

from sqlalchemy import JSON, Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = f"sqlite:///{(ROOT / 'data' / 'cat_psych.db').as_posix()}"

Base = declarative_base()


class UserSessionRecord(Base):
    __tablename__ = "user_sessions"

    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    ocean_scores = Column(JSON)
    cat_category = Column(String(100))
    llm_analysis = Column(Text)
    raw_responses = Column(JSON)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
