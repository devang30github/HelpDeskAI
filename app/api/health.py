from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import get_settings
from app.rag.vector_store import index_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready(db: Session = Depends(get_db)) -> dict:
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(503, "Database not ready")
    idx = index_status(get_settings().index_dir)
    # Ticketing works without an index; chat (Stage 3) will need it, hence "degraded".
    return {"status": "ready" if idx["built"] else "degraded", "database": "ok", "index": idx}