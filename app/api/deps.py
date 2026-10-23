import secrets
from collections.abc import Iterator

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import SessionLocal


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Development-time guard only. Production needs real authentication,
    authorization and secure session handling."""
    expected = get_settings().admin_api_key
    if (
        not expected
        or not x_admin_key
        or not secrets.compare_digest(x_admin_key.encode(), expected.encode())
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Admin access required")