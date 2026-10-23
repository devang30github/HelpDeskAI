import secrets
from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import decode_access_token
from app.database.session import SessionLocal
from app.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Not authenticated", headers={"WWW-Authenticate": "Bearer"}
    )
    if creds is None:
        raise unauthorized
    username = decode_access_token(creds.credentials)
    if not username:
        raise unauthorized
    user = db.scalar(select(User).where(User.username == username))
    if user is None or not user.is_active:
        raise unauthorized
    return user


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