import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import burn_verification, hash_password, verify_password
from app.models.user import User

USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,40}$")


class UserError(Exception):
    pass


def normalize_username(username: str) -> str:
    return username.strip().lower()


def get_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == normalize_username(username)))


def create_user(
    db: Session, username: str, password: str, full_name: str, *, reset: bool = False
) -> User:
    username = normalize_username(username)
    full_name = full_name.strip()
    if not USERNAME_RE.match(username):
        raise UserError("username must be 3-40 characters: letters, digits, '.', '_' or '-'")
    if len(password) < 8:
        raise UserError("password must be at least 8 characters")
    if not full_name or len(full_name) > 120:
        raise UserError("full name is required (max 120 characters)")

    existing = get_by_username(db, username)
    if existing:
        if not reset:
            raise UserError(f"user '{username}' already exists (use --reset to update)")
        existing.password_hash = hash_password(password)
        existing.full_name = full_name
        existing.is_active = True
        db.commit()
        return existing

    user = User(username=username, full_name=full_name, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    return user


def authenticate(db: Session, username: str, password: str) -> User | None:
    user = get_by_username(db, username)
    if user is None:
        burn_verification(password)
        return None
    if not verify_password(password, user.password_hash) or not user.is_active:
        return None
    return user