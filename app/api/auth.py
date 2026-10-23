from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import get_settings
from app.core.security import create_access_token
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenOut, UserOut
from app.services.users import authenticate

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenOut)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate(db, payload.username, payload.password)
    if user is None:
        # Same message for unknown user, wrong password and disabled account.
        raise HTTPException(401, "Invalid username or password")
    return TokenOut(
        access_token=create_access_token(user.username),
        expires_in=get_settings().jwt_expire_minutes * 60,
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user