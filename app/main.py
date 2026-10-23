import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, health, tickets
from app.core.config import get_settings
from app.database.session import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if len(settings.jwt_secret) < 32:
        logger.warning("JWT_SECRET is missing or shorter than 32 characters; employee login is disabled")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(tickets.router)