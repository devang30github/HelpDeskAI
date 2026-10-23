import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, tickets
from app.core.config import get_settings
from app.database.session import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health.router)
app.include_router(tickets.router)