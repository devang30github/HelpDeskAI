import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, chat, health, tickets
from app.core.config import get_settings
from app.database.session import init_db
from app.rag.generation import LLMError
from app.rag.vector_store import IndexCorrupt, IndexMismatch, IndexNotFound

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if len(settings.jwt_secret) < 32:
        logger.warning("JWT_SECRET is missing or shorter than 32 characters; employee login is disabled")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    logger.error("LLM unavailable: %s", exc)  # message contains no keys or prompts
    return JSONResponse(
        status_code=503,
        content={"detail": "The assistant is temporarily unavailable. Please try again in a moment."},
    )


async def index_error_handler(request: Request, exc: Exception):
    logger.error("Knowledge base index problem: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The knowledge base is not ready yet. Please contact the helpdesk administrator."},
    )


for _exc in (IndexNotFound, IndexMismatch, IndexCorrupt):
    app.add_exception_handler(_exc, index_error_handler)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(tickets.router)
app.include_router(chat.router)

app.mount("/employee", StaticFiles(directory=STATIC_DIR / "employee", html=True), name="employee")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/employee/")