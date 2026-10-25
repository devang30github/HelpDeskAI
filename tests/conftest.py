import os
import tempfile

# Must be set BEFORE importing the app so settings and the engine pick them up.
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["INDEX_DIR"] = f"{_tmp}/index"
os.environ["JWT_SECRET"] = "test-jwt-secret-0123456789-abcdefghijklmnop"
os.environ["GROQ_API_KEY"] = ""  # tests must never call the real LLM
os.environ["KB_DIR"] = f"{_tmp}/kb"
os.environ["WARMUP_ON_STARTUP"] = "false"
os.environ["GROQ_API_KEYS"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database.session import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"X-Admin-Key": "test-admin-key"}


@pytest.fixture(autouse=True)
def clean_db():
    import app.models  # noqa: F401

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin():
    return ADMIN


@pytest.fixture
def make_user():
    from app.services.users import create_user

    def _make(username="alice", password="correct-horse-1", full_name="Alice Test"):
        with SessionLocal() as db:
            create_user(db, username, password, full_name, reset=True)
        return username, password

    return _make


@pytest.fixture
def auth_for(client, make_user):
    """Factory: log in as a user (creating it if needed) and return auth headers."""

    def _auth(username="alice", password="correct-horse-1"):
        make_user(username, password, username.title())
        r = client.post("/api/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _auth


@pytest.fixture
def employee_headers(auth_for):
    return auth_for("alice")


@pytest.fixture
def ticket_payload():
    return {
        "title": "Wi-Fi drops every few minutes",
        "description": "My laptop disconnects from office Wi-Fi roughly every five minutes.",
        "category": "network_wifi",
        "priority": "high",
    }

@pytest.fixture(autouse=True)
def clean_kb_and_index():
    import shutil

    from app.core.config import get_settings
    from app.services.kb_admin import reindexer

    s = get_settings()
    for d in (s.kb_dir, s.index_dir):
        shutil.rmtree(d, ignore_errors=True)
    s.kb_dir.mkdir(parents=True, exist_ok=True)
    reindexer.reset()
    yield