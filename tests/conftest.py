import os
import tempfile

# Must be set BEFORE importing the app so settings and the engine pick them up.
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["ADMIN_API_KEY"] = "test-admin-key"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database.session import Base, engine  # noqa: E402
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
def ticket_payload():
    return {
        "title": "Wi-Fi drops every few minutes",
        "description": "My laptop disconnects from office Wi-Fi roughly every five minutes.",
        "category": "network_wifi",
        "priority": "high",
        "requester": "jdoe",
    }