from app.database.session import SessionLocal
from app.models.user import User
from app.services.users import get_by_username


def test_login_and_me(client, make_user):
    make_user("alice", "correct-horse-1", "Alice Test")
    r = client.post("/api/auth/login", json={"username": "Alice", "password": "correct-horse-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["expires_in"] > 0
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.json() == {"username": "alice", "full_name": "Alice Test"}


def test_wrong_password_and_unknown_user_look_identical(client, make_user):
    make_user("alice", "correct-horse-1")
    wrong = client.post("/api/auth/login", json={"username": "alice", "password": "nope-nope-1"})
    unknown = client.post("/api/auth/login", json={"username": "ghost", "password": "nope-nope-1"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_inactive_user_cannot_log_in(client, make_user):
    make_user("alice", "correct-horse-1")
    with SessionLocal() as db:
        get_by_username(db, "alice").is_active = False
        db.commit()
    r = client.post("/api/auth/login", json={"username": "alice", "password": "correct-horse-1"})
    assert r.status_code == 401


def test_me_requires_a_valid_token(client, auth_for):
    assert client.get("/api/auth/me").status_code == 401
    headers = auth_for("alice")
    tampered = {"Authorization": headers["Authorization"] + "x"}
    assert client.get("/api/auth/me", headers=tampered).status_code == 401


def test_password_is_stored_hashed(make_user):
    make_user("alice", "correct-horse-1")
    with SessionLocal() as db:
        user = db.query(User).filter_by(username="alice").one()
        assert "correct-horse-1" not in user.password_hash
        assert user.password_hash.startswith("$argon2")