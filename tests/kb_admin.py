import hashlib
import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from docx import Document

from app.core.config import get_settings
from app.rag.types import Chunk
from app.rag.vector_store import VectorStore
from app.services import kb_admin

GOOD = b"Title: VPN Guide\nCategory: network_wifi\n\nRestart the VPN client and try the backup gateway.\n"


def upload(client, admin, name, data, replace=False):
    return client.post(
        "/api/admin/kb/documents",
        files={"file": (name, data, "application/octet-stream")},
        data={"replace": "true" if replace else "false"},
        headers=admin,
    )


def _docx_bytes(*paragraphs):
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _build_fake_index(files: dict[str, bytes]):
    s = get_settings()
    chunks = [
        Chunk(chunk_id=f"{Path(n).stem}::chunk-000", doc_id=kb_admin.doc_id_for(n), source_name=n,
              chunk_index=0, text="t", metadata={"title": "T"})
        for n in files
    ]
    manifest = {
        "embedding_model": s.embedding_model, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
        "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
        "documents": {kb_admin.doc_id_for(n): {"source_name": n, "title": "T", "category": None, "num_chunks": 1}
                      for n in files},
        "num_documents": len(files), "built_at": "2026-01-01T00:00:00+00:00",
    }
    VectorStore.build(chunks, np.eye(len(chunks), 4, dtype="float32"), manifest).save(s.index_dir)


def test_kb_routes_require_admin(client):
    assert client.get("/api/admin/kb").status_code == 401
    assert client.post("/api/admin/kb/documents", files={"file": ("a.txt", GOOD)}).status_code == 401
    assert client.post("/api/admin/kb/reindex").status_code == 401


def test_upload_txt_and_list(client, admin):
    r = upload(client, admin, "VPN Guide.txt", GOOD)
    assert r.status_code == 201
    body = r.json()
    assert body["filename"] == "vpn-guide.txt" and body["doc_id"] == "vpn-guide"
    assert body["category"] == "network_wifi" and body["warnings"] == []
    assert (get_settings().kb_dir / "vpn-guide.txt").exists()

    docs = client.get("/api/admin/kb", headers=admin).json()["documents"]
    assert [(d["filename"], d["state"]) for d in docs] == [("vpn-guide.txt", "new")]


def test_duplicate_needs_replace(client, admin):
    assert upload(client, admin, "guide.txt", GOOD).status_code == 201
    assert upload(client, admin, "guide.txt", GOOD).status_code == 409
    r = upload(client, admin, "guide.txt", GOOD + b"Extra line here.\n", replace=True)
    assert r.status_code == 200 and r.json()["replaced"] is True


def test_document_id_collision_is_rejected(client, admin):
    assert upload(client, admin, "a-b.txt", GOOD).status_code == 201
    r = upload(client, admin, "a_b.txt", GOOD)
    assert r.status_code == 409 and "already used" in r.json()["detail"]


@pytest.mark.parametrize(
    "name,data,status",
    [
        ("tool.exe", b"MZ....", 415),
        ("fake.pdf", b"this is plain text", 422),
        ("fake.docx", b"not a zip file at all", 422),
        ("empty.txt", b"", 422),
        ("headeronly.txt", b"Title: Only a header\nCategory: network_wifi\n", 422),
        ("binary.txt", b"hello\x00world", 422),
        ("README.md", GOOD, 422),
    ],
)
def test_bad_uploads_are_rejected(client, admin, name, data, status):
    assert upload(client, admin, name, data).status_code == status
    assert list(get_settings().kb_dir.iterdir()) == []  # nothing left behind, not even temp files


def test_path_traversal_is_neutralised(client, admin):
    r = upload(client, admin, "../../evil.txt", GOOD)
    assert r.status_code == 201 and r.json()["filename"] == "evil.txt"
    kb_dir = get_settings().kb_dir
    assert (kb_dir / "evil.txt").exists()
    assert not (kb_dir.parent / "evil.txt").exists()
    with pytest.raises(kb_admin.DocumentNotFound):
        kb_admin.delete_document("../evil.txt")


def test_size_limit(client, admin, monkeypatch):
    monkeypatch.setattr(get_settings(), "kb_max_upload_mb", 0.0001)  # about 100 bytes
    assert upload(client, admin, "big.txt", GOOD * 5).status_code == 413


def test_docx_upload(client, admin):
    data = _docx_bytes("Title: Printing Guide", "Category: printing_office_equipment", "Clear the paper jam in tray two.")
    r = upload(client, admin, "Printing Guide.docx", data)
    assert r.status_code == 201
    assert r.json()["doc_id"] == "printing-guide" and r.json()["category"] == "printing_office_equipment"


def test_pdf_upload(client, admin):
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, "Title: Facilities Guide\nCategory: facilities_maintenance\n\nReport leaks immediately.")
    r = upload(client, admin, "facilities.pdf", bytes(pdf.output()))
    assert r.status_code == 201 and r.json()["doc_id"] == "facilities"


def test_missing_headers_produce_warnings(client, admin):
    r = upload(client, admin, "notes.txt", b"Just a plain paragraph of text with no headers at all.")
    assert r.status_code == 201
    assert len(r.json()["warnings"]) == 2


def test_delete_document(client, admin):
    upload(client, admin, "guide.txt", GOOD)
    assert client.delete("/api/admin/kb/documents/guide.txt", headers=admin).status_code == 204
    assert client.get("/api/admin/kb", headers=admin).json()["documents"] == []
    assert client.delete("/api/admin/kb/documents/guide.txt", headers=admin).status_code == 404


def test_overview_states_follow_the_index(client, admin):
    assert upload(client, admin, "vpn-guide.txt", GOOD).status_code == 201
    overview = client.get("/api/admin/kb", headers=admin).json()
    assert overview["index"]["built"] is False and overview["stale"] is True
    assert [d["state"] for d in overview["documents"]] == ["new"]

    _build_fake_index({"vpn-guide.txt": GOOD})
    overview = client.get("/api/admin/kb", headers=admin).json()
    assert [d["state"] for d in overview["documents"]] == ["indexed"] and overview["stale"] is False

    assert upload(client, admin, "vpn-guide.txt", GOOD + b"More detail.\n", replace=True).status_code == 200
    overview = client.get("/api/admin/kb", headers=admin).json()
    assert [d["state"] for d in overview["documents"]] == ["changed"] and overview["stale"] is True

    client.delete("/api/admin/kb/documents/vpn-guide.txt", headers=admin)
    overview = client.get("/api/admin/kb", headers=admin).json()
    assert [d["state"] for d in overview["documents"]] == ["removed"]


def _wait_for_reindex(client, admin, done, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/admin/kb/reindex", headers=admin).json()
        if done(status):
            return status
        time.sleep(0.05)
    raise AssertionError("reindex did not finish in time")


def test_reindex_runs_in_background_and_blocks_concurrent_starts(client, admin, monkeypatch):
    gate = threading.Event()

    def fake_build():
        gate.wait(timeout=5)
        return {"num_documents": 2, "num_chunks": 5, "built_at": "now"}

    monkeypatch.setattr(kb_admin, "build_index", fake_build)
    first = client.post("/api/admin/kb/reindex", headers=admin)
    assert first.status_code == 202 and first.json()["state"] == "running"
    assert client.post("/api/admin/kb/reindex", headers=admin).status_code == 409

    gate.set()
    status = _wait_for_reindex(client, admin, lambda s: s["state"] != "running")
    assert status["state"] == "succeeded" and status["result"]["num_chunks"] == 5


def test_reindex_failure_is_reported(client, admin, monkeypatch):
    def failing_build():
        raise ValueError("Duplicate doc_id 'x'")

    monkeypatch.setattr(kb_admin, "build_index", failing_build)
    client.post("/api/admin/kb/reindex", headers=admin)
    status = _wait_for_reindex(client, admin, lambda s: s["state"] != "running")
    assert status["state"] == "failed" and "Duplicate" in status["error"]
    # a failed run does not block the next attempt
    assert client.post("/api/admin/kb/reindex", headers=admin).status_code == 202
    _wait_for_reindex(client, admin, lambda s: s["state"] != "running")