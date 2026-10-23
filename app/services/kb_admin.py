"""Admin-side knowledge-base management: safe uploads, listing, deletion and
background re-indexing. Nothing here can put the index into a state where a
rebuild fails because of a bad file."""
import hashlib
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.rag.ingestion import (
    VALID_CATEGORIES,
    build_index,
    check_index,
    doc_id_for,
    list_kb_files,
    parse_header,
    read_document_text,
)
from app.rag.vector_store import index_status, read_manifest

logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class KBError(Exception):
    status_code = 400


class InvalidUpload(KBError):
    status_code = 422


class UnsupportedType(KBError):
    status_code = 415


class UploadTooLarge(KBError):
    status_code = 413


class UploadConflict(KBError):
    status_code = 409


class DocumentNotFound(KBError):
    status_code = 404


class ReindexBusy(KBError):
    status_code = 409


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_filename(raw: str) -> str:
    base = Path(raw.replace("\\", "/")).name  # drop any directory components
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise UnsupportedType(f"unsupported file type '{ext or base}'; allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._").lower()
    if not stem:
        raise InvalidUpload("the file name is empty after cleaning")
    if stem.startswith("readme"):
        raise InvalidUpload("README files are not indexed; choose another name")
    return f"{stem[:100]}{ext}"


def _check_content_type(ext: str, data: bytes) -> None:
    if ext == ".pdf" and not data.startswith(b"%PDF-"):
        raise InvalidUpload("the file is not a valid PDF")
    if ext == ".docx" and not data.startswith(b"PK\x03\x04"):
        raise InvalidUpload("the file is not a valid DOCX")
    if ext in (".txt", ".md"):
        if b"\x00" in data:
            raise InvalidUpload("the text file contains binary data")
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise InvalidUpload("text files must be UTF-8 encoded")


def save_upload(filename: str, data: bytes, replace: bool = False) -> dict:
    s = get_settings()
    kb_dir = Path(s.kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(s.kb_max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise UploadTooLarge(f"the file is larger than the {s.kb_max_upload_mb:g} MB limit")
    if not data:
        raise InvalidUpload("the file is empty")

    name = sanitize_filename(filename or "")
    ext = Path(name).suffix
    _check_content_type(ext, data)

    doc_id = doc_id_for(name)
    replaced = False
    for existing in list_kb_files(kb_dir):
        if doc_id_for(existing.name) != doc_id:
            continue
        if existing.name == name:
            if not replace:
                raise UploadConflict(f"'{name}' already exists; tick 'Replace if it exists' to overwrite it")
            replaced = True
        else:
            raise UploadConflict(f"document id '{doc_id}' is already used by '{existing.name}'")

    # Write to a hidden temp name (the indexer ignores dot-files), validate, then rename into place.
    tmp = kb_dir / f".upload-{uuid.uuid4().hex}{ext}"
    try:
        tmp.write_bytes(data)
        try:
            text = read_document_text(tmp)
        except Exception as e:
            logger.warning("upload %s could not be read: %s", name, type(e).__name__)
            raise InvalidUpload("the document could not be read; it may be corrupt or password-protected")
        meta, body = parse_header(text)
        if not body:
            raise InvalidUpload("no extractable text found (scanned PDFs without a text layer are not supported)")
        os.replace(tmp, kb_dir / name)
    finally:
        tmp.unlink(missing_ok=True)

    warnings = []
    if "title" not in meta:
        warnings.append("No 'Title:' header found; the file name will be used as the title.")
    if "category" not in meta:
        warnings.append("No 'Category:' header found; the document will be indexed without a category.")
    elif meta["category"] not in VALID_CATEGORIES:
        warnings.append(f"Unknown category '{meta['category']}'; it will be ignored.")

    logger.info("kb upload saved: %s (%d bytes, replaced=%s)", name, len(data), replaced)
    return {
        "filename": name,
        "doc_id": doc_id,
        "size_bytes": len(data),
        "title": meta.get("title"),
        "category": meta.get("category"),
        "words": len(body.split()),
        "replaced": replaced,
        "warnings": warnings,
    }


def delete_document(filename: str) -> None:
    if filename != Path(filename).name:  # no path components, ever
        raise DocumentNotFound(f"'{filename}' was not found")
    for path in list_kb_files(get_settings().kb_dir):
        if path.name == filename:
            path.unlink()
            logger.info("kb document deleted: %s", filename)
            return
    raise DocumentNotFound(f"'{filename}' was not found")


def kb_overview() -> dict:
    s = get_settings()
    Path(s.kb_dir).mkdir(parents=True, exist_ok=True)
    files = {p.name: p for p in list_kb_files(s.kb_dir)}

    status = index_status(s.index_dir)
    manifest = read_manifest(s.index_dir) if status["built"] else {}
    indexed_files = manifest.get("files", {})
    indexed_docs = manifest.get("documents", {})

    documents = []
    for name, path in sorted(files.items()):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if name not in indexed_files:
            state = "new"
        elif indexed_files[name] != sha:
            state = "changed"
        else:
            state = "indexed"
        info = indexed_docs.get(doc_id_for(name), {})
        stat = path.stat()
        documents.append({
            "filename": name,
            "doc_id": doc_id_for(name),
            "state": state,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "title": info.get("title"),
            "category": info.get("category"),
            "num_chunks": info.get("num_chunks"),
        })
    for name in sorted(indexed_files.keys() - files.keys()):  # indexed earlier, deleted from disk since
        documents.append({
            "filename": name, "doc_id": doc_id_for(name), "state": "removed", "size_bytes": None,
            "modified_at": None, "title": None, "category": None, "num_chunks": None,
        })

    stale, reasons = check_index(s)
    return {
        "documents": documents,
        "index": status,
        "stale": stale,
        "stale_reasons": reasons,
        "reindex": reindexer.status(),
        "limits": {"max_upload_mb": s.kb_max_upload_mb, "extensions": sorted(ALLOWED_UPLOAD_EXTENSIONS)},
    }


class ReindexManager:
    """Runs index rebuilds in a background thread, one at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {"state": "idle", "started_at": None, "finished_at": None, "error": None, "result": None}

    def reset(self) -> None:
        with self._lock:
            self._status = self._idle()

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def start(self) -> dict:
        with self._lock:
            if self._status["state"] == "running":
                raise ReindexBusy("a rebuild is already running")
            self._status = {**self._idle(), "state": "running", "started_at": _iso_now()}
        threading.Thread(target=self._run, daemon=True, name="kb-reindex").start()
        return self.status()

    def _finish(self, state: str, error: str | None = None, result: dict | None = None) -> None:
        with self._lock:
            self._status.update(state=state, finished_at=_iso_now(), error=error, result=result)

    def _run(self) -> None:
        try:
            manifest = build_index()  # looked up at call time so tests can substitute it
            self._finish("succeeded", result={
                "num_documents": manifest["num_documents"],
                "num_chunks": manifest["num_chunks"],
                "built_at": manifest["built_at"],
            })
        except Exception as e:
            logger.exception("index rebuild failed")
            self._finish("failed", error=str(e) or type(e).__name__)


reindexer = ReindexManager()