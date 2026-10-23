import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_db, require_admin
from app.core.config import get_settings
from app.models.conversation import Message
from app.schemas.admin import StatsOut
from app.schemas.chat import MessageOut
from app.services import kb_admin
from app.tickets import service as tickets_service

# Every route here requires the dev-only admin key.
router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db)):
    return tickets_service.ticket_stats(db)


@router.get("/tickets/{ticket_id}/conversation", response_model=list[MessageOut])
def ticket_conversation(ticket_id: str, db: Session = Depends(get_db)):
    try:
        ticket = tickets_service.get_ticket(db, ticket_id)
    except tickets_service.TicketNotFound:
        raise HTTPException(404, "Ticket not found")
    if not ticket.conversation_id:
        return []
    rows = db.scalars(
        select(Message).where(Message.conversation_id == ticket.conversation_id).order_by(Message.id)
    )
    out = []
    for m in rows:
        meta = json.loads(m.meta_json or "{}")
        cited = sorted({s["source_name"] for s in meta.get("sources", []) if s.get("cited")})
        out.append(MessageOut(role=m.role, content=m.content, created_at=m.created_at,
                              meta={"intent": meta.get("intent"), "sources": cited}))
    return out


@router.get("/kb")
def kb_overview():
    return kb_admin.kb_overview()


@router.post("/kb/documents")
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    replace: bool = Form(False),
):
    max_bytes = int(get_settings().kb_max_upload_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)  # never buffer more than the limit plus one byte
    try:
        info = await run_in_threadpool(kb_admin.save_upload, file.filename or "", data, replace)
    except kb_admin.KBError as e:
        raise HTTPException(e.status_code, str(e))
    response.status_code = 200 if info["replaced"] else 201
    return info


@router.delete("/kb/documents/{filename}", status_code=204)
def delete_document(filename: str):
    try:
        kb_admin.delete_document(filename)
    except kb_admin.KBError as e:
        raise HTTPException(e.status_code, str(e))
    return Response(status_code=204)


@router.post("/kb/reindex", status_code=202)
def start_reindex():
    try:
        return kb_admin.reindexer.start()
    except kb_admin.KBError as e:
        raise HTTPException(e.status_code, str(e))


@router.get("/kb/reindex")
def reindex_status():
    return kb_admin.reindexer.status()