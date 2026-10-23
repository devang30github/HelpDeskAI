import json
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.rag.generation import get_llm
from app.rag.pipeline import get_pipeline
from app.schemas.chat import ChatRequest, ChatResponse, ConversationOut, MessageOut
from app.services.chat_service import ChatService, ConversationNotFound

router = APIRouter(prefix="/api/chat", tags=["chat"])


def get_chat_service(db: Session = Depends(get_db)) -> ChatService:
    # Factories are lazy: the index and LLM are only touched when a turn needs them.
    return ChatService(db, pipeline_factory=get_pipeline, llm_factory=get_llm)


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    try:
        reply = service.handle(user, payload.message, payload.conversation_id)
    except ConversationNotFound:
        raise HTTPException(404, "Conversation not found")
    return ChatResponse(**asdict(reply))


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    try:
        conv, messages = service.get_conversation(user, conversation_id)
    except ConversationNotFound:
        raise HTTPException(404, "Conversation not found")
    state = json.loads(conv.state_json or "{}")
    return ConversationOut(
        conversation_id=conv.id,
        mode=state.get("mode", "idle"),
        messages=[
            MessageOut(role=m.role, content=m.content, created_at=m.created_at,
                       meta=json.loads(m.meta_json or "{}"))
            for m in messages
        ],
    )