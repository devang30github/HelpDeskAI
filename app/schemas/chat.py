from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.core.constants import Category, Priority, Status


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=36)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    source_name: str
    title: str
    score: float
    rank: int
    cited: bool


class TicketRefOut(BaseModel):
    ticket_id: str
    title: str
    category: Category
    priority: Priority
    status: Status


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    intent: str
    mode: str
    answerable: bool | None = None
    sources: list[SourceOut] = []
    ticket: TicketRefOut | None = None
    quick_replies: list[str] = []
    debug: dict = {}


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime
    meta: dict = {}


class ConversationOut(BaseModel):
    conversation_id: str
    mode: str
    messages: list[MessageOut]