from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import TEAM_MEMBERS, Category, Priority, Status


class TicketCreate(BaseModel):
    title: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=10, max_length=5000)
    category: Category
    priority: Priority = Priority.MEDIUM
    requester: str = Field(min_length=1, max_length=120)
    conversation_id: str | None = Field(default=None, max_length=64)
    transcript_summary: str | None = Field(default=None, max_length=8000)
    idempotency_key: str | None = Field(default=None, max_length=128)

    @field_validator("title", "description", "requester")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class TicketUpdate(BaseModel):
    status: Status | None = None
    priority: Priority | None = None
    assignee: str | None = None
    resolution_notes: str | None = Field(default=None, max_length=5000)

    @field_validator("assignee")
    @classmethod
    def check_assignee(cls, v: str | None) -> str | None:
        if v is not None and v not in TEAM_MEMBERS:
            raise ValueError(f"assignee must be one of {TEAM_MEMBERS}")
        return v


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticket_id: str
    title: str
    description: str
    category: Category
    priority: Priority
    status: Status
    requester: str
    assignee: str | None
    conversation_id: str | None
    transcript_summary: str | None
    resolution_notes: str | None
    created_at: datetime
    updated_at: datetime
    due_at: datetime | None