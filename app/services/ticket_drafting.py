"""LLM-assisted ticket drafting. The LLM only SUGGESTS values; every field is
validated here, and the employee confirms before anything is created."""
import logging
from dataclasses import asdict, dataclass

from app.core.constants import CATEGORY_LABELS, Category, Priority
from app.rag.generation import LLMError
from app.services.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_CATEGORY_LIST = "; ".join(f"{c.value} ({CATEGORY_LABELS[c]})" for c in Category)

DRAFT_SYSTEM_PROMPT = f"""You are a ticket drafting assistant for an internal IT support and facilities helpdesk.
From the conversation, extract the problem the EMPLOYEE wants to report. Use only facts the employee stated; do not use the assistant's advice and do not invent details. Ignore the employee's request to create a ticket itself.
Reply with ONLY a JSON object with these keys:
- "title": a short summary of at most 80 characters, or null
- "description": the employee's problem in 1-4 sentences, including error codes, devices, locations and what they already tried, or null if no problem has been described
- "category": one of the values below, or null if unclear
- "priority": one of low, medium, high, critical, or null if unclear
Category values: {_CATEGORY_LIST}
Priority guide: critical = safety risk or an outage affecting many people; high = the employee cannot work and has no workaround; medium = a problem with a workaround; low = a request or cosmetic issue."""


@dataclass
class TicketDraft:
    title: str | None = None
    description: str | None = None
    category: str | None = None
    priority: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _enum_value(enum_cls, value) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return enum_cls(value.strip().lower()).value
    except ValueError:
        return None


def _text(raw: dict, key: str, minimum: int, maximum: int) -> str | None:
    value = raw.get(key)
    if not isinstance(value, str):
        return None
    value = " ".join(value.split()) if key == "title" else value.strip()
    return value[:maximum] if len(value) >= minimum else None


def sanitize_draft(raw: dict | None) -> TicketDraft:
    raw = raw or {}
    return TicketDraft(
        title=_text(raw, "title", 5, 200),
        description=_text(raw, "description", 10, 5000),
        category=_enum_value(Category, raw.get("category")),
        priority=_enum_value(Priority, raw.get("priority")),
    )


def suggest_draft(transcript: str, llm_factory) -> TicketDraft:
    """Never raises: on any LLM failure the draft is simply empty and the chat
    flow asks the employee for the missing details."""
    messages = [
        {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation:\n{transcript}"},
    ]
    try:
        result = llm_factory().complete(messages)
    except LLMError as e:
        logger.warning("ticket drafting failed (%s); asking the employee instead", type(e).__name__)
        return TicketDraft()
    return sanitize_draft(extract_json_object(result.text))