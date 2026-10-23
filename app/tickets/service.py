"""Ticket business rules. The LLM never calls this directly; the backend does,
after validating inputs through the Pydantic schemas."""
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.constants import ALLOWED_TRANSITIONS, SLA_HOURS, Status
from app.core.timeutils import utcnow
from app.models.ticket import Ticket
from app.schemas.ticket import TicketCreate, TicketCreateIn, TicketUpdate

class TicketNotFound(Exception):
    pass


class InvalidTransition(Exception):
    pass


class TicketRuleViolation(Exception):
    pass


def _get_by_idempotency_key(db: Session, key: str) -> Ticket | None:
    return db.scalar(select(Ticket).where(Ticket.idempotency_key == key))


def create_ticket(db: Session, data: TicketCreate) -> tuple[Ticket, bool]:
    """Returns (ticket, created). created=False means an existing ticket was
    returned because the idempotency key was already used."""
    if data.idempotency_key:
        existing = _get_by_idempotency_key(db, data.idempotency_key)
        if existing:
            return existing, False

    now = utcnow()
    ticket = Ticket(
        ticket_id=f"TMP-{uuid4().hex}",  # replaced below once the numeric id exists
        title=data.title,
        description=data.description,
        category=data.category.value,
        priority=data.priority.value,
        status=Status.OPEN.value,
        requester=data.requester,
        conversation_id=data.conversation_id,
        transcript_summary=data.transcript_summary,
        idempotency_key=data.idempotency_key,
        created_at=now,
        updated_at=now,
        due_at=now + timedelta(hours=SLA_HOURS[data.priority]),
    )
    db.add(ticket)
    try:
        db.flush()  # assigns ticket.id
        ticket.ticket_id = f"HD-{ticket.id:05d}"
        db.commit()
    except IntegrityError:
        db.rollback()
        if data.idempotency_key:  # lost a race with an identical request
            existing = _get_by_idempotency_key(db, data.idempotency_key)
            if existing:
                return existing, False
        raise
    return ticket, True

def create_ticket_for_user(db: Session, username: str, payload: TicketCreateIn) -> tuple[Ticket, bool]:
    """Single entry point for employee-created tickets (used by the REST API and by chat).
    The requester always comes from the authenticated user, and the idempotency key is
    scoped to that user so two users can never collide on the same key."""
    data = TicketCreate(
        **payload.model_dump(exclude={"idempotency_key"}),
        requester=username,
        idempotency_key=f"{username}:{payload.idempotency_key}" if payload.idempotency_key else None,
    )
    return create_ticket(db, data)


def get_ticket(db: Session, ticket_id: str) -> Ticket:
    ticket = db.scalar(select(Ticket).where(Ticket.ticket_id == ticket_id))
    if ticket is None:
        raise TicketNotFound(ticket_id)
    return ticket


def list_tickets(
    db: Session,
    category: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    assignee: str | None = None,
    q: str | None = None,
) -> list[Ticket]:
    stmt = select(Ticket).order_by(Ticket.created_at.desc(), Ticket.id.desc())
    if category:
        stmt = stmt.where(Ticket.category == category)
    if priority:
        stmt = stmt.where(Ticket.priority == priority)
    if status:
        stmt = stmt.where(Ticket.status == status)
    if assignee:
        stmt = stmt.where(Ticket.assignee == assignee)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(Ticket.ticket_id.ilike(like), Ticket.title.ilike(like)))
    return list(db.scalars(stmt))


def update_ticket(db: Session, ticket_id: str, data: TicketUpdate) -> Ticket:
    ticket = get_ticket(db, ticket_id)
    current = Status(ticket.status)

    if data.status is not None and data.status != current:
        if data.status not in ALLOWED_TRANSITIONS[current]:
            raise InvalidTransition(
                f"Cannot move from '{current.value}' to '{data.status.value}'"
            )
        notes = data.resolution_notes or ticket.resolution_notes
        if data.status == Status.RESOLVED and not (notes and notes.strip()):
            raise TicketRuleViolation("resolution_notes are required to resolve a ticket")
        ticket.status = data.status.value

    if data.priority is not None:
        ticket.priority = data.priority.value
    if data.assignee is not None:
        ticket.assignee = data.assignee
    if data.resolution_notes is not None:
        ticket.resolution_notes = data.resolution_notes.strip() or None

    db.commit()
    return ticket